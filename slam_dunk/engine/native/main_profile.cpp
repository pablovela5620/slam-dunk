#include <basalt/vi_estimator/sqrt_keypoint_vio.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <mutex>
using namespace std;
using namespace Eigen;
using namespace basalt;
using V3 = Vector3d;
using Q = Quaterniond;
#include "anchor.h"
struct ImuRec {
    double t;
    V3 w, a;
};
struct Frame {
    int64_t t;
    string fn[2];
};
struct Track {
    unordered_map<size_t, Anchor> anchors;
    cv::Mat img;
    vector<cv::Point2f> xy;
    vector<size_t> ids;
};
Q expq(V3 v) {
    double a = v.norm();
    return a < 1e-12 ? Q::Identity() : Q(AngleAxisd(a, v / a));
}
double envd(const char *n, double d) {
    char *p = getenv(n);
    return p ? atof(p) : d;
}
int main(int argc, char **argv) {
    if (argc < 3)
        return 1;
    string root = argv[1], out = argv[2];
    cv::setNumThreads(1);
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                                int(envd("THREADS", 8)));
    vector<Frame> fs;
    {
        ifstream f(root + "/frames.txt");
        Frame x;
        while (f >> x.t >> x.fn[0] >> x.fn[1])
            fs.push_back(x);
    }
    vector<ImuRec> imu;
    {
        ifstream f(root + "/imu.bin", ios::binary);
        double a[7];
        while (f.read((char *)a, sizeof(a))) {
            imu.push_back({a[0], V3(a[1], a[2], a[3]), V3(a[4], a[5], a[6])});
        }
    }
    if (fs.empty() || imu.empty())
        return 2;
    Calibration<double> cal;
    vector<cv::Mat> lut(2);
    int width[2], height[2];
    vector<vector<double>> origK(2);
    nlohmann::json js;
    {
        ifstream f(root + "/calib.json");
        f >> js;
    }
    for (int c = 0; c < 2; c++) {
        auto j = js[c];
        width[c] = j["resolution"]["width"];
        height[c] = j["resolution"]["height"];
        auto q = j["T_b_s"]["qvec"].get<vector<double>>();
        auto p = j["T_b_s"]["tvec"].get<vector<double>>();
        cal.T_i_c.push_back(Sophus::SE3d(Q(q[3], q[0], q[1], q[2]), V3(p[0], p[1], p[2])));
        auto cam = GenericCamera<double>::fromString("kb4");
        Vector4d intr(240., 240., width[c] / 2., height[c] / 2.);
        cam.setFromInit(intr);
        cal.intrinsics.push_back(cam);
        cal.resolution.emplace_back(width[c], height[c]);
        origK[c] = j["params"].get<vector<double>>();
        lut[c] = cv::Mat(height[c], width[c], CV_32FC2);
        ifstream f(root + "/cam" + to_string(c) + "_lut.bin", ios::binary);
        f.read((char *)lut[c].data, width[c] * height[c] * sizeof(float) * 2);
    }
    cal.imu_update_rate = 1e9 / ((imu.back().t - imu.front().t) / (imu.size() - 1));
    cal.gyro_noise_std.setConstant(envd("GYRO_NOISE", 0.0005));
    cal.accel_noise_std.setConstant(envd("ACC_NOISE", 0.015));
    cal.gyro_bias_std.setConstant(envd("GYRO_RW", 0.000005));
    cal.accel_bias_std.setConstant(envd("ACC_RW", 0.001));
    VioConfig cfg;
    cfg.vio_max_states = int(envd("MAX_STATES", 3));
    cfg.vio_max_kfs = int(envd("MAX_KFS", 10));
    cfg.vio_min_frames_after_kf = int(envd("KF_INTERVAL", 2));
    cfg.vio_max_iterations = int(envd("MAX_ITERS", 4));
    cfg.vio_obs_std_dev = envd("OBS_NOISE", 0.5);
    cfg.vio_obs_huber_thresh = 1.;
    cfg.vio_min_triangulation_dist = .05;
    cfg.vio_debug = false;
    cfg.vio_extended_logging = false;
    cfg.vio_enforce_realtime = false;
    auto vio = make_shared<SqrtKeypointVioEstimator<double>>(V3(0, 0, -9.80665), cal, cfg);
    tbb::concurrent_bounded_queue<PoseVelBiasState<double>::Ptr> oq;
    oq.set_capacity(50);
    vio->out_state_queue = &oq;
    ofstream traj(out);
    traj << setprecision(15);
    atomic<int> nprocessed = 0;
    thread output([&] {
        while (true) {
            PoseVelBiasState<double>::Ptr s;
            oq.pop(s);
            if (!s)
                break;
            auto q = s->T_w_i.unit_quaternion();
            auto p = s->T_w_i.translation();
            traj << s->t_ns << " " << p.transpose() << " " << q.x() << " " << q.y() << " " << q.z()
                 << " " << q.w() << "\n";
            if ((++nprocessed) % 100 == 0)
                traj.flush();
        }
    });
    vio->initialize(V3::Zero(), V3::Zero());
    thread feedimu([&] {
        for (const auto &x : imu) {
            if (vio->finished || x.t > fs.back().t + 20000000)
                break;
            auto d = make_shared<ImuData<double>>();
            d->t_ns = llround(x.t);
            d->gyro = x.w;
            d->accel = x.a;
            vio->imu_data_queue.push(d);
        }
        vio->imu_data_queue.push(nullptr);
    });
    int stride = int(envd("STRIDE", max(1, int(ceil(double(fs.size()) / 12000)))));
    int maxfeatures = int(envd("FEATURES", 160));
    int sendstride = int(envd("SEND_STRIDE", 1));
    cerr << "Input " << fs.size() << " frames, stride " << stride << " max features " << maxfeatures
         << "\n";
    // Keep JPEG decoding off the tracking and estimator threads.
    tbb::concurrent_bounded_queue<cv::Mat> image_queue[2];
    thread readers[2];
    atomic<bool> stop_decode = false;
    for (int c = 0; c < 2; c++) {
        image_queue[c].set_capacity(3);
        readers[c] = thread([&, c] {
            try {
                for (size_t j = 0; j < fs.size() && !stop_decode; j += stride) {
                    cv::Mat im;
                    try {
                        im = cv::imread(fs[j].fn[c], cv::IMREAD_GRAYSCALE);
                    } catch (const cv::Exception &) {
                    }
                    image_queue[c].push(im);
                }
            } catch (const std::exception &) {
            }
        });
    }

    Track tr[2];
    size_t nextid[2] = {0, 0};
    size_t gi = 0;
    int64_t prevt = fs.front().t;
    int fi = 0;
    double flowtime[2] = {0, 0}, detecttime[2] = {0, 0}, othertime[2] = {0, 0}, waittime = 0;
    auto started = chrono::steady_clock::now();
    ofstream obs(root + "/tracks.bin", ios::binary);
    auto ideal = [&](int c, cv::Point2f p) -> cv::Point2f {
        int x = int(floor(p.x)), y = int(floor(p.y));
        if (x < 0 || x + 1 >= width[c] || y < 0 || y + 1 >= height[c])
            return {-1000, -1000};
        float ax = p.x - x, ay = p.y - y;
        auto a = lut[c].at<cv::Vec2f>(y, x) * (1 - ax) + lut[c].at<cv::Vec2f>(y, x + 1) * ax;
        auto b =
            lut[c].at<cv::Vec2f>(y + 1, x) * (1 - ax) + lut[c].at<cv::Vec2f>(y + 1, x + 1) * ax;
        auto v = a * (1 - ay) + b * ay;
        return {v[0], v[1]};
    };
    auto project = [&](int c, V3 d) -> cv::Point2f {
        const auto &k = origK[c];
        double r = hypot(d.x(), d.y());
        double theta = atan2(r, d.z()), tt = theta * theta, td = theta;
        double poly = 1, power = tt;
        for (int i = 4; i < 10; i++) {
            poly += k[i] * power;
            power *= tt;
        }
        td *= poly;
        double u = d.x() * td / max(r, 1e-10), v = d.y() * td / max(r, 1e-10), rr = u * u + v * v;
        return {float(k[0] * (u + k[10] * (rr + 2 * u * u) + 2 * k[11] * u * v + k[12] * rr +
                              k[13] * rr * rr) +
                      k[2]),
                float(k[1] * (v + k[11] * (rr + 2 * v * v) + 2 * k[10] * u * v + k[14] * rr +
                              k[15] * rr * rr) +
                      k[3])};
    };
    for (size_t idx = 0; idx < fs.size(); idx += stride, fi++) {
        cv::Mat decoded[2];
        for (int c = 0; c < 2; c++)
            image_queue[c].pop(decoded[c]);
        const auto &fr = fs[idx];
        Q dq = Q::Identity();
        while (gi + 1 < imu.size() && imu[gi + 1].t <= prevt)
            gi++;
        size_t gj = gi;
        while (gj + 1 < imu.size() && imu[gj].t < fr.t) {
            double a = max(double(prevt), imu[gj].t), b = min(double(fr.t), imu[gj + 1].t);
            if (b > a)
                dq = (dq * expq(imu[gj].w * ((b - a) * 1e-9))).normalized();
            gj++;
        }
        gi = gj > 0 ? gj - 1 : 0;
        auto of = make_shared<OpticalFlowResult>();
        of->t_ns = fr.t;
        of->observations.resize(2);
        tbb::parallel_for(0, 2, [&](int c) {
            auto tstart = chrono::steady_clock::now();
            cv::Mat img = decoded[c];
            if (img.empty())
                return;
            vector<cv::Point2f> pts;
            vector<size_t> ids;
            if (!tr[c].img.empty() && !tr[c].xy.empty()) {
                auto old = tr[c].xy;
                vector<cv::Point2f> next;
                next.reserve(old.size());
                Q rc = cal.T_i_c[c].unit_quaternion();
                Q dc = rc.conjugate() * dq.conjugate() * rc;
                for (auto p : old) {
                    auto u = ideal(c, p);
                    double x = (u.x - width[c] / 2.) / 240., y = (u.y - height[c] / 2.) / 240.,
                           theta = hypot(x, y);
                    V3 ray(theta < 1e-9 ? 0 : sin(theta) * x / theta,
                           theta < 1e-9 ? 0 : sin(theta) * y / theta, cos(theta));
                    auto pp = project(c, dc * ray);
                    if (!isfinite(pp.x) || !isfinite(pp.y))
                        pp = p;
                    next.push_back(pp);
                }
                vector<uchar> status, backstatus;
                vector<float> err, berr;
                cv::calcOpticalFlowPyrLK(
                    tr[c].img, img, old, next, status, err, cv::Size(21, 21), 3,
                    cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, .01),
                    cv::OPTFLOW_USE_INITIAL_FLOW, 1e-4);
                vector<cv::Point2f> back = old;
                cv::calcOpticalFlowPyrLK(
                    img, tr[c].img, next, back, backstatus, berr, cv::Size(21, 21), 3,
                    cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 15, .02),
                    cv::OPTFLOW_USE_INITIAL_FLOW, 1e-4);
                for (size_t j = 0; j < old.size(); j++) {
                    auto p = next[j];
                    if (!status[j] || !backstatus[j] ||
                        cv::norm(back[j] - old[j]) > envd("FB", .6) || err[j] > 35 || p.x < 9 ||
                        p.x >= img.cols - 9 || p.y < 9 || p.y >= img.rows - 9)
                        continue;
                    if (envd("ANCHOR_FLOW", 0) > 0) {
                        auto &anc = tr[c].anchors.at(tr[c].ids[j]);
                        auto before = p;
                        if (!anc.refine(img, p, envd("ANCHOR_ERR", .18)) ||
                            cv::norm(p - before) > 2.5)
                            continue;
                    }
                    auto uv = ideal(c, p);
                    if (uv.x < -10 || uv.y < -10 || uv.x > width[c] + 10 || uv.y > height[c] + 10)
                        continue;
                    pts.push_back(p);
                    ids.push_back(tr[c].ids[j]);
                }
            }
            auto tflow = chrono::steady_clock::now();
            flowtime[c] += chrono::duration<double>(tflow - tstart).count();
            cv::Mat mask(img.size(), CV_8U, cv::Scalar(255));
            for (auto p : pts)
                cv::circle(mask, p, int(envd("MIN_DIST", 23)), cv::Scalar(0), -1);
            // Keep the fisheye's dark or invalid corners out of the detector.
            cv::rectangle(mask, cv::Rect(0, 0, img.cols, 9), cv::Scalar(0), -1);
            cv::rectangle(mask, cv::Rect(0, img.rows - 9, img.cols, 9), cv::Scalar(0), -1);
            vector<cv::Point2f> add;
            if (int(pts.size()) < maxfeatures)
                cv::goodFeaturesToTrack(img, add, maxfeatures - pts.size(), .012,
                                        envd("MIN_DIST", 23), mask, 3, false, .04);
            if (!add.empty())
                cv::cornerSubPix(
                    img, add, cv::Size(4, 4), cv::Size(-1, -1),
                    cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, .01));
            auto tdetect = chrono::steady_clock::now();
            detecttime[c] += chrono::duration<double>(tdetect - tflow).count();
            for (auto p : add) {
                auto uv = ideal(c, p);
                if (uv.x < 0 || uv.y < 0 || uv.x > width[c] || uv.y > height[c])
                    continue;
                size_t id = 2 * (nextid[c]++) + size_t(c);
                if (envd("ANCHOR_FLOW", 0) > 0) {
                    Anchor anc(img, p);
                    if (!anc.valid)
                        continue;
                    tr[c].anchors.emplace(id, anc);
                }
                pts.push_back(p);
                ids.push_back(id);
            }
            for (size_t j = 0; j < pts.size(); j++) {
                auto uv = ideal(c, pts[j]);
                Eigen::AffineCompact2f a = Eigen::AffineCompact2f::Identity();
                a.translation() << uv.x, uv.y;
                of->observations[c][ids[j]] = a;
            }
            if (envd("ANCHOR_FLOW", 0) > 0) {
                unordered_set<size_t> live(ids.begin(), ids.end());
                for (auto it = tr[c].anchors.begin(); it != tr[c].anchors.end();) {
                    if (!live.count(it->first))
                        it = tr[c].anchors.erase(it);
                    else
                        ++it;
                }
            }
            tr[c].img = img;
            tr[c].xy = move(pts);
            tr[c].ids = move(ids);
            othertime[c] += chrono::duration<double>(chrono::steady_clock::now() - tdetect).count();
        });
        if (fi % sendstride == 0) {
            for (int c = 0; c < 2; c++)
                for (size_t j = 0; j < tr[c].xy.size(); j++) {
                    double v[6] = {double(fr.t),          double(c),
                                   double(tr[c].ids[j]),  double(tr[c].xy[j].x),
                                   double(tr[c].xy[j].y), double(idx)};
                    obs.write((char *)v, sizeof(v));
                }
            auto twait = chrono::steady_clock::now();
            vio->vision_data_queue.push(of);
            waittime += chrono::duration<double>(chrono::steady_clock::now() - twait).count();
        }
        prevt = fr.t;
        if (fi % 500 == 0)
            cerr << "Frame " << idx << "/" << fs.size() << " tracks " << tr[0].ids.size() << ","
                 << tr[1].ids.size() << " elapsed "
                 << chrono::duration<double>(chrono::steady_clock::now() - started).count() << "\n";
        if (vio->finished)
            break;
    }
    cerr << "PROFILE FLOW " << flowtime[0] + flowtime[1] << " DETECT "
         << detecttime[0] + detecttime[1] << " OTHER " << othertime[0] + othertime[1] << " QUEUE "
         << waittime << endl;
    stop_decode = true;
    for (int c = 0; c < 2; c++)
        image_queue[c].abort();
    for (auto &r : readers)
        r.join();
    vio->vision_data_queue.push(nullptr);
    vio->maybe_join();
    vio->drain_input_queues();
    feedimu.join();
    output.join();
    traj.close();
    obs.close();
    cerr << "DONE " << nprocessed << " states in "
         << chrono::duration<double>(chrono::steady_clock::now() - started).count() << "s\n";
}

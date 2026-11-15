#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <functional>
#include <map>
#include <unistd.h>

#include "estimator/geometry.hpp"
#include "estimator/preintegration.hpp"
#include "estimator/visual_factors.hpp"
#include "estimator/stereo.hpp"
#include "estimator/inertial_factors.hpp"
#include "estimator/solver_support.hpp"

#include "earth_inertial.h"

#include <ceres/product_manifold.h>
#include "pose_cross.h"
#include "linear_bias.h"
#include "whiten_visual.h"
#include "recover_tracks.h"
#include "precision_loss.h"
// Driver: cold packet -> chunked calibration -> robust solves -> export sidecars.
int main(int argc, char **argv) {
    auto programstart = chrono::steady_clock::now();
    tbb::global_control parallel_limit(tbb::global_control::max_allowed_parallelism,
                                       max(1, int(envd("THREADS", 4))));
    if (argc < 3)
        return 1;
    ifstream convention(string(argv[1]) + ".stereo");
    string schema;
    convention >> schema;
    if (schema != "stereo-reference-v1") {
        cerr << "Expected a stereo-reference-v1 packet; rerun prepare." << endl;
        return 2;
    }
    // Packet header: poses, landmarks, observations, IMU samples.
    ifstream f(argv[1], ios::binary);
    int64_t H[4];
    f.read((char *)H, sizeof(H));
    int N = H[0], P = H[1], O = H[2], I = H[3];
    double K[32], ext[14];
    rd(f, K, 32);
    rd(f, ext, 14);
    const StereoReference stereo_reference(ext);
    vector<double> times(N);
    rd(f, times.data(), N);
    vector<Pose> poses(N);
    for (auto &s : poses) {
        rd(f, s.q, 4);
        rd(f, s.p, 3);
    }
    for (auto &s : poses)
        rd(f, s.v, 3);
    vector<array<double, 3>> points(P);
    for (auto &p : points)
        rd(f, p.data(), 3);
    vector<array<double, 5>> obs(O);
    for (auto &o : obs)
        rd(f, o.data(), 5);
    imu.resize(I);
    for (auto &s : imu) {
        double x[7];
        rd(f, x, 7);
        s.t = x[0];
        s.w = Map<V3>(x + 1);
        s.a = Map<V3>(x + 4);
    }
    cerr << "LOADED " << N << " " << P << " " << O << " " << I << endl;
    // Relative information for independently tracked feature families.
    vector<double> point_weights(P, 1.0);
    vector<double> observation_weights(O, 1.);
    if (getenv("SCALE_WEIGHT")) {
        ifstream wf(string(argv[1]) + ".obsweights", ios::binary);
        if (wf)
            rd(wf, observation_weights.data(), O);
    }
    ifstream weights_file(string(argv[1]) + ".weights", ios::binary);
    if (weights_file)
        rd(weights_file, point_weights.data(), P);
    for (double &weight : point_weights)
        if (!isfinite(weight) || weight <= 0 || weight > 1)
            weight = 1;

    vector<array<double, 7>> metadata(N);
    double heights[2] = {480, 480};
    bool usepls = getenv("PLS_REFINE") || getenv("PLS_FIXED") || getenv("PLS_FROZEN");
    bool usetemp = getenv("THERMAL") != nullptr;
    if (usepls || usetemp || getenv("BIAS_THERMAL")) {
        ifstream meta(string(argv[1]) + ".meta", ios::binary);
        if (!meta) {
            cerr << "Missing image metadata" << endl;
            return 3;
        }
        rd(meta, heights, 2);
        for (auto &x : metadata)
            rd(meta, x.data(), 7);
    }
    if (usetemp) {
        for (auto &m : metadata)
            gyroTref += .5 * (m[5] + m[6]) / N;
        size_t j = 0;
        for (auto &z : imu) {
            while (j + 2 < times.size() && times[j + 1] < z.t)
                j++;
            double w =
                clamp((z.t - times[j]) / (times[min(j + 1, times.size() - 1)] - times[j]), 0., 1.);
            double a = .5 * (metadata[j][5] + metadata[j][6]),
                   b = .5 * (metadata[min(j + 1, times.size() - 1)][5] +
                             metadata[min(j + 1, times.size() - 1)][6]);
            z.temp = (1 - w) * a + w * b - gyroTref;
        }
    }
    vector<array<double, 2>> camera_offsets(N, {0, 0});
    {
        ifstream offsets(string(argv[1]) + ".camtime", ios::binary);
        if (offsets)
            for (auto &x : camera_offsets)
                rd(offsets, x.data(), 2);
    }
    double pls[1] = {envd("PLS_INITIAL", 1)};
    vector<Pose> initialposes = poses;
    if (getenv("SPLIT_STEREO")) {
        vector<int> dup(P, -1);
        for (auto &o : obs)
            if (int(o[1]) == 0) {
                int p = int(o[2]);
                if (dup[p] < 0) {
                    dup[p] = points.size();
                    points.push_back(points[p]);
                }
                o[2] = dup[p];
            }
        P = points.size();
        point_weights.resize(P, 1.0);
    }
    if (getenv("CAMERA_ONLY")) {
        int cc = atoi(getenv("CAMERA_ONLY"));
        obs.erase(remove_if(obs.begin(), obs.end(),
                            [&](const array<double, 5> &o) { return int(o[1]) != cc; }),
                  obs.end());
    }
    vector<int> low(P, N), high(P, -1);
    for (auto &o : obs) {
        int p = int(o[2]), i = int(o[0]);
        low[p] = min(low[p], i);
        high[p] = max(high[p], i);
    }
    if (getenv("SPLIT_STEREO") || getenv("CAMERA_ONLY"))
        obs.erase(remove_if(obs.begin(), obs.end(),
                            [&](const array<double, 5> &o) {
                                int p = int(o[2]);
                                return low[p] >= high[p];
                            }),
                  obs.end());
    vector<bool> used(P, false);
    for (auto &o : obs)
        used[int(o[2])] = true;
    cerr << "ACTIVE OBS " << obs.size() << endl;
    // Initialize per-chunk camera/bias states and global hardware corrections.
    double chunkdt = envd("CHUNK_DT", 30), duration = times.back() - times.front();
    int NC = int(duration / chunkdt) + 1;
    vector<array<double, 4>> stereo_q(NC);
    vector<array<double, 3>> stereo_t(NC), bg(NC), ba(NC);
    vector<int> ci(N);
    for (int k = 0; k < NC; k++) {
        stereo_reference.relative(ext + 7, ext + 11, stereo_q[k].data(), stereo_t[k].data());
        bg[k] = {0, 0, 0};
        ba[k] = {0, 0, 0};
    }
    for (int i = 0; i < N; i++)
        ci[i] = min(NC - 1, int((times[i] - times.front()) / chunkdt));
    vector<Pre> initial_integrals(N - 1);
    tbb::parallel_for(0, N - 1, [&](int i) {
        initial_integrals[i] = integ(times[i], times[i + 1], V3::Zero(), V3::Zero());
    });
    V3 av = V3::Zero();
    double total = 0;
    for (int i = 0; i < N - 1; i++) {
        const auto &z = initial_integrals[i];
        av += poses[i].rot() * z.v;
        total += z.dt;
    }
    V3 grav = -av.normalized() * 9.806;
    double g[3] = {grav.x(), grav.y(), grav.z()};
    double B = (Map<V3>(ext + 4) - Map<V3>(ext + 11)).norm();
    double tm[2] = {0, envd("ACC_TIME_INITIAL", 0)}, tinitial = tm[1];
    const auto baseimu = imu;
    vector<double> chunk_temperature(NC, 0), chunk_count(NC, 0);
    for (int i = 0; i < N; i++) {
        chunk_temperature[ci[i]] += .5 * (metadata[i][5] + metadata[i][6]);
        chunk_count[ci[i]]++;
    }
    for (int k = 0; k < NC; k++)
        chunk_temperature[k] = chunk_count[k] > 0 ? chunk_temperature[k] / chunk_count[k]
                                                  : (k ? chunk_temperature[k - 1] : 0);
    double bias_temp[3] = {0};
    double earth[3] = {0};
    double gm[9] = {0};
    double gt[3] = {0};
    double ar[3] = {0, 0, 0}, sc[6] = {0, 0, 0, 0, 0, 0}, foc[4] = {K[0], K[1], K[16], K[17]},
           intr[12] = {0};
    double gs = envd("SIGMA_G", .0025),
           as = envd("SIGMA_A", .10); // R10: preserve the reference two-pass estimate, then refine
                                      // with calibration fixed.
    bool final_refine = getenv("FINAL_REFINE") != nullptr;
    int outer = final_refine ? 3 : int(envd("OUTER", 2));
    vector<array<double, 4>> fcs(NC);
    for (int k = 0; k < NC; k++)
        fcs[k] = {K[0], K[1], K[16], K[17]};
    double fprior = envd("FOCAL_PRIOR", .5) * sqrt(double(NC)), frw = envd("FOCAL_RW", .05);
    // The gyro series already carries the current camera-IMU clock estimate.
    // Freeze each observation's short motion integration at this outer pass.
    auto observation_rotation = [&](double t0, double t1, V3 bias) -> M3 {
        bool reverse = t1 < t0;
        if (reverse)
            swap(t0, t1);
        Q q = Q::Identity();
        bool midpoint = getenv("MIDPOINT") != nullptr;
        auto it =
            lower_bound(imu.begin(), imu.end(), t0, [](const Imu &a, double b) { return a.t < b; });
        if (it != imu.begin())
            --it;
        for (; it != imu.end() && it->t < t1; ++it) {
            auto next = it + 1;
            double a = max(t0, it->t), b = min(t1, next == imu.end() ? t1 : next->t);
            if (b <= a)
                continue;
            V3 raw = it->w;
            double temp = it->temp;
            if (midpoint && next != imu.end()) {
                double f = (.5 * (a + b) - it->t) / (next->t - it->t);
                raw = (1 - f) * it->w + f * next->w;
                temp = (1 - f) * it->temp + f * next->temp;
            }
            V3 w = raw + Map<const M3r>(gm) * raw + temp * Map<V3>(gt).cwiseProduct(raw) - bias;
            q = (q * expq(w * (b - a))).normalized();
        }
        return (reverse ? q : q.conjugate()).toRotationMatrix();
    };
    // Camera-only sharing: bias ci/chunkdt/NC remain exactly the Full five-second basis.
    const bool static_camera = getenv("FACTORIAL_STATIC_CAMERA") != nullptr;
    auto camera_index = [&](int k) { return static_camera ? 0 : k; };
    vector<int> camera_ci = ci;
    if (static_camera)
        fill(camera_ci.begin(), camera_ci.end(), 0);
    double robust_scale = envd("SOFT_CAUCHY", 1.);
    double warmradius = 1e4;
    vector<array<double, 3>> initialpoints = points;
    // Reintegrate and fit each pass in the original robust/final schedule.
    for (int pass = 0; pass < outer; pass++) {
        shift_imu(baseimu, tm[0], tm[1]);
        // Stateless losses are shared between observations. Ceres owns residuals,
        // while these owners outlive the problem and keep its loss pointers valid.
        unique_ptr<ceres::LossFunction> visual_base(
            (pass > 0 && getenv("SOFT_CAUCHY"))
                ? static_cast<ceres::LossFunction *>(new ceres::CauchyLoss(robust_scale))
                : static_cast<ceres::LossFunction *>(new ceres::HuberLoss(envd("HUBER_PX", 1.))));
        map<double, unique_ptr<ceres::LossFunction>> scaled_losses;
        vector<ceres::LossFunction *> visual_losses(P, visual_base.get());
        for (int p = 0; p < P; p++)
            if (point_weights[p] != 1.) {
                auto &loss = scaled_losses[point_weights[p]];
                if (!loss)
                    loss = make_unique<ceres::ScaledLoss>(visual_base.get(), point_weights[p],
                                                          ceres::DO_NOT_TAKE_OWNERSHIP);
                visual_losses[p] = loss.get();
            }
        bool whiten_active = getenv("WHITEN_VISUAL") && (!getenv("LATE_PRECISION") || pass == 2);
        vector<unique_ptr<PrecisionLoss>> precision_losses;
        if (whiten_active)
            precision_losses.reserve(obs.size());
        observation_weights.resize(obs.size(), 1.);
        vector<ceres::LossFunction *> observation_losses(obs.size(), visual_base.get());
        for (size_t j = 0; j < obs.size(); j++) {
            int p = int(obs[j][2]);
            if (whiten_active && observation_weights[j] != 1.) {
                auto loss = make_unique<PrecisionLoss>(visual_base.get(), observation_weights[j],
                                                       point_weights[p]);
                observation_losses[j] = loss.get();
                precision_losses.push_back(move(loss));
                continue;
            }
            double weight =
                point_weights[p] * (getenv("WHITEN_VISUAL") ? 1. : observation_weights[j]);
            if (weight == 1.)
                continue;
            auto &loss = scaled_losses[weight];
            if (!loss)
                loss = make_unique<ceres::ScaledLoss>(visual_base.get(), weight,
                                                      ceres::DO_NOT_TAKE_OWNERSHIP);
            observation_losses[j] = loss.get();
        }
        ceres::Problem::Options problem_options;
        problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        ceres::Problem pr(problem_options);
        auto ordering = make_shared<ceres::ParameterBlockOrdering>();
        for (int p = 0; p < P; p++)
            if (used[p]) {
                pr.AddParameterBlock(points[p].data(), 3);
                if (getenv("POINT_PRIOR"))
                    pr.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<VecPrior, 3, 3>(new VecPrior(
                            Map<V3>(initialpoints[p].data()), envd("POINT_PRIOR", 10000))),
                        nullptr, points[p].data());
                ordering->AddElementToGroup(points[p].data(), 0);
            }
        for (auto &s : poses) {
            pr.AddParameterBlock(s.q, 7,
                                 new ceres::ProductManifold<ceres::QuaternionManifold,
                                                            ceres::EuclideanManifold<3>>());
            pr.AddParameterBlock(s.v, 3);
            ordering->AddElementToGroup(s.q, 1);
            ordering->AddElementToGroup(s.v, 1);
        }
        if (usepls) {
            pr.AddResidualBlock(new ceres::AutoDiffCostFunction<ScalarPrior, 1, 1>(
                                    new ScalarPrior(1, envd("PLS_PRIOR", .5))),
                                nullptr, pls);
            ordering->AddElementToGroup(pls, 1);
            pr.SetParameterLowerBound(pls, 0, 0);
            pr.SetParameterUpperBound(pls, 0, 3);
            if (getenv("PLS_FIXED") || getenv("PLS_FROZEN"))
                pr.SetParameterBlockConstant(pls);
        }
        pr.AddResidualBlock(new ceres::AutoDiffCostFunction<VecPrior, 3, 3>(
                                new VecPrior(V3::Zero(), envd("BIAS_TEMP_SIG", .00005))),
                            nullptr, bias_temp);
        ordering->AddElementToGroup(bias_temp, 1);
        if (!getenv("BIAS_THERMAL"))
            pr.SetParameterBlockConstant(bias_temp);
        pr.AddResidualBlock(new ceres::AutoDiffCostFunction<VecPrior, 3, 3>(
                                new VecPrior(V3::Zero(), envd("EARTH_SIG", .00005))),
                            nullptr, earth);
        ordering->AddElementToGroup(earth, 1);
        if (!getenv("EARTH"))
            pr.SetParameterBlockConstant(earth);
        pr.SetParameterBlockConstant(poses[0].q);
        pr.AddResidualBlock(
            new ceres::AutoDiffCostFunction<VecPrior, 3, 3>(new VecPrior(Map<V3>(poses[0].v), .3)),
            nullptr, poses[0].v);
        for (int k = 0; k < NC; k++) {
            if (!static_camera || k == 0) {
                pr.AddParameterBlock(stereo_q[camera_index(k)].data(), 4,
                                     new StereoRotationManifold(stereo_reference.rotation));
                pr.AddParameterBlock(stereo_t[camera_index(k)].data(), 3,
                                     new StereoTranslationManifold(stereo_reference.rotation));
            }
            ordering->AddElementToGroup(stereo_q[camera_index(k)].data(), 1);
            ordering->AddElementToGroup(stereo_t[camera_index(k)].data(), 1);
            pr.AddResidualBlock(new ceres::AutoDiffCostFunction<NormPrior, 1, 3>(
                                    new NormPrior(V3::Zero(), B, .0005)),
                                nullptr, stereo_t[camera_index(k)].data());
            pr.AddResidualBlock(
                new ceres::AutoDiffCostFunction<VecPrior, 3, 3>(
                    new VecPrior(V3::Zero(), envd("BG_PRIOR", .002) *
                                                 (getenv("BG_NORMALIZE") ? sqrt(double(NC)) : 1.))),
                nullptr, bg[k].data());
            pr.AddResidualBlock(
                new ceres::AutoDiffCostFunction<VecPrior, 3, 3>(new VecPrior(V3::Zero(), .4)),
                nullptr, ba[k].data());
            ordering->AddElementToGroup(bg[k].data(), 1);
            ordering->AddElementToGroup(ba[k].data(), 1);
            if (k > 0) {
                if (!static_camera) {
                    pr.AddResidualBlock(new ceres::AutoDiffCostFunction<RotDiff, 3, 4, 4>(
                                            new RotDiff(envd("EXTR_RW", .001))),
                                        nullptr, stereo_q[camera_index(k - 1)].data(),
                                        stereo_q[camera_index(k)].data());
                    pr.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<Diff, 3, 3, 3>(new Diff(.0005)), nullptr,
                        stereo_t[camera_index(k - 1)].data(), stereo_t[camera_index(k)].data());
                }
                pr.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<BiasTempDiff, 3, 3, 3, 3>(new BiasTempDiff(
                        chunk_temperature[k] - chunk_temperature[k - 1], envd("BIAS_RW", .00015))),
                    nullptr, bg[k - 1].data(), bg[k].data(), bias_temp);
                pr.AddResidualBlock(new ceres::AutoDiffCostFunction<Diff, 3, 3, 3>(
                                        new Diff(envd("ACC_BIAS_RW", .04))),
                                    nullptr, ba[k - 1].data(), ba[k].data());
            }
            if (getenv("FIX_EXTR") || getenv("CAMERA_ONLY")) {
                pr.SetParameterBlockConstant(stereo_q[camera_index(k)].data());
                pr.SetParameterBlockConstant(stereo_t[camera_index(k)].data());
            }
        }
        pr.AddResidualBlock(
            new ceres::AutoDiffCostFunction<NormPrior, 1, 3>(new NormPrior(V3::Zero(), 9.806, .01)),
            nullptr, g);
        ordering->AddElementToGroup(g, 1);
        pr.AddResidualBlock(new ceres::AutoDiffCostFunction<VecPrior, 3, 3>(
                                new VecPrior(V3::Zero(), envd("ACC_ALIGN_SIG", .01))),
                            nullptr, ar);
        ordering->AddElementToGroup(ar, 1);
        if (!getenv("ACC_ALIGN"))
            pr.SetParameterBlockConstant(ar);
        for (int k = 0; k < NC; k++)
            for (int c = 0; c < 2; c++) {
                double *fk = fcs[camera_index(k)].data() + 2 * c;
                pr.AddResidualBlock(new ceres::AutoDiffCostFunction<FocalPrior, 2, 2>(
                                        new FocalPrior(K[16 * c], K[16 * c + 1], fprior)),
                                    nullptr, fk);
                ordering->AddElementToGroup(fk, 1);
                if (!getenv("REFINE_FOCAL"))
                    pr.SetParameterBlockConstant(fk);
                if (getenv("FOCAL_EQUAL"))
                    pr.AddResidualBlock(new ceres::AutoDiffCostFunction<Aspect, 1, 2>(new Aspect()),
                                        nullptr, fk);
                if (k > 0 && !static_camera)
                    pr.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<Diff2, 2, 2, 2>(new Diff2(frw)), nullptr,
                        fcs[camera_index(k - 1)].data() + 2 * c, fk);
            }
        for (int c = 0; c < 2; c++) {
            pr.AddResidualBlock(new ceres::AutoDiffCostFunction<IntrPrior, 6, 6>(new IntrPrior()),
                                nullptr, intr + 6 * c);
            ordering->AddElementToGroup(intr + 6 * c, 1);
            if (getenv("PRINCIPAL_ONLY"))
                pr.SetManifold(intr + 6 * c, new ceres::SubsetManifold(6, {2, 3, 4, 5}));
            else if (!getenv("REFINE_INTR"))
                pr.SetParameterBlockConstant(intr + 6 * c);
        }
        pr.AddResidualBlock(
            new ceres::AutoDiffCostFunction<TimePrior, 2, 2>(new TimePrior(tinitial)), nullptr, tm);
        ordering->AddElementToGroup(tm, 1);
        if (!getenv("TIME_REFINE"))
            pr.SetParameterBlockConstant(tm);
        pr.AddResidualBlock(new ceres::AutoDiffCostFunction<VecPrior, 3, 3>(
                                new VecPrior(V3::Zero(), envd("GYRO_TEMP_SIG", .00003))),
                            nullptr, gt);
        ordering->AddElementToGroup(gt, 1);
        if (!usetemp)
            pr.SetParameterBlockConstant(gt);
        pr.AddResidualBlock(new ceres::AutoDiffCostFunction<GyroPrior, 9, 9>(new GyroPrior(
                                envd("GYRO_PRIOR", .002), envd("GSENS_PRIOR", .00005))),
                            nullptr, gm);
        ordering->AddElementToGroup(gm, 1);
        if (!getenv("GYRO_REFINE"))
            pr.SetParameterBlockConstant(gm);
        pr.AddResidualBlock(new ceres::AutoDiffCostFunction<AccPrior, 6, 6>(new AccPrior(
                                envd("ACC_SCALE_SIG", .003), envd("ACC_CROSS_SIG", .001))),
                            nullptr, sc);
        ordering->AddElementToGroup(sc, 1);
        if (!getenv("ACC_SCALE"))
            pr.SetParameterBlockConstant(sc);
        else if (!getenv("ACC_CROSS"))
            pr.SetManifold(sc, new ceres::SubsetManifold(6, {3, 4, 5}));
        if (final_refine && pass == 2) {
            for (int k = 0; k < NC; k++) {
                pr.SetParameterBlockConstant(stereo_q[camera_index(k)].data());
                pr.SetParameterBlockConstant(stereo_t[camera_index(k)].data());
                for (int c = 0; c < 2; c++)
                    pr.SetParameterBlockConstant(fcs[camera_index(k)].data() + 2 * c);
            }
            for (double *x : {g, ar, sc, gm, gt, tm, bias_temp, earth})
                if (pr.HasParameterBlock(x))
                    pr.SetParameterBlockConstant(x);
            for (int c = 0; c < 2; c++)
                pr.SetParameterBlockConstant(intr + 6 * c);
            if (usepls)
                pr.SetParameterBlockConstant(pls);
            cerr << "FINAL_CALIBRATION_FROZEN pose_velocity_bias_landmark_refinement" << endl;
            if (getenv("FINAL_TIMING_VELOCITY"))
                cerr << "FINAL_TIMING_VELOCITY_REFRESH source=main_solve_velocity "
                        "frozen_during_final"
                     << endl;
        }
        auto timed_visual = [&](int i, int c, double u, double v,
                                double delay) -> ceres::CostFunction * {
            delay += camera_offsets[i][c];
            V3 velocity = (getenv("OBS_VELOCITY") ||
                           (final_refine && pass == 2 && getenv("FINAL_TIMING_VELOCITY")))
                              ? Map<V3>(poses[i].v).eval()
                              : Map<V3>(initialposes[i].v).eval();
            if (getenv("OBS_MOTION"))
                return new VisualTimed(
                    K + 16 * c, u, v,
                    observation_rotation(times[i], times[i] + delay, Map<V3>(bg[ci[i]].data())),
                    velocity * delay);
            return new VisualTimed(K + 16 * c, u, v, delay,
                                   V3(metadata[i][2], metadata[i][3], metadata[i][4]), velocity);
        };
        bool collect_for_late = getenv("LATE_PRECISION") && getenv("WHITEN_VISUAL") &&
                                getenv("ADAPTIVE_LOSS") && pass == 1;
        vector<ceres::ResidualBlockId> visual_blocks;
        if (collect_for_late)
            visual_blocks.reserve(obs.size());
        for (auto &o : obs) {
            int i = int(o[0]), c = int(o[1]), p = int(o[2]);
            double *eq = c ? stereo_q[camera_index(ci[i])].data() : ext;
            double *et = c ? stereo_t[camera_index(ci[i])].data() : ext + 4;
            ceres::CostFunction *vc;
            if (usepls) {
                double ro = .00912 * clamp(o[4] / heights[c], 0., 1.),
                       ex = max(metadata[i][c], 1e-6),
                       delay = .5 * ro * .01 * (1 + ro / ex) / (1 + .01 * ro / ex);
                V3 w(metadata[i][2], metadata[i][3], metadata[i][4]);
                vc = timed_visual(i, c, o[3], o[4], delay * pls[0]);
            } else
                vc = new VisualAnalytic(K + 16 * c, o[3], o[4]);
            if (c == 1)
                vc = new StereoVisual(vc, stereo_reference);
            ceres::CostFunction *pose_cost = new PoseVisual(vc);
            auto visual_block = pr.AddResidualBlock(
                pose_cost, observation_losses[&o - obs.data()], poses[i].q, points[p].data(), eq,
                et, fcs[camera_index(ci[i])].data() + 2 * c, intr + 6 * c);
            if (collect_for_late)
                visual_blocks.push_back(visual_block);
        }
        if (pr.HasParameterBlock(ext))
            pr.SetParameterBlockConstant(ext);
        if (pr.HasParameterBlock(ext + 4))
            pr.SetParameterBlockConstant(ext + 4);
        // Independent integrations keep their original summation order internally.
        vector<Pre> integrals;
        if (!getenv("BIAS_LINEAR")) {
            if (pass == 0 && tm[0] == 0 && tm[1] == 0)
                integrals = initial_integrals;
            else {
                integrals.resize(N - 1);
                tbb::parallel_for(0, N - 1, [&](int i) {
                    int k = ci[i];
                    integrals[i] =
                        integ(times[i], times[i + 1], Map<V3>(bg[k].data()), Map<V3>(ba[k].data()),
                              Map<V3>(ar), Map<V6>(sc), gm, tm, gt);
                });
            }
        }
        for (int i = 0; i < N - 1; i++) {
            int k = ci[i], kb = k, kc = k;
            double bw = 0;
            if (getenv("BIAS_LINEAR")) {
                double x = (.5 * (times[i] + times[i + 1]) - times.front()) / chunkdt - .5;
                kb = clamp(int(floor(x)), 0, NC - 1);
                kc = clamp(kb + 1, 0, NC - 1);
                if (x < 0)
                    kc = kb;
                bw = clamp(x - kb, 0., 1.);
            }
            V3 effective = (1 - bw) * Map<V3>(bg[kb].data()) + bw * Map<V3>(bg[kc].data());
            auto z = integrals.empty()
                         ? integ(times[i], times[i + 1], effective, Map<V3>(ba[k].data()),
                                 Map<V3>(ar), Map<V6>(sc), gm, tm, gt)
                         : integrals[i];
            auto *cost = new PoseInertial(new EarthInertial(
                getenv("AUTODIFF")
                    ? static_cast<ceres::CostFunction *>(
                          new ceres::AutoDiffCostFunction<Inertial, 9, 4, 3, 3, 4, 3, 3, 3, 3, 3, 3,
                                                          6, 9, 2, 3>(new Inertial(z, gs, as)))
                    : static_cast<ceres::CostFunction *>(new InertialAnalytic(z, gs, as)),
                z.dt));
            if (kb == kc)
                pr.AddResidualBlock(cost, nullptr, poses[i].q, poses[i].v, poses[i + 1].q,
                                    poses[i + 1].v, bg[kb].data(), ba[k].data(), g, ar, sc, gm, tm,
                                    gt, earth);
            else
                pr.AddResidualBlock(new LinearGyroBias(cost, bw), nullptr, poses[i].q, poses[i].v,
                                    poses[i + 1].q, poses[i + 1].v, bg[kb].data(), ba[k].data(), g,
                                    ar, sc, gm, tm, gt, earth, bg[kc].data());
        }
        auto save_snapshot = [&]() {
            string path = argv[2];
            auto put = [&](const string &suffix, auto render) {
                string dst = path + suffix, tmp = dst + ".tmp";
                ofstream os(tmp);
                os << setprecision(15);
                render(os);
                os.close();
                rename(tmp.c_str(), dst.c_str());
            };
            put(".vel", [&](ostream &o) {
                for (int i = 0; i < N; i++)
                    o << llround(times[i] * 1e9) << " " << Map<V3>(poses[i].v).transpose() << "\n";
            });
            put(".gravity", [&](ostream &o) { o << Map<V3>(g).transpose() << "\n"; });
            put(".timing", [&](ostream &o) { o << tm[0] << " " << tm[1] << "\n"; });
            put(".gyromat", [&](ostream &o) { o << Map<M3r>(gm) << "\n"; });

            put(".gyrot",
                [&](ostream &o) { o << Map<V3>(gt).transpose() << " " << gyroTref << "\n"; });
            put(".accalign", [&](ostream &o) { o << Map<V3>(ar).transpose() << "\n"; });
            put(".accscale", [&](ostream &o) { o << Map<V6>(sc).transpose() << "\n"; });
            put(".diag", [&](ostream &o) {
                for (int k = 0; k < NC; k++)
                    o << k << " " << Map<V3>(bg[k].data()).transpose() << " "
                      << Map<V3>(ba[k].data()).transpose() << " " << stereo_q[camera_index(k)][0] << " "
                      << stereo_q[camera_index(k)][1] << " " << stereo_q[camera_index(k)][2] << " "
                      << stereo_q[camera_index(k)][3] << " "
                      << Map<V3>(stereo_t[camera_index(k)].data()).transpose() << "\n";
            });
            // Publish the poses last; every supporting file is already complete.
            put("", [&](ostream &o) {
                for (int i = 0; i < N; i++) {
                    auto &s = poses[i];
                    o << llround(times[i] * 1e9) << " " << Map<V3>(s.p).transpose() << " " << s.q[1]
                      << " " << s.q[2] << " " << s.q[3] << " " << s.q[0] << "\n";
                }
            });
        };
        PoseMonitor monitor(poses);
        monitor.snapshot = save_snapshot;
        monitor.interval = envd("CHECKPOINT_INTERVAL", 5);
        ceres::Solver::Options op;
        op.update_state_every_iteration = true;
        op.callbacks.push_back(&monitor);
        op.max_num_iterations = int(envd("ITERS", 30));
        op.max_solver_time_in_seconds = envd("MAX_TIME", 240);
        if (getenv("TOTAL_TIME")) {
            double remaining =
                envd("TOTAL_TIME", 240) -
                chrono::duration<double>(chrono::steady_clock::now() - programstart).count() - 3;
            op.max_solver_time_in_seconds =
                max(1., remaining * ((final_refine ? pass == 0 : pass + 1 < outer)
                                         ? envd("FIRST_FRACTION", .55)
                                         : 1.));
        }
        op.num_threads = int(envd("THREADS", 4));
        op.linear_solver_type = ceres::SPARSE_SCHUR;
        op.linear_solver_ordering = ordering;
        op.check_gradients = getenv("CHECK_GRADIENTS") != nullptr;
        op.gradient_check_relative_precision = 1e-6;
        op.parameter_tolerance = envd("PARAM_TOL", 1e-8);
        op.function_tolerance = envd("FUN_TOL", 1e-8);
        op.initial_trust_region_radius =
            pass ? (getenv("RADIUS_CARRY") ? warmradius : envd("REFINE_RADIUS", 1e4))
                 : envd("INITIAL_RADIUS", 1e4);
        op.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary sm;
        const auto pass_started = chrono::steady_clock::now();
        const double original_pass_budget = op.max_solver_time_in_seconds;
        ceres::Solve(op, &pr, &sm);
        const double required_body_step = (pass == 2 ? .001 : .0001),
                     required_angle_step = (pass == 2 ? .0001 : .00001);
        const bool failed_existing_stage_gate =
            final_refine && pass >= 1 &&
            !(monitor.max_step < required_body_step && monitor.max_angle < required_angle_step);
        const bool parameter_stop = sm.termination_type == ceres::CONVERGENCE &&
                                    sm.message.find("Parameter tolerance reached") != string::npos;
        const bool small_unsettled_iteration_stop =
            sm.termination_type == ceres::NO_CONVERGENCE && op.max_num_iterations == 250 &&
            sm.message.find("Maximum number of iterations reached") != string::npos &&
            monitor.max_step <= .05;
        if (getenv("PARAMETER_CONTINUATION") && failed_existing_stage_gate &&
            (parameter_stop || small_unsettled_iteration_stop)) {
            cerr << "PASS_CONTINUATION_TRIGGER pass=" << pass << " cause="
                 << (parameter_stop ? "parameter_tolerance" : "small_unsettled_iteration_cap")
                 << endl;
            const double pass_left =
                original_pass_budget -
                chrono::duration<double>(chrono::steady_clock::now() - pass_started).count() - .5;
            const double native_left =
                envd("TOTAL_TIME", 1200) -
                chrono::duration<double>(chrono::steady_clock::now() - programstart).count() - 3.;
            const double continuation_budget = min(pass_left, native_left);
            if (continuation_budget > 10.) {
                cerr << "PASS_CONTINUATION_PRIMARY pass=" << pass
                     << " body_max_step_m=" << monitor.max_step
                     << " remaining_pass_seconds=" << pass_left
                     << " remaining_native_seconds=" << native_left << endl;
                cerr << "POSE_STABILITY " << monitor.max_step << " " << monitor.mean_step << " "
                     << monitor.max_angle << endl;
                cerr << "STOP_REASON " << sm.message << endl;
                cerr << sm.BriefReport() << endl;
                vector<double *> saved_blocks;
                pr.GetParameterBlocks(&saved_blocks);
                vector<vector<double>> saved_values;
                for (double *block : saved_blocks)
                    saved_values.emplace_back(block, block + pr.ParameterBlockSize(block));
                const auto primary_summary = sm;
                const double primary_max = monitor.max_step, primary_mean = monitor.mean_step,
                             primary_angle = monitor.max_angle;
                const string capture_prefix = string(argv[2]) + ".continuation-" +
                                              to_string(getpid()) + "-pass" + to_string(pass);
                auto capture_parameters = [&](const string &kind) {
                    ofstream capture(capture_prefix + "-" + kind + ".bin", ios::binary);
                    uint64_t n = saved_blocks.size();
                    capture.write((const char *)&n, sizeof(n));
                    for (double *block : saved_blocks) {
                        uint64_t identity = reinterpret_cast<uintptr_t>(block),
                                 width = pr.ParameterBlockSize(block);
                        capture.write((const char *)&identity, sizeof(identity));
                        capture.write((const char *)&width, sizeof(width));
                        capture.write((const char *)block, width * sizeof(double));
                    }
                };
                capture_parameters("primary");
                const double actual_continuation_budget =
                    min(continuation_budget,
                        min(original_pass_budget -
                                chrono::duration<double>(chrono::steady_clock::now() - pass_started)
                                    .count() -
                                .25,
                            envd("TOTAL_TIME", 1200) -
                                chrono::duration<double>(chrono::steady_clock::now() - programstart)
                                    .count() -
                                3.));
                if (actual_continuation_budget > 10.) {
                    op.parameter_tolerance = 0.;
                    op.max_num_iterations = 2000;
                    op.max_solver_time_in_seconds = actual_continuation_budget;
                    monitor.stable = 0;
                    cerr << "PASS_CONTINUATION_START pass=" << pass
                         << " source=current_in_memory_state parameter_tolerance=0 "
                            "max_iterations=2000 required_body_step="
                         << required_body_step << " required_angle_step=" << required_angle_step
                         << " max_seconds=" << actual_continuation_budget << endl;
                    ceres::Solve(op, &pr, &sm);
                    capture_parameters("candidate");
                    bool accepted = (sm.termination_type == ceres::CONVERGENCE ||
                                     sm.termination_type == ceres::USER_SUCCESS) &&
                                    isfinite(sm.final_cost) && sm.final_cost <= sm.initial_cost &&
                                    monitor.max_step < required_body_step &&
                                    monitor.max_angle < required_angle_step;
                    for (double *block : saved_blocks)
                        for (int j = 0; j < pr.ParameterBlockSize(block); j++)
                            accepted = accepted && isfinite(block[j]);
                    cerr << "PASS_CONTINUATION_CANDIDATE_STOP pass=" << pass
                         << " message=" << sm.message << endl;
                    cerr << "PASS_CONTINUATION_CANDIDATE_REPORT pass=" << pass << " "
                         << sm.BriefReport() << endl;
                    cerr << "PASS_CONTINUATION_END pass=" << pass << " accepted=" << int(accepted)
                         << " body_max_step_m=" << monitor.max_step
                         << " body_mean_step_m=" << monitor.mean_step << " elapsed_pass_seconds="
                         << chrono::duration<double>(chrono::steady_clock::now() - pass_started)
                                .count()
                         << " parameter_capture_prefix=" << capture_prefix << endl;
                    if (!accepted) {
                        for (size_t j = 0; j < saved_blocks.size(); j++)
                            copy(saved_values[j].begin(), saved_values[j].end(), saved_blocks[j]);
                        sm = primary_summary;
                        monitor.max_step = primary_max;
                        monitor.mean_step = primary_mean;
                        monitor.max_angle = primary_angle;
                        monitor.previous = poses;
                        monitor.stable = 0;
                        cerr << "PASS_CONTINUATION_REJECTED pass=" << pass
                             << " primary_parameter_blocks_restored=" << saved_blocks.size()
                             << endl;
                    }
                } else
                    cerr << "PASS_CONTINUATION_SKIPPED pass=" << pass
                         << " reason=capture_used_original_budget remaining_seconds="
                         << actual_continuation_budget << endl;
            } else
                cerr << "PASS_CONTINUATION_SKIPPED pass=" << pass
                     << " reason=original_budget remaining_seconds=" << continuation_budget << endl;
        }
        if (!sm.iterations.empty())
            warmradius = clamp(sm.iterations.back().trust_region_radius, 1e4, 1e7);
        if (getenv("FULL_REPORT"))
            cerr << sm.FullReport() << endl;
        cerr << "POSE_STABILITY " << monitor.max_step << " " << monitor.mean_step << " "
             << monitor.max_angle << endl;
        for (int z = 0; z < 4; z++) {
            double m = 0, lo = 1e9, hi = -1e9;
            for (int k = 0; k < NC; k++) {
                m += fcs[camera_index(k)][z];
                lo = min(lo, fcs[camera_index(k)][z]);
                hi = max(hi, fcs[camera_index(k)][z]);
            }
            foc[z] = m / NC;
            cerr << "FOCAL " << z << " mean " << foc[z] << " min " << lo << " max " << hi << endl;
        }
        cerr << "STOP_REASON " << sm.message << endl;
        cerr << sm.BriefReport() << " g=" << Map<V3>(g).transpose()
             << " accrot=" << Map<V3>(ar).transpose() << " accscale=" << Map<V6>(sc).transpose()
             << " focal=" << foc[0] << "," << foc[1] << "," << foc[2] << "," << foc[3] << endl;
        cerr << "GYRO_BIAS_TEMP " << Map<V3>(bias_temp).transpose() << endl;
        cerr << "EARTH " << Map<V3>(earth).transpose() << endl;
        cerr << "GYROMAT " << Map<M3r>(gm) << endl;
        string path = argv[2];
        ofstream gravityout(path + ".gravity.tmp");
        gravityout << setprecision(12) << g[0] << " " << g[1] << " " << g[2] << "\n";
        gravityout.close();
        rename((path + ".gravity.tmp").c_str(), (path + ".gravity").c_str());
        ofstream tout(path + ".timing.tmp");
        tout << setprecision(12) << tm[0] << " " << tm[1] << "\n";
        tout.close();
        rename((path + ".timing.tmp").c_str(), (path + ".timing").c_str());
        ofstream plsout(path + ".pls");
        plsout << setprecision(12) << pls[0] << "\n";
        plsout.close();
        cerr << "PLS " << pls[0] << endl;
        cerr << "TIMING " << tm[0] << " " << tm[1] << endl;
        ofstream gtout(path + ".gyrot.tmp");
        gtout << setprecision(12) << gt[0] << " " << gt[1] << " " << gt[2] << " " << gyroTref
              << "\n";
        gtout.close();
        rename((path + ".gyrot.tmp").c_str(), (path + ".gyrot").c_str());
        cerr << "GYRO_THERMAL " << Map<V3>(gt).transpose() << endl;
        ofstream gout(path + ".gyromat.tmp");
        gout << setprecision(12) << Map<M3r>(gm) << "\n";
        gout.close();
        rename((path + ".gyromat.tmp").c_str(), (path + ".gyromat").c_str());
        ofstream out(path + ".tmp");
        out << setprecision(12);
        for (int i = 0; i < N; i++) {
            auto &s = poses[i];
            out << llround(times[i] * 1e9) << " " << s.p[0] << " " << s.p[1] << " " << s.p[2] << " "
                << s.q[1] << " " << s.q[2] << " " << s.q[3] << " " << s.q[0] << "\n";
        }
        out.close();
        ofstream vout(path + ".vel");
        vout << setprecision(15);
        for (int i = 0; i < N; i++)
            vout << llround(times[i] * 1e9) << " " << Map<V3>(poses[i].v).transpose() << "\n";
        vout.close();
        ofstream ico(path + ".intr");
        ico << setprecision(12);
        for (double x : foc)
            ico << x << " ";
        for (double x : intr)
            ico << x << " ";
        ico << "\n";
        ico.close();
        ofstream scout(path + ".accscale.tmp");
        scout << setprecision(12) << Map<V6>(sc).transpose() << "\n";
        scout.close();
        rename((path + ".accscale.tmp").c_str(), (path + ".accscale").c_str());
        ofstream acout(path + ".accalign.tmp");
        acout << setprecision(12) << ar[0] << " " << ar[1] << " " << ar[2] << "\n";
        acout.close();
        rename((path + ".accalign.tmp").c_str(), (path + ".accalign").c_str());
        rename((path + ".tmp").c_str(), path.c_str());
        ofstream diag(path + ".diag.tmp");
        diag << setprecision(10);
        for (int k = 0; k < NC; k++)
            diag << k << " " << Map<V3>(bg[k].data()).transpose() << " "
                 << Map<V3>(ba[k].data()).transpose() << " " << stereo_q[camera_index(k)][0] << " "
                 << stereo_q[camera_index(k)][1] << " " << stereo_q[camera_index(k)][2] << " "
                 << stereo_q[camera_index(k)][3] << " " << Map<V3>(stereo_t[camera_index(k)].data()).transpose()
                 << "\n";
        diag.close();
        rename((path + ".diag.tmp").c_str(), (path + ".diag").c_str());

        if (getenv("SAVE_STATE")) {
            ofstream obout(path + ".observations", ios::binary);
            for (auto &o : obs)
                obout.write((char *)o.data(), 5 * sizeof(double));
            ofstream ptsout(path + ".points", ios::binary);
            for (auto &x : points)
                ptsout.write((char *)x.data(), 3 * sizeof(double));
            ofstream fcsout(path + ".focals");
            fcsout << setprecision(15);
            for (int k = 0; k < NC; k++) {
                const auto &x = fcs[camera_index(k)];
                fcsout << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << "\n";
            }
        }
        if (pass + 1 < outer && (!final_refine || pass == 0) && envd("OUTLIER", 0) > 0 &&
            (!getenv("OUTLIER_GUARD") || sm.termination_type == ceres::CONVERGENCE ||
             (monitor.max_step < envd("OUTLIER_MAX_STEP", .05) && monitor.max_angle < .001))) {
            double threshold = envd("OUTLIER", 4);
            size_t old = obs.size();
            vector<double> visual_errors;
            vector<array<double, 5>> kept;
            vector<double> kept_weights;
            vector<int> cnt(P, 0);
            for (auto &o : obs) {
                int i = int(o[0]), c = int(o[1]), p = int(o[2]);
                double *eq = c ? stereo_q[camera_index(ci[i])].data() : ext;
                double *et = c ? stereo_t[camera_index(ci[i])].data() : ext + 4;
                double body_q[4], body_t[3];
                if (c == 1) {
                    stereo_reference.compose(eq, et, body_q, body_t);
                    eq = body_q;
                    et = body_t;
                }
                Q rc(eq[0], eq[1], eq[2], eq[3]);
                V3 pt = rc.conjugate() * (poses[i].rot().conjugate() *
                                              (Map<V3>(points[p].data()) - Map<V3>(poses[i].p)) -
                                          Map<V3>(et));
                double rr[2];
                if (usepls) {
                    double ro = .00912 * clamp(o[4] / heights[c], 0., 1.),
                           ex = max(metadata[i][c], 1e-6),
                           delay = .5 * ro * .01 * (1 + ro / ex) / (1 + .01 * ro / ex);
                    V3 w(metadata[i][2], metadata[i][3], metadata[i][4]);
                    VisualPLS cost(K + 16 * c, o[3], o[4], delay, w);
                    const double *par[] = {poses[i].q,
                                           poses[i].p,
                                           points[p].data(),
                                           eq,
                                           et,
                                           fcs[camera_index(ci[i])].data() + 2 * c,
                                           intr + 6 * c,
                                           getenv("PLS_VEL_FIXED") ? initialposes[i].v : poses[i].v,
                                           pls};
                    if (getenv("PLS_FROZEN")) {
                        unique_ptr<ceres::CostFunction> fixed(
                            timed_visual(i, c, o[3], o[4], delay * pls[0]));
                        fixed->Evaluate(par, rr, nullptr);
                    } else
                        cost.Evaluate(par, rr, nullptr);
                } else
                    Visual(K + 16 * c, o[3], o[4])(poses[i].q, poses[i].p, points[p].data(), eq, et,
                                                   fcs[camera_index(ci[i])].data() + 2 * c,
                                                   intr + 6 * c, rr);
                if (pt.z() > .05 && isfinite(rr[0]) && isfinite(rr[1]))
                    visual_errors.push_back(hypot(rr[0], rr[1]) *
                                            ((getenv("WHITEN_VISUAL") && !getenv("LATE_PRECISION"))
                                                 ? sqrt(observation_weights[&o - obs.data()])
                                                 : 1.));
                if (pt.z() > .05 && isfinite(rr[0]) && isfinite(rr[1]) &&
                    hypot(rr[0], rr[1]) <=
                        threshold +
                            (getenv("BLUR_OUTLIER") && usepls
                                 ? envd("BLUR_OUTLIER", .5) * metadata[i][c] * K[16 * c] *
                                       V3(metadata[i][2], metadata[i][3], metadata[i][4]).norm()
                                 : 0.)) {
                    kept.push_back(o);
                    kept_weights.push_back(observation_weights[&o - obs.data()]);
                    cnt[p]++;
                }
            }
            if (getenv("ADAPTIVE_LOSS") && visual_errors.size() > 100) {
                size_t mid = visual_errors.size() / 2;
                nth_element(visual_errors.begin(), visual_errors.begin() + mid,
                            visual_errors.end());
                robust_scale = clamp(2.3849 * visual_errors[mid] / sqrt(2 * log(2.)),
                                     (getenv("LATE_PRECISION") ? .6 : envd("ADAPTIVE_MIN", .6)),
                                     envd("ADAPTIVE_MAX", 1.5));
                cerr << "LOSS_SCALE " << robust_scale << " median_residual " << visual_errors[mid]
                     << endl;
            }
            obs.clear();
            observation_weights.clear();
            fill(used.begin(), used.end(), false);
            for (size_t j = 0; j < kept.size(); j++) {
                auto &o = kept[j];
                if (cnt[int(o[2])] >= int(envd("KEEP_OBS", 3))) {
                    obs.push_back(o);
                    observation_weights.push_back(kept_weights[j]);
                    used[int(o[2])] = true;
                }
            }
            cerr << "OUTLIERS removed " << old - obs.size() << " remain " << obs.size() << endl;
        }
        if (final_refine && pass == 1) {
            vector<string> suffixes = {"",         ".vel",   ".gravity",  ".timing",
                                       ".gyromat", ".gyrot", ".accalign", ".accscale",
                                       ".diag",    ".intr",  ".pls"};
            for (auto &suffix : suffixes) {
                ifstream a(string(argv[2]) + suffix, ios::binary);
                if (a) {
                    ofstream b(string(argv[2]) + ".base" + suffix, ios::binary);
                    b << a.rdbuf();
                }
            }
            double remaining =
                envd("TOTAL_TIME", 1200) -
                chrono::duration<double>(chrono::steady_clock::now() - programstart).count();
            if (remaining < 30) {
                cerr << "FINAL_SELECTION baseline reason=remaining_budget " << remaining << endl;
                break;
            }
            if (!(monitor.max_step < .0001 && monitor.max_angle < .00001)) {
                cerr << "FINAL_SELECTION baseline reason=unsettled_main_solve" << endl;
                break;
            }
            if (getenv("LATE_PRECISION") && getenv("WHITEN_VISUAL") && getenv("ADAPTIVE_LOSS")) {
                vector<double> standardized;
                standardized.reserve(obs.size());
                for (size_t j = 0; j < visual_blocks.size(); j++) {
                    double rr[2], cost = 0;
                    if (pr.EvaluateResidualBlock(visual_blocks[j], false, &cost, rr, nullptr)) {
                        double e = hypot(rr[0], rr[1]) * sqrt(observation_weights[j]);
                        if (isfinite(e))
                            standardized.push_back(e);
                    }
                }
                if (standardized.size() > 100) {
                    size_t mid = standardized.size() / 2;
                    nth_element(standardized.begin(), standardized.begin() + mid,
                                standardized.end());
                    robust_scale = clamp(2.3849 * standardized[mid] / sqrt(2 * log(2.)),
                                         envd("ADAPTIVE_MIN", .45), envd("ADAPTIVE_MAX", 1.5));
                    cerr << "LATE_LOSS_SCALE " << robust_scale << " standardized_median "
                         << standardized[mid] << endl;
                }
            }
            size_t before = obs.size();
            setenv("RECOVER_OLD_ONLY", "1", 1);
            setenv("RECOVER_GATE", ".8", 1);
            recover_tracks(argv[1], poses, points, obs, camera_ci, stereo_q, stereo_t, fcs, K, ext, intr,
                           metadata, heights, usepls, pls[0], initialposes, used,
                           observation_weights, getenv("FINAL_TIMING_VELOCITY") != nullptr);
            cerr << "FINAL_ADDED_OBSERVATIONS " << obs.size() - before << " remaining_seconds "
                 << remaining << endl;
            if (obs.size() == before) {
                cerr << "FINAL_SELECTION baseline reason=no_additions" << endl;
                break;
            }
            P = points.size();
            initialpoints = points;
            point_weights.resize(P, 1.);
            observation_weights.resize(obs.size(), 1.);
        }
        if (final_refine && pass == 2) {
            bool ok = sm.termination_type != ceres::FAILURE && isfinite(sm.final_cost) &&
                      sm.final_cost <= sm.initial_cost && monitor.max_step < .001 &&
                      monitor.max_angle < .0001;
            for (auto &p : poses) {
                for (double x : p.p)
                    ok = ok && isfinite(x);
                for (double x : p.q)
                    ok = ok && isfinite(x);
                for (double x : p.v)
                    ok = ok && isfinite(x);
                ok = ok && abs(p.rot().norm() - 1) < 1e-6;
            }
            if (!ok) {
                vector<string> suffixes = {"",         ".vel",   ".gravity",  ".timing",
                                           ".gyromat", ".gyrot", ".accalign", ".accscale",
                                           ".diag",    ".intr",  ".pls"};
                for (auto &suffix : suffixes) {
                    ifstream a(string(argv[2]) + ".base" + suffix, ios::binary);
                    if (a) {
                        ofstream b(string(argv[2]) + suffix, ios::binary);
                        b << a.rdbuf();
                    }
                }
            }
            cerr << "FINAL_SELECTION " << (ok ? "refined" : "baseline")
                 << " reason=" << (ok ? "accepted_stable_cost_decrease" : "failed_refinement")
                 << endl;
        }
        if (pass == 0 && getenv("RECOVER")) {
            recover_tracks(argv[1], poses, points, obs, camera_ci, stereo_q, stereo_t, fcs, K, ext, intr,
                           metadata, heights, usepls, pls[0], initialposes, used,
                           observation_weights);
            P = points.size();
            initialpoints = points;
            point_weights.resize(P, 1.);
            observation_weights.resize(obs.size(), 1.);
        }
    }
    return 0;
}

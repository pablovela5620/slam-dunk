#pragma once

struct AccPrior {
    double a, b;
    AccPrior(double aa, double bb) : a(aa), b(bb) {
    }
    template <class T> bool operator()(const T *x, T *r) const {
        for (int k = 0; k < 6; k++)
            r[k] = x[k] / T(k < 3 ? a : b);
        return true;
    }
};
struct GyroPrior {
    double sig, gsig;
    GyroPrior(double s, double g) : sig(s), gsig(g) {
    }
    template <class T> bool operator()(const T *x, T *r) const {
        for (int k = 0; k < 9; k++)
            r[k] = x[k] / T(sig);
        return true;
    }
};
struct TimePrior {
    double init;
    TimePrior(double a) : init(a) {
    }
    template <class T> bool operator()(const T *x, T *r) const {
        r[0] = x[0] / T(.001);
        r[1] = (x[1] - T(init)) / T(.001);
        return true;
    }
};
void shift_imu(const vector<Imu> &base, double tg, double ta) {
    size_t jg = 0, ja = 0;
    for (size_t i = 0; i < base.size(); i++) {
        double x = base[i].t + tg, y = x + ta;
        while (jg + 2 < base.size() && base[jg + 1].t < x)
            jg++;
        while (ja + 2 < base.size() && base[ja + 1].t < y)
            ja++;
        double wg = clamp((x - base[jg].t) / (base[jg + 1].t - base[jg].t), 0., 1.),
               wa = clamp((y - base[ja].t) / (base[ja + 1].t - base[ja].t), 0., 1.);
        imu[i].w = (1 - wg) * base[jg].w + wg * base[jg + 1].w;
        imu[i].a = (1 - wa) * base[ja].a + wa * base[ja + 1].a;
        imu[i].temp = (1 - wg) * base[jg].temp + wg * base[jg + 1].temp;
    }
}
struct Diff2 {
    double sig;
    Diff2(double s) : sig(s) {
    }
    template <class T> bool operator()(const T *a, const T *b, T *r) const {
        for (int k = 0; k < 2; k++)
            r[k] = (b[k] - a[k]) / T(sig);
        return true;
    }
};
struct IntrPrior {
    template <class T> bool operator()(const T *d, T *r) const {
        double sig[6] = {.5, .5, .001, .001, .0003, .0003};
        for (int k = 0; k < 6; k++)
            r[k] = d[k] / T(sig[k]);
        return true;
    }
};
struct FocalPrior {
    double a, b, sig;
    FocalPrior(double aa, double bb, double ss) : a(aa), b(bb), sig(ss) {
    }
    template <class T> bool operator()(const T *f, T *r) const {
        r[0] = (f[0] - T(a)) / T(sig);
        r[1] = (f[1] - T(b)) / T(sig);
        return true;
    }
};
struct Aspect {
    template <class T> bool operator()(const T *f, T *r) const {
        r[0] = (f[0] - f[1]) / T(.05);
        return true;
    }
};
void rd(ifstream &f, double *p, size_t n) {
    f.read(reinterpret_cast<char *>(p), n * sizeof(double));
}
double envd(const char *n, double d) {
    char *p = getenv(n);
    return p ? atof(p) : d;
}
struct PoseMonitor : ceres::IterationCallback {
    vector<Pose> &poses;
    vector<Pose> previous;
    std::function<void()> snapshot;
    double interval = 5;
    chrono::steady_clock::time_point last = chrono::steady_clock::now();
    double max_step = 1e9, mean_step = 1e9, max_angle = 1e9;
    int stable = 0;
    explicit PoseMonitor(vector<Pose> &p) : poses(p), previous(p) {
    }
    ceres::CallbackReturnType operator()(const ceres::IterationSummary &s) override {
        if (s.iteration == 0) {
            previous = poses;
            return ceres::SOLVER_CONTINUE;
        }
        if (!s.step_is_successful)
            return ceres::SOLVER_CONTINUE;
        max_step = mean_step = max_angle = 0;
        for (size_t i = 0; i < poses.size(); i++) {
            double d = (Map<V3>(poses[i].p) - Map<V3>(previous[i].p)).norm();
            max_step = max(max_step, d);
            mean_step += d;
            max_angle =
                max(max_angle, 2 * (previous[i].rot().conjugate() * poses[i].rot()).vec().norm());
        }
        mean_step /= poses.size();
        previous = poses;
        if (getenv("POSE_STOP") && s.iteration > 12 && max_step < envd("POSE_STOP_STEP", 1e-5) &&
            max_angle < envd("POSE_STOP_ANGLE", 1e-7)) {
            if (++stable >= 3)
                return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
        } else
            stable = 0;
        if (snapshot && interval > 0 && s.iteration >= 3 &&
            chrono::duration<double>(chrono::steady_clock::now() - last).count() >= interval) {
            snapshot();
            last = chrono::steady_clock::now();
        }
        return ceres::SOLVER_CONTINUE;
    }
};

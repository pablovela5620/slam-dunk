#pragma once

struct Visual {
    array<double, 16> K;
    double u, v;
    Visual(double *kp, double uu, double vv) : u(uu), v(vv) {
        copy(kp, kp + 16, K.begin());
    }
    template <class T>
    bool operator()(const T *q, const T *p, const T *x, const T *qc, const T *tc, const T *ff,
                    const T *dd, T *r) const {
        using V = Matrix<T, 3, 1>;
        Quaternion<T> rb(q[0], q[1], q[2], q[3]), rc(qc[0], qc[1], qc[2], qc[3]);
        V d(x[0] - p[0], x[1] - p[1], x[2] - p[2]);
        V tr(tc[0], tc[1], tc[2]);
        V pt = rc.conjugate() * (rb.conjugate() * d - tr);
        T aa = pt.x() / pt.z(), bb = pt.y() / pt.z(), rho = sqrt(aa * aa + bb * bb + T(1e-16)),
          theta = atan(rho), tt = theta * theta;
        T td =
            theta *
            (T(1) + tt * ((T(K[4]) + dd[2]) +
                          tt * ((T(K[5]) + dd[3]) +
                                tt * (T(K[6]) + tt * (T(K[7]) + tt * (T(K[8]) + tt * T(K[9])))))));
        T un = aa * td / rho, vn = bb * td / rho, rr = un * un + vn * vn;
        T up = un + (T(K[10]) + dd[4]) * (rr + T(2) * un * un) +
               (T(2 * K[11]) + T(2) * dd[5]) * un * vn + T(K[12]) * rr + T(K[13]) * rr * rr;
        T vp = vn + (T(K[11]) + dd[5]) * (rr + T(2) * vn * vn) +
               (T(2 * K[10]) + T(2) * dd[4]) * un * vn + T(K[14]) * rr + T(K[15]) * rr * rr;
        r[0] = ff[0] * up + T(K[2] - u) + dd[0];
        r[1] = ff[1] * vp + T(K[3] - v) + dd[1];
        return true;
    }
};
// Exact camera derivatives without automatic-differentiation temporaries.
void manualProject(const Visual &m, const V3 &pt, const double *ff, const double *dd, double *r,
                   Matrix<double, 2, 3> *J) {
    const auto &K = m.K;
    double aa = pt.x() / pt.z(), bb = pt.y() / pt.z(), rad = sqrt(aa * aa + bb * bb + 1e-16),
           theta = atan(rad), tt = theta * theta;
    double td =
        theta *
        (1 + tt * ((K[4] + dd[2]) +
                   tt * ((K[5] + dd[3]) + tt * (K[6] + tt * (K[7] + tt * (K[8] + tt * K[9]))))));
    double un = aa * td / rad, vn = bb * td / rad, rr = un * un + vn * vn, p1 = K[10] + dd[4],
           p2 = K[11] + dd[5];
    double up = un + p1 * (rr + 2 * un * un) + 2 * p2 * un * vn + K[12] * rr + K[13] * rr * rr;
    double vp = vn + p2 * (rr + 2 * vn * vn) + 2 * p1 * un * vn + K[14] * rr + K[15] * rr * rr;
    r[0] = ff[0] * up + K[2] - m.u + dd[0];
    r[1] = ff[1] * vp + K[3] - m.v + dd[1];
    if (!J)
        return;
    double power = tt, deriv = 1.;
    for (int k = 0; k < 6; k++) {
        deriv += (2 * k + 3) * (K[4 + k] + (k < 2 ? dd[k + 2] : 0)) * power;
        power *= tt;
    }
    Matrix2d D;
    D << 1 + 6 * p1 * un + 2 * p2 * vn + 2 * K[12] * un + 4 * K[13] * rr * un,
        2 * p1 * vn + 2 * p2 * un + 2 * K[12] * vn + 4 * K[13] * rr * vn,
        2 * p2 * un + 2 * p1 * vn + 2 * K[14] * un + 4 * K[15] * rr * un,
        1 + 6 * p2 * vn + 2 * p1 * un + 2 * K[14] * vn + 4 * K[15] * rr * vn;
    D.row(0) *= ff[0];
    D.row(1) *= ff[1];
    Vector2d v(aa, bb);
    Matrix2d Jn = td / rad * Matrix2d::Identity() +
                  (deriv / (1 + rad * rad) - td / rad) / (rad * rad) * (v * v.transpose());
    Matrix<double, 2, 3> persp;
    persp << 1 / pt.z(), 0, -aa / pt.z(), 0, 1 / pt.z(), -bb / pt.z();
    *J = D * Jn * persp;
}

// Chain a small projection derivative through exact rigid-body Jacobians.
// This avoids propagating 25-component autodiff jets through each observation.
class VisualAnalytic : public ceres::SizedCostFunction<2, 4, 3, 3, 4, 3, 2, 6> {
  public:
    Visual visual;
    static Matrix<double, 4, 3> plus(const double *q) {
        Matrix<double, 4, 3> J;
        J << -q[1], -q[2], -q[3], q[0], q[3], -q[2], -q[3], q[0], q[1], q[2], -q[1], q[0];
        return J;
    }
    template <class T>
    void project(const Matrix<T, 3, 1> &pt, const T *ff, const T *dd, T *r) const {
        const auto &K = visual.K;
        T aa = pt.x() / pt.z(), bb = pt.y() / pt.z(), rho = sqrt(aa * aa + bb * bb + T(1e-16)),
          theta = atan(rho), tt = theta * theta;
        T td =
            theta *
            (T(1) + tt * ((T(K[4]) + dd[2]) +
                          tt * ((T(K[5]) + dd[3]) +
                                tt * (T(K[6]) + tt * (T(K[7]) + tt * (T(K[8]) + tt * T(K[9])))))));
        T un = aa * td / rho, vn = bb * td / rho, rr = un * un + vn * vn;
        T up = un + (T(K[10]) + dd[4]) * (rr + T(2) * un * un) +
               (T(2 * K[11]) + T(2) * dd[5]) * un * vn + T(K[12]) * rr + T(K[13]) * rr * rr;
        T vp = vn + (T(K[11]) + dd[5]) * (rr + T(2) * vn * vn) +
               (T(2 * K[10]) + T(2) * dd[4]) * un * vn + T(K[14]) * rr + T(K[15]) * rr * rr;
        r[0] = ff[0] * up + T(K[2] - visual.u) + dd[0];
        r[1] = ff[1] * vp + T(K[3] - visual.v) + dd[1];
    }

  public:
    VisualAnalytic(double *K, double u, double v) : visual(K, u, v) {
    }
    bool Evaluate(double const *const *a, double *r, double **jac) const override {
        if (jac && jac[6]) {
            ceres::AutoDiffCostFunction<Visual, 2, 4, 3, 3, 4, 3, 2, 6> cost(new Visual(visual));
            return cost.Evaluate(a, r, jac);
        }
        Q rb(a[0][0], a[0][1], a[0][2], a[0][3]), rc(a[3][0], a[3][1], a[3][2], a[3][3]);
        M3 B = rb.toRotationMatrix().transpose(), C = rc.toRotationMatrix().transpose();
        V3 delta = Map<const V3>(a[2]) - Map<const V3>(a[1]),
           body = B * delta - Map<const V3>(a[4]), pt = C * body;
        if (!jac) {
            manualProject(visual, pt, a[5], a[6], r, nullptr);
            return true;
        }
        Matrix<double, 2, 3> G;
        manualProject(visual, pt, a[5], a[6], r, &G);
        if (jac[0]) {
            Map<Matrix<double, 2, 4, RowMajor>> Jout(jac[0]);
            Jout = 2 * G * C * B * skew(delta) * plus(a[0]).transpose();
        }
        if (jac[1]) {
            Map<Matrix<double, 2, 3, RowMajor>> Jout(jac[1]);
            Jout = -G * C * B;
        }
        if (jac[2]) {
            Map<Matrix<double, 2, 3, RowMajor>> Jout(jac[2]);
            Jout = G * C * B;
        }
        if (jac[3]) {
            Map<Matrix<double, 2, 4, RowMajor>> Jout(jac[3]);
            Jout = 2 * G * C * skew(body) * plus(a[3]).transpose();
        }
        if (jac[4]) {
            Map<Matrix<double, 2, 3, RowMajor>> Jout(jac[4]);
            Jout = -G * C;
        }
        if (jac[5]) {
            Map<Matrix<double, 2, 2, RowMajor>> F(jac[5]);
            F.setZero();
            F(0, 0) = (r[0] - visual.K[2] + visual.u - a[6][0]) / a[5][0];
            F(1, 1) = (r[1] - visual.K[3] + visual.v - a[6][1]) / a[5][1];
        }
        return true;
    }
};

class VisualTimed : public ceres::SizedCostFunction<2, 4, 3, 3, 4, 3, 2, 6> {
  public:
    Visual visual;
    M3 motion;
    V3 displacement;
    static Matrix<double, 4, 3> plus(const double *q) {
        Matrix<double, 4, 3> J;
        J << -q[1], -q[2], -q[3], q[0], q[3], -q[2], -q[3], q[0], q[1], q[2], -q[1], q[0];
        return J;
    }
    template <class T>
    void project(const Matrix<T, 3, 1> &pt, const T *ff, const T *dd, T *r) const {
        const auto &K = visual.K;
        T aa = pt.x() / pt.z(), bb = pt.y() / pt.z(), rho = sqrt(aa * aa + bb * bb + T(1e-16)),
          theta = atan(rho), tt = theta * theta;
        T td =
            theta *
            (T(1) + tt * ((T(K[4]) + dd[2]) +
                          tt * ((T(K[5]) + dd[3]) +
                                tt * (T(K[6]) + tt * (T(K[7]) + tt * (T(K[8]) + tt * T(K[9])))))));
        T un = aa * td / rho, vn = bb * td / rho, rr = un * un + vn * vn;
        T up = un + (T(K[10]) + dd[4]) * (rr + T(2) * un * un) +
               (T(2 * K[11]) + T(2) * dd[5]) * un * vn + T(K[12]) * rr + T(K[13]) * rr * rr;
        T vp = vn + (T(K[11]) + dd[5]) * (rr + T(2) * vn * vn) +
               (T(2 * K[10]) + T(2) * dd[4]) * un * vn + T(K[14]) * rr + T(K[15]) * rr * rr;
        r[0] = ff[0] * up + T(K[2] - visual.u) + dd[0];
        r[1] = ff[1] * vp + T(K[3] - visual.v) + dd[1];
    }

  public:
    VisualTimed(double *K, double u, double v, double delay, V3 omega, V3 velocity)
        : visual(K, u, v), motion(expq(-omega * delay).toRotationMatrix()),
          displacement(velocity * delay) {
    }
    VisualTimed(double *K, double u, double v, const M3 &R, const V3 &translation)
        : visual(K, u, v), motion(R), displacement(translation) {
    }
    bool Evaluate(double const *const *a, double *r, double **jac) const override {
        Q rb(a[0][0], a[0][1], a[0][2], a[0][3]), rc(a[3][0], a[3][1], a[3][2], a[3][3]);
        M3 B = motion * rb.toRotationMatrix().transpose(), C = rc.toRotationMatrix().transpose();
        V3 delta = Map<const V3>(a[2]) - Map<const V3>(a[1]) - displacement,
           body = B * delta - Map<const V3>(a[4]), pt = C * body;
        if (!jac) {
            manualProject(visual, pt, a[5], a[6], r, nullptr);
            return true;
        }
        Matrix<double, 2, 3> G;
        manualProject(visual, pt, a[5], a[6], r, &G);
        if (jac[0]) {
            Map<Matrix<double, 2, 4, RowMajor>> Jout(jac[0]);
            Jout = 2 * G * C * B * skew(delta) * plus(a[0]).transpose();
        }
        if (jac[1]) {
            Map<Matrix<double, 2, 3, RowMajor>> Jout(jac[1]);
            Jout = -G * C * B;
        }
        if (jac[2]) {
            Map<Matrix<double, 2, 3, RowMajor>> Jout(jac[2]);
            Jout = G * C * B;
        }
        if (jac[3]) {
            Map<Matrix<double, 2, 4, RowMajor>> Jout(jac[3]);
            Jout = 2 * G * C * skew(body) * plus(a[3]).transpose();
        }
        if (jac[4]) {
            Map<Matrix<double, 2, 3, RowMajor>> Jout(jac[4]);
            Jout = -G * C;
        }
        if (jac[5]) {
            Map<Matrix<double, 2, 2, RowMajor>> F(jac[5]);
            F.setZero();
            F(0, 0) = (r[0] - visual.K[2] + visual.u - a[6][0]) / a[5][0];
            F(1, 1) = (r[1] - visual.K[3] + visual.v - a[6][1]) / a[5][1];
        }
        if (jac[6] && getenv("PRINCIPAL_ONLY")) {
            Map<Matrix<double, 2, 6, RowMajor>> F(jac[6]);
            F.setZero();
            F(0, 0) = F(1, 1) = 1;
        } else if (jac[6]) {
            using JJ = ceres::Jet<double, 6>;
            Matrix<JJ, 3, 1> pjj;
            JJ fjj[2], djj[6], rjj[2];
            for (int k = 0; k < 3; k++)
                pjj[k] = JJ(pt[k]);
            for (int k = 0; k < 2; k++)
                fjj[k] = JJ(a[5][k]);
            for (int k = 0; k < 6; k++) {
                djj[k] = JJ(a[6][k]);
                djj[k].v[k] = 1;
            }
            project<JJ>(pjj, fjj, djj, rjj);
            Map<Matrix<double, 2, 6, RowMajor>> F(jac[6]);
            for (int k = 0; k < 2; k++)
                F.row(k) = rjj[k].v.transpose();
        }
        return true;
    }
};

// Camera motion during the effective row-dependent exposure delay.
class VisualPLS : public ceres::SizedCostFunction<2, 4, 3, 3, 4, 3, 2, 6, 3, 1> {
    VisualAnalytic projection;
    double delay;
    V3 omega;

  public:
    VisualPLS(double *K, double u, double v, double dt, V3 w)
        : projection(K, u, v), delay(dt), omega(w) {
    }
    bool Evaluate(double const *const *a, double *r, double **jac) const override {
        Q rb(a[0][0], a[0][1], a[0][2], a[0][3]), rc(a[3][0], a[3][1], a[3][2], a[3][3]);
        double dt = delay * a[8][0];
        M3 B = rb.toRotationMatrix().transpose(), C = rc.toRotationMatrix().transpose(),
           D = expq(-omega * dt).toRotationMatrix();
        V3 vel = Map<const V3>(a[7]), delta = Map<const V3>(a[2]) - Map<const V3>(a[1]) - vel * dt;
        V3 body = D * B * delta - Map<const V3>(a[4]), pt = C * body;
        if (!jac) {
            manualProject(projection.visual, pt, a[5], a[6], r, nullptr);
            return true;
        }
        Matrix<double, 2, 3> G;
        manualProject(projection.visual, pt, a[5], a[6], r, &G);
        if (jac[0]) {
            Map<Matrix<double, 2, 4, RowMajor>> Jout(jac[0]);
            Jout = 2 * G * C * D * B * skew(delta) * VisualAnalytic::plus(a[0]).transpose();
        }
        if (jac[1]) {
            Map<Matrix<double, 2, 3, RowMajor>> Jout(jac[1]);
            Jout = -G * C * D * B;
        }
        if (jac[2]) {
            Map<Matrix<double, 2, 3, RowMajor>> Jout(jac[2]);
            Jout = G * C * D * B;
        }
        if (jac[3]) {
            Map<Matrix<double, 2, 4, RowMajor>> Jout(jac[3]);
            Jout = 2 * G * C * skew(body) * VisualAnalytic::plus(a[3]).transpose();
        }
        if (jac[4]) {
            Map<Matrix<double, 2, 3, RowMajor>> Jout(jac[4]);
            Jout = -G * C;
        }
        if (jac[5]) {
            Map<Matrix<double, 2, 2, RowMajor>> F(jac[5]);
            F.setZero();
            F(0, 0) = (r[0] - projection.visual.K[2] + projection.visual.u - a[6][0]) / a[5][0];
            F(1, 1) = (r[1] - projection.visual.K[3] + projection.visual.v - a[6][1]) / a[5][1];
        }
        if (jac[6]) {
            using JJ = ceres::Jet<double, 6>;
            Matrix<JJ, 3, 1> pjj;
            JJ fjj[2], djj[6], rjj[2];
            for (int k = 0; k < 3; k++)
                pjj[k] = JJ(pt[k]);
            for (int k = 0; k < 2; k++)
                fjj[k] = JJ(a[5][k]);
            for (int k = 0; k < 6; k++) {
                djj[k] = JJ(a[6][k]);
                djj[k].v[k] = 1;
            }
            projection.project<JJ>(pjj, fjj, djj, rjj);
            Map<Matrix<double, 2, 6, RowMajor>> F(jac[6]);
            for (int k = 0; k < 2; k++)
                F.row(k) = rjj[k].v.transpose();
        }
        if (jac[7]) {
            Map<Matrix<double, 2, 3, RowMajor>> Jout(jac[7]);
            Jout = -dt * G * C * D * B;
        }
        if (jac[8]) {
            Map<Matrix<double, 2, 1>> Jout(jac[8]);
            Jout = G * C * (-skew(omega * delay) * D * B * delta - D * B * vel * delay);
        }
        return true;
    }
};
struct ScalarPrior {
    double init, sig;
    ScalarPrior(double x, double s) : init(x), sig(s) {
    }
    template <class T> bool operator()(const T *x, T *r) const {
        r[0] = (x[0] - T(init)) / T(sig);
        return true;
    }
};

struct VecPrior {
    V3 x;
    double sig;
    VecPrior(V3 p, double s) : x(p), sig(s) {
    }
    template <class T> bool operator()(const T *p, T *r) const {
        for (int k = 0; k < 3; k++)
            r[k] = (p[k] - T(x[k])) / T(sig);
        return true;
    }
};
struct Diff {
    double sig;
    Diff(double s) : sig(s) {
    }
    template <class T> bool operator()(const T *a, const T *b, T *r) const {
        for (int k = 0; k < 3; k++)
            r[k] = (b[k] - a[k]) / T(sig);
        return true;
    }
};
// Let smooth gyro-bias drift follow the measured temperature trend.
// Bias states keep their original sensor interpretation; only their prior changes.
struct BiasTempDiff {
    double change, sig;
    BiasTempDiff(double d, double s) : change(d), sig(s) {
    }
    template <class T> bool operator()(const T *a, const T *b, const T *slope, T *r) const {
        for (int k = 0; k < 3; k++)
            r[k] = (b[k] - a[k] - T(change) * slope[k]) / T(sig);
        return true;
    }
};
struct RotDiff {
    double sig;
    RotDiff(double s) : sig(s) {
    }
    template <class T> bool operator()(const T *a, const T *b, T *r) const {
        Quaternion<T> aa(a[0], a[1], a[2], a[3]), bb(b[0], b[1], b[2], b[3]);
        Quaternion<T> d = aa.conjugate() * bb;
        T sg = d.w() < T(0) ? T(-1) : T(1);
        for (int k = 0; k < 3; k++)
            r[k] = T(2 / sig) * sg * d.vec()[k];
        return true;
    }
};
struct NormPrior {
    V3 origin;
    double norm, sig;
    NormPrior(V3 a, double n, double s) : origin(a), norm(n), sig(s) {
    }
    template <class T> bool operator()(const T *p, T *r) const {
        T sum = T(0);
        for (int k = 0; k < 3; k++) {
            T v = p[k] - T(origin[k]);
            sum += v * v;
        }
        r[0] = (sqrt(sum) - T(norm)) / T(sig);
        return true;
    }
};

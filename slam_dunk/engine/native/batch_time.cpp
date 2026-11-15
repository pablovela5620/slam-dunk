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
using namespace std;
using namespace Eigen;
using V3 = Vector3d;
using M3 = Matrix3d;
using Q = Quaterniond;
using M39 = Matrix<double, 3, 9>;
using M3r = Matrix<double, 3, 3, RowMajor>;
M3 skew(V3 v) {
    M3 s;
    s << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return s;
}
Q expq(V3 v) {
    double a = v.norm();
    return a < 1e-12 ? Q::Identity() : Q(AngleAxisd(a, v / a));
}
struct Pose {
    double q[4], p[3], v[3];
    Q rot() const {
        return Q(q[0], q[1], q[2], q[3]);
    }
};
struct Imu {
    double t;
    V3 w, a;
    double temp = 0;
};
vector<Imu> imu;
double gyroTref = 0;
using M32 = Matrix<double, 3, 2>;
struct Pre {
    M3 Jrt = M3::Zero(), Jvt = M3::Zero(), Jpt = M3::Zero();
    V3 gt = V3::Zero();
    M32 Jtr = M32::Zero(), Jtv = M32::Zero(), Jtp = M32::Zero();
    Vector2d tm = Vector2d::Zero();
    M39 Jrg = M39::Zero(), Jvgm = M39::Zero(), Jpgm = M39::Zero();
    Matrix<double, 9, 1> gm = Matrix<double, 9, 1>::Zero();
    Q r = Q::Identity();
    V3 v = V3::Zero(), p = V3::Zero(), bg = V3::Zero(), ba = V3::Zero(), ar = V3::Zero(),
       sc = V3::Zero();
    M3 Jscv = M3::Zero(), Jscp = M3::Zero();
    M3 Jarv = M3::Zero(), Jarp = M3::Zero();
    M3 Jr = M3::Zero(), Jvg = M3::Zero(), Jva = M3::Zero(), Jpg = M3::Zero(), Jpa = M3::Zero();
    double dt = 0;
};
Pre integ_left(double t0, double t1, V3 bg, V3 ba, V3 ar = V3::Zero(), V3 sc = V3::Zero(),
               const double *gm = nullptr, const double *tm = nullptr, const double *gt = nullptr) {
    Pre z;
    if (gt)
        z.gt = Map<const V3>(gt);
    if (tm)
        z.tm = Map<const Vector2d>(tm);
    V3 wstart = V3::Zero(), astart = V3::Zero(), wend = V3::Zero(), aend = V3::Zero();
    bool started = false;
    if (gm)
        z.gm = Map<const Matrix<double, 9, 1>>(gm);
    z.bg = bg;
    z.ba = ba;
    z.ar = ar;
    z.sc = sc;
    auto it =
        lower_bound(imu.begin(), imu.end(), t0, [](const Imu &s, double t) { return s.t < t; });
    if (it != imu.begin())
        --it;
    for (; it != imu.end() && it->t < t1; ++it) {
        double a = max(t0, it->t), b = min(t1, (it + 1 == imu.end() ? t1 : (it + 1)->t)),
               dt = b - a;
        if (dt <= 0)
            continue;
        V3 w = it->w + Map<const M3r>(z.gm.data()) * it->w + it->temp * z.gt.cwiseProduct(it->w) -
               bg,
           ac = it->a + skew(ar) * it->a + sc.cwiseProduct(it->a) - ba;
        if (!started) {
            wstart = w;
            astart = ac;
            started = true;
        }
        wend = w;
        aend = ac;
        M3 R = z.r.toRotationMatrix();
        V3 adot = V3::Zero();
        if (it + 1 != imu.end()) {
            V3 rawad = ((it + 1)->a - it->a) / ((it + 1)->t - it->t);
            adot = rawad + skew(ar) * rawad + sc.cwiseProduct(rawad);
        }
        z.Jtp.col(1) += z.Jtv.col(1) * dt + .5 * R * adot * dt * dt;
        z.Jtv.col(1) += R * adot * dt;
        z.Jpt += z.Jvt * dt - .5 * R * skew(ac) * z.Jrt * dt * dt;
        z.Jvt -= R * skew(ac) * z.Jrt * dt;
        z.Jpgm += z.Jvgm * dt - .5 * R * skew(ac) * z.Jrg * dt * dt;
        z.Jvgm -= R * skew(ac) * z.Jrg * dt;
        z.p += z.v * dt + .5 * R * ac * dt * dt;
        z.Jscp += z.Jscv * dt + .5 * R * it->a.asDiagonal() * dt * dt;
        z.Jscv += R * it->a.asDiagonal() * dt;
        z.Jarp += z.Jarv * dt - .5 * R * skew(it->a) * dt * dt;
        z.Jarv -= R * skew(it->a) * dt;
        z.Jpa += z.Jva * dt - .5 * R * dt * dt;
        z.Jpg += z.Jvg * dt - .5 * R * skew(ac) * z.Jr * dt * dt;
        z.v += R * ac * dt;
        z.Jvg -= R * skew(ac) * z.Jr * dt;
        z.Jva -= R * dt;
        Q d = expq(w * dt);
        z.Jr = d.toRotationMatrix().transpose() * z.Jr - M3::Identity() * dt;
        M39 Jw = M39::Zero();
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                Jw(a, 3 * a + b) = it->w[b];
        z.Jrg = d.toRotationMatrix().transpose() * z.Jrg + Jw * dt;
        z.Jrt = d.toRotationMatrix().transpose() * z.Jrt;
        z.Jrt.diagonal() += it->temp * dt * it->w;
        z.r = (z.r * d).normalized();
        z.dt += dt;
    }
    z.Jtr.col(0) = wend - z.r.conjugate() * wstart;
    z.Jtv.col(0) = -skew(wstart) * z.v - astart + z.r * aend;
    z.Jtp.col(0) = -skew(wstart) * z.p - astart * z.dt + z.v;
    return z;
}
M3 rightJac(const V3 &v) {
    double a = v.norm();
    M3 S = skew(v);
    if (a < 1e-5)
        return M3::Identity() - .5 * S + (1. / 6) * S * S;
    return M3::Identity() - (1 - cos(a)) / (a * a) * S + (a - sin(a)) / (a * a * a) * S * S;
}
Pre integ_mid(double t0, double t1, V3 bg, V3 ba, V3 ar, V3 sc, const double *gm, const double *tm,
              const double *gt) {
    Pre z;
    z.bg = bg;
    z.ba = ba;
    z.ar = ar;
    z.sc = sc;
    if (gm)
        z.gm = Map<const Matrix<double, 9, 1>>(gm);
    if (tm)
        z.tm = Map<const Vector2d>(tm);
    if (gt)
        z.gt = Map<const V3>(gt);
    auto it =
        lower_bound(imu.begin(), imu.end(), t0, [](const Imu &s, double t) { return s.t < t; });
    if (it != imu.begin())
        --it;
    V3 wstart = V3::Zero(), astart = V3::Zero(), wend = V3::Zero(), aend = V3::Zero();
    bool first = true;
    auto calcw = [&](const V3 &w, double temp) -> V3 {
        return w + Map<const M3r>(z.gm.data()) * w + temp * z.gt.cwiseProduct(w) - bg;
    };
    auto calca = [&](const V3 &a) -> V3 { return a + skew(ar) * a + sc.cwiseProduct(a) - ba; };
    for (; it != imu.end() && it->t < t1; ++it) {
        auto nx = (it + 1 == imu.end() ? it : it + 1);
        double ta = max(t0, it->t), tb = min(t1, (nx == it ? t1 : nx->t)), dt = tb - ta;
        if (dt <= 0)
            continue;
        double span = (nx == it ? 1. : nx->t - it->t);
        double fa = (ta - it->t) / span, fb = (tb - it->t) / span, alpha = .5 * (fa + fb);
        V3 raww = (1 - alpha) * it->w + alpha * nx->w, rawa = (1 - alpha) * it->a + alpha * nx->a;
        double temp = (1 - alpha) * it->temp + alpha * nx->temp;
        V3 w = calcw(raww, temp), ac = calca(rawa);
        if (first) {
            wstart = calcw((1 - fa) * it->w + fa * nx->w, (1 - fa) * it->temp + fa * nx->temp);
            astart = calca((1 - fa) * it->a + fa * nx->a);
            first = false;
        }
        wend = calcw((1 - fb) * it->w + fb * nx->w, (1 - fb) * it->temp + fb * nx->temp);
        aend = calca((1 - fb) * it->a + fb * nx->a);
        M3 D = expq(w * dt).toRotationMatrix(), Dh = expq(w * (dt * .5)).toRotationMatrix(),
           H = rightJac(w * (dt * .5)) * (dt * .5), J = rightJac(w * dt) * dt,
           R = z.r.toRotationMatrix() * Dh;
        M39 Wg = M39::Zero();
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                Wg(a, 3 * a + b) = raww[b];
        M3 Wt = temp * raww.asDiagonal();
        M3 Br = Dh.transpose() * z.Jr - H, Bt = Dh.transpose() * z.Jrt + H * Wt;
        M39 Bgm = Dh.transpose() * z.Jrg + H * Wg;
        auto propagate = [&](auto &Jp, auto &Jv, const auto &B) {
            auto Da = (-R * skew(ac) * B).eval();
            Jp += Jv * dt + .5 * Da * dt * dt;
            Jv += Da * dt;
        };
        propagate(z.Jpg, z.Jvg, Br);
        propagate(z.Jpgm, z.Jvgm, Bgm);
        propagate(z.Jpt, z.Jvt, Bt);
        auto addacc = [&](M3 &Jp, M3 &Jv, const M3 &A) {
            Jp += Jv * dt + .5 * R * A * dt * dt;
            Jv += R * A * dt;
        };
        addacc(z.Jpa, z.Jva, -M3::Identity());
        addacc(z.Jarp, z.Jarv, -skew(rawa));
        addacc(z.Jscp, z.Jscv, rawa.asDiagonal());
        V3 adot = calca((nx->a - it->a) / span) + ba;
        z.Jtp.col(1) += z.Jtv.col(1) * dt + .5 * R * adot * dt * dt;
        z.Jtv.col(1) += R * adot * dt;
        z.p += z.v * dt + .5 * R * ac * dt * dt;
        z.v += R * ac * dt;
        z.Jr = D.transpose() * z.Jr - J;
        z.Jrg = D.transpose() * z.Jrg + J * Wg;
        z.Jrt = D.transpose() * z.Jrt + J * Wt;
        z.r = (z.r * expq(w * dt)).normalized();
        z.dt += dt;
    }
    z.Jtr.col(0) = wend - z.r.conjugate() * wstart;
    z.Jtv.col(0) = -skew(wstart) * z.v - astart + z.r * aend;
    z.Jtp.col(0) = -skew(wstart) * z.p - astart * z.dt + z.v;
    return z;
}
Pre integ(double t0, double t1, V3 bg, V3 ba, V3 ar = V3::Zero(), V3 sc = V3::Zero(),
          const double *gm = nullptr, const double *tm = nullptr, const double *gt = nullptr) {
    if (getenv("MIDPOINT"))
        return integ_mid(t0, t1, bg, ba, ar, sc, gm, tm, gt);
    return integ_left(t0, t1, bg, ba, ar, sc, gm, tm, gt);
}

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
struct Inertial {
    Pre z;
    double gs, as;
    Inertial(Pre p, double g, double a) : z(p), gs(g), as(a) {
    }
    template <class T>
    bool operator()(const T *qi, const T *pi, const T *vi, const T *qj, const T *pj, const T *vj,
                    const T *bg, const T *ba, const T *gg, const T *ar, const T *sc, const T *gm,
                    const T *tm, const T *gt, T *r) const {
        Matrix<T, 2, 1> dtime(tm[0] - T(z.tm[0]), tm[1] - T(z.tm[1]));
        using V = Matrix<T, 3, 1>;
        V dgt(gt[0] - T(z.gt[0]), gt[1] - T(z.gt[1]), gt[2] - T(z.gt[2]));
        Matrix<T, 9, 1> dg;
        for (int k = 0; k < 9; k++)
            dg[k] = gm[k] - T(z.gm[k]);
        Quaternion<T> ai(qi[0], qi[1], qi[2], qi[3]), aj(qj[0], qj[1], qj[2], qj[3]);
        V db(bg[0] - T(z.bg[0]), bg[1] - T(z.bg[1]), bg[2] - T(z.bg[2])),
            da(ba[0] - T(z.ba[0]), ba[1] - T(z.ba[1]), ba[2] - T(z.ba[2])), g(gg[0], gg[1], gg[2]);
        V dsc(sc[0] - T(z.sc[0]), sc[1] - T(z.sc[1]), sc[2] - T(z.sc[2]));
        V dar(ar[0] - T(z.ar[0]), ar[1] - T(z.ar[1]), ar[2] - T(z.ar[2]));
        V dr = z.Jr.cast<T>() * db + z.Jrg.cast<T>() * dg + z.Jtr.cast<T>() * dtime +
               z.Jrt.cast<T>() * dgt;
        Quaternion<T> dq(T(1), dr[0] / T(2), dr[1] / T(2), dr[2] / T(2)),
            er = (z.r.cast<T>() * dq).conjugate() * (ai.conjugate() * aj);
        T sn = er.w() < T(0) ? T(-1) : T(1);
        double sr = gs * sqrt(z.dt) + .0001, sv = as * sqrt(z.dt),
               sp = .5 * as * sqrt(z.dt) * z.dt + .001;
        for (int k = 0; k < 3; k++)
            r[k] = T(2 / sr) * sn * er.vec()[k];
        V dpi(pj[0] - pi[0] - vi[0] * T(z.dt), pj[1] - pi[1] - vi[1] * T(z.dt),
              pj[2] - pi[2] - vi[2] * T(z.dt)),
            dvi(vj[0] - vi[0], vj[1] - vi[1], vj[2] - vi[2]);
        V ep = ai.conjugate() * (dpi - g * T(z.dt * z.dt / 2)) - z.p.cast<T>() -
               z.Jpg.cast<T>() * db - z.Jpa.cast<T>() * da - z.Jarp.cast<T>() * dar -
               z.Jscp.cast<T>() * dsc - z.Jpgm.cast<T>() * dg - z.Jtp.cast<T>() * dtime -
               z.Jpt.cast<T>() * dgt,
          ev = ai.conjugate() * (dvi - g * T(z.dt)) - z.v.cast<T>() - z.Jvg.cast<T>() * db -
               z.Jva.cast<T>() * da - z.Jarv.cast<T>() * dar - z.Jscv.cast<T>() * dsc -
               z.Jvgm.cast<T>() * dg - z.Jtv.cast<T>() * dtime - z.Jvt.cast<T>() * dgt;
        for (int k = 0; k < 3; k++) {
            r[3 + k] = ev[k] / T(sv);
            r[6 + k] = ep[k] / T(sp);
        }
        return true;
    }
};
#include "inertial_analytic.h"
struct GyroPrior {
    double sig;
    GyroPrior(double s) : sig(s) {
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
        if (getenv("POSE_STOP") && s.iteration > 12 && max_step < 1e-5 && max_angle < 1e-7) {
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

#include "earth_inertial.h"
#include <ceres/product_manifold.h>
#include "pose_blocks.h"
int main(int argc, char **argv) {
    auto programstart = chrono::steady_clock::now();
    if (argc < 3)
        return 1;
    ifstream f(argv[1], ios::binary);
    int64_t H[4];
    f.read((char *)H, sizeof(H));
    int N = H[0], P = H[1], O = H[2], I = H[3];
    double K[32], ext[14];
    rd(f, K, 32);
    rd(f, ext, 14);
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
    double chunkdt = envd("CHUNK_DT", 30), duration = times.back() - times.front();
    int NC = int(duration / chunkdt) + 1;
    vector<array<double, 4>> cq(NC);
    vector<array<double, 3>> ct(NC), bg(NC), ba(NC);
    vector<int> ci(N);
    for (int k = 0; k < NC; k++) {
        copy(ext, ext + 4, cq[k].begin());
        copy(ext + 4, ext + 7, ct[k].begin());
        bg[k] = {0, 0, 0};
        ba[k] = {0, 0, 0};
    }
    for (int i = 0; i < N; i++)
        ci[i] = min(NC - 1, int((times[i] - times.front()) / chunkdt));
    V3 av = V3::Zero();
    double total = 0;
    for (int i = 0; i < N - 1; i++) {
        auto z = integ(times[i], times[i + 1], V3::Zero(), V3::Zero());
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
    double ar[3] = {0, 0, 0}, sc[3] = {0, 0, 0}, foc[4] = {K[0], K[1], K[16], K[17]},
           intr[12] = {0};
    double gs = envd("SIGMA_G", .0025), as = envd("SIGMA_A", .10);
    int outer = int(envd("OUTER", 2));
    vector<array<double, 4>> fcs(NC);
    for (int k = 0; k < NC; k++)
        fcs[k] = {K[0], K[1], K[16], K[17]};
    double fprior = envd("FOCAL_PRIOR", .5) * sqrt(double(NC)), frw = envd("FOCAL_RW", .05);
    double warmradius = 1e4;
    vector<array<double, 3>> initialpoints = points;
    for (int pass = 0; pass < outer; pass++) {
        shift_imu(baseimu, tm[0], tm[1]);
        // Stateless losses are shared between observations. Ceres owns residuals,
        // while these owners outlive the problem and keep its loss pointers valid.
        unique_ptr<ceres::LossFunction> visual_base(
            (pass > 0 && getenv("SOFT_CAUCHY"))
                ? static_cast<ceres::LossFunction *>(new ceres::CauchyLoss(envd("SOFT_CAUCHY", 1.)))
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
            pr.AddParameterBlock(cq[k].data(), 4, new ceres::QuaternionManifold());
            pr.AddParameterBlock(ct[k].data(), 3);
            ordering->AddElementToGroup(cq[k].data(), 1);
            ordering->AddElementToGroup(ct[k].data(), 1);
            pr.AddResidualBlock(new ceres::AutoDiffCostFunction<NormPrior, 1, 3>(
                                    new NormPrior(Map<V3>(ext + 11), B, .0005)),
                                nullptr, ct[k].data());
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
                pr.AddResidualBlock(new ceres::AutoDiffCostFunction<RotDiff, 3, 4, 4>(
                                        new RotDiff(envd("EXTR_RW", .001))),
                                    nullptr, cq[k - 1].data(), cq[k].data());
                pr.AddResidualBlock(new ceres::AutoDiffCostFunction<Diff, 3, 3, 3>(new Diff(.0005)),
                                    nullptr, ct[k - 1].data(), ct[k].data());
                pr.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<BiasTempDiff, 3, 3, 3, 3>(new BiasTempDiff(
                        chunk_temperature[k] - chunk_temperature[k - 1], envd("BIAS_RW", .00015))),
                    nullptr, bg[k - 1].data(), bg[k].data(), bias_temp);
                pr.AddResidualBlock(new ceres::AutoDiffCostFunction<Diff, 3, 3, 3>(
                                        new Diff(envd("ACC_BIAS_RW", .04))),
                                    nullptr, ba[k - 1].data(), ba[k].data());
            }
            if (getenv("FIX_EXTR") || getenv("CAMERA_ONLY")) {
                pr.SetParameterBlockConstant(cq[k].data());
                pr.SetParameterBlockConstant(ct[k].data());
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
                double *fk = fcs[k].data() + 2 * c;
                pr.AddResidualBlock(new ceres::AutoDiffCostFunction<FocalPrior, 2, 2>(
                                        new FocalPrior(K[16 * c], K[16 * c + 1], fprior)),
                                    nullptr, fk);
                ordering->AddElementToGroup(fk, 1);
                if (!getenv("REFINE_FOCAL"))
                    pr.SetParameterBlockConstant(fk);
                if (getenv("FOCAL_EQUAL"))
                    pr.AddResidualBlock(new ceres::AutoDiffCostFunction<Aspect, 1, 2>(new Aspect()),
                                        nullptr, fk);
                if (k > 0)
                    pr.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<Diff2, 2, 2, 2>(new Diff2(frw)), nullptr,
                        fcs[k - 1].data() + 2 * c, fk);
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
        pr.AddResidualBlock(new ceres::AutoDiffCostFunction<GyroPrior, 9, 9>(
                                new GyroPrior(envd("GYRO_PRIOR", .002))),
                            nullptr, gm);
        ordering->AddElementToGroup(gm, 1);
        if (!getenv("GYRO_REFINE"))
            pr.SetParameterBlockConstant(gm);
        pr.AddResidualBlock(new ceres::AutoDiffCostFunction<VecPrior, 3, 3>(
                                new VecPrior(V3::Zero(), envd("ACC_SCALE_SIG", .003))),
                            nullptr, sc);
        ordering->AddElementToGroup(sc, 1);
        if (!getenv("ACC_SCALE"))
            pr.SetParameterBlockConstant(sc);
        for (auto &o : obs) {
            int i = int(o[0]), c = int(o[1]), p = int(o[2]);
            double *eq = c ? ext + 7 : cq[ci[i]].data();
            double *et = c ? ext + 11 : ct[ci[i]].data();
            ceres::CostFunction *vc;
            if (usepls) {
                double ro = .00912 * clamp(o[4] / heights[c], 0., 1.),
                       ex = max(metadata[i][c], 1e-6),
                       delay = .5 * ro * .01 * (1 + ro / ex) / (1 + .01 * ro / ex);
                V3 w(metadata[i][2], metadata[i][3], metadata[i][4]);
                vc = new VisualTimed(K + 16 * c, o[3], o[4], delay * pls[0], w,
                                     Map<V3>(initialposes[i].v));
            } else
                vc = new VisualAnalytic(K + 16 * c, o[3], o[4]);
            pr.AddResidualBlock(new PoseVisual(vc), visual_losses[p], poses[i].q, points[p].data(),
                                eq, et, fcs[ci[i]].data() + 2 * c, intr + 6 * c);
        }
        if (pr.HasParameterBlock(ext + 7))
            pr.SetParameterBlockConstant(ext + 7);
        if (pr.HasParameterBlock(ext + 11))
            pr.SetParameterBlockConstant(ext + 11);
        for (int i = 0; i < N - 1; i++) {
            int k = ci[i];
            auto z = integ(times[i], times[i + 1], Map<V3>(bg[k].data()), Map<V3>(ba[k].data()),
                           Map<V3>(ar), Map<V3>(sc), gm, tm, gt);
            pr.AddResidualBlock(
                new PoseInertial(new EarthInertial(
                    getenv("AUTODIFF")
                        ? static_cast<ceres::CostFunction *>(
                              new ceres::AutoDiffCostFunction<Inertial, 9, 4, 3, 3, 4, 3, 3, 3, 3,
                                                              3, 3, 3, 9, 2, 3>(
                                  new Inertial(z, gs, as)))
                        : static_cast<ceres::CostFunction *>(new InertialAnalytic(z, gs, as)),
                    z.dt)),
                nullptr, poses[i].q, poses[i].v, poses[i + 1].q, poses[i + 1].v, bg[k].data(),
                ba[k].data(), g, ar, sc, gm, tm, gt, earth);
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
            put(".accscale", [&](ostream &o) { o << Map<V3>(sc).transpose() << "\n"; });
            put(".diag", [&](ostream &o) {
                for (int k = 0; k < NC; k++)
                    o << k << " " << Map<V3>(bg[k].data()).transpose() << " "
                      << Map<V3>(ba[k].data()).transpose() << " " << cq[k][0] << " " << cq[k][1]
                      << " " << cq[k][2] << " " << cq[k][3] << " "
                      << Map<V3>(ct[k].data()).transpose() << "\n";
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
                max(1., remaining * (pass + 1 < outer ? envd("FIRST_FRACTION", .55) : 1.));
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
        ceres::Solve(op, &pr, &sm);
        if (!sm.iterations.empty())
            warmradius = clamp(sm.iterations.back().trust_region_radius, 1e4, 1e7);
        if (getenv("FULL_REPORT"))
            cerr << sm.FullReport() << endl;
        cerr << "POSE_STABILITY " << monitor.max_step << " " << monitor.mean_step << " "
             << monitor.max_angle << endl;
        for (int z = 0; z < 4; z++) {
            double m = 0, lo = 1e9, hi = -1e9;
            for (int k = 0; k < NC; k++) {
                m += fcs[k][z];
                lo = min(lo, fcs[k][z]);
                hi = max(hi, fcs[k][z]);
            }
            foc[z] = m / NC;
            cerr << "FOCAL " << z << " mean " << foc[z] << " min " << lo << " max " << hi << endl;
        }
        cerr << "STOP_REASON " << sm.message << endl;
        cerr << sm.BriefReport() << " g=" << Map<V3>(g).transpose()
             << " accrot=" << Map<V3>(ar).transpose() << " accscale=" << Map<V3>(sc).transpose()
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
        scout << setprecision(12) << sc[0] << " " << sc[1] << " " << sc[2] << "\n";
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
                 << Map<V3>(ba[k].data()).transpose() << " " << cq[k][0] << " " << cq[k][1] << " "
                 << cq[k][2] << " " << cq[k][3] << " " << Map<V3>(ct[k].data()).transpose() << "\n";
        diag.close();
        rename((path + ".diag.tmp").c_str(), (path + ".diag").c_str());

        if (getenv("SAVE_STATE")) {
            ofstream ptsout(path + ".points", ios::binary);
            for (auto &x : points)
                ptsout.write((char *)x.data(), 3 * sizeof(double));
            ofstream fcsout(path + ".focals");
            fcsout << setprecision(15);
            for (auto &x : fcs)
                fcsout << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << "\n";
        }
        if (pass + 1 < outer && envd("OUTLIER", 0) > 0 &&
            (!getenv("OUTLIER_GUARD") || sm.termination_type == ceres::CONVERGENCE ||
             (monitor.max_step < envd("OUTLIER_MAX_STEP", .05) && monitor.max_angle < .001))) {
            double threshold = envd("OUTLIER", 4);
            size_t old = obs.size();
            vector<array<double, 5>> kept;
            vector<int> cnt(P, 0);
            for (auto &o : obs) {
                int i = int(o[0]), c = int(o[1]), p = int(o[2]);
                double *eq = c ? ext + 7 : cq[ci[i]].data();
                double *et = c ? ext + 11 : ct[ci[i]].data();
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
                                           fcs[ci[i]].data() + 2 * c,
                                           intr + 6 * c,
                                           getenv("PLS_VEL_FIXED") ? initialposes[i].v : poses[i].v,
                                           pls};
                    if (getenv("PLS_FROZEN")) {
                        VisualTimed fixed(K + 16 * c, o[3], o[4], delay * pls[0], w,
                                          Map<V3>(initialposes[i].v));
                        fixed.Evaluate(par, rr, nullptr);
                    } else
                        cost.Evaluate(par, rr, nullptr);
                } else
                    Visual(K + 16 * c, o[3], o[4])(poses[i].q, poses[i].p, points[p].data(), eq, et,
                                                   fcs[ci[i]].data() + 2 * c, intr + 6 * c, rr);
                if (pt.z() > .05 && isfinite(rr[0]) && isfinite(rr[1]) &&
                    hypot(rr[0], rr[1]) <=
                        threshold +
                            (getenv("BLUR_OUTLIER") && usepls
                                 ? envd("BLUR_OUTLIER", .5) * metadata[i][c] * K[16 * c] *
                                       V3(metadata[i][2], metadata[i][3], metadata[i][4]).norm()
                                 : 0.)) {
                    kept.push_back(o);
                    cnt[p]++;
                }
            }
            obs.clear();
            fill(used.begin(), used.end(), false);
            for (auto &o : kept)
                if (cnt[int(o[2])] >= int(envd("KEEP_OBS", 3))) {
                    obs.push_back(o);
                    used[int(o[2])] = true;
                }
            cerr << "OUTLIERS removed " << old - obs.size() << " remain " << obs.size() << endl;
        }
    }
    return 0;
}

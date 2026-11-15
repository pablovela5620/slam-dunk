#pragma once

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
    V3 v = V3::Zero(), p = V3::Zero(), bg = V3::Zero(), ba = V3::Zero(), ar = V3::Zero();
    V6 sc = V6::Zero();
    M36 Jscv = M36::Zero(), Jscp = M36::Zero();
    M3 Jarv = M3::Zero(), Jarp = M3::Zero();
    M3 Jr = M3::Zero(), Jvg = M3::Zero(), Jva = M3::Zero(), Jpg = M3::Zero(), Jpa = M3::Zero();
    double dt = 0;
};
Pre integ_left(double t0, double t1, V3 bg, V3 ba, V3 ar = V3::Zero(), V6 sc = V6::Zero(),
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
           ac = it->a + skew(ar) * it->a + symmetric(sc) * it->a - ba;
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
            adot = rawad + skew(ar) * rawad + symmetric(sc) * rawad;
        }
        z.Jtp.col(1) += z.Jtv.col(1) * dt + .5 * R * adot * dt * dt;
        z.Jtv.col(1) += R * adot * dt;
        z.Jpt += z.Jvt * dt - .5 * R * skew(ac) * z.Jrt * dt * dt;
        z.Jvt -= R * skew(ac) * z.Jrt * dt;
        z.Jpgm += z.Jvgm * dt - .5 * R * skew(ac) * z.Jrg * dt * dt;
        z.Jvgm -= R * skew(ac) * z.Jrg * dt;
        z.p += z.v * dt + .5 * R * ac * dt * dt;
        z.Jscp += z.Jscv * dt + .5 * R * accDesign(it->a) * dt * dt;
        z.Jscv += R * accDesign(it->a) * dt;
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
Pre integ_mid(double t0, double t1, V3 bg, V3 ba, V3 ar, V6 sc, const double *gm, const double *tm,
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
    auto calcw = [&](const V3 &w, const V3 &a, double temp) -> V3 {
        return w + Map<const M3r>(z.gm.data()) * w + temp * z.gt.cwiseProduct(w) - bg;
    };
    auto calca = [&](const V3 &a) -> V3 { return a + skew(ar) * a + symmetric(sc) * a - ba; };
    for (; it != imu.end() && it->t < t1; ++it) {
        auto nx = (it + 1 == imu.end() ? it : it + 1);
        double ta = max(t0, it->t), tb = min(t1, (nx == it ? t1 : nx->t)), dt = tb - ta;
        if (dt <= 0)
            continue;
        double span = (nx == it ? 1. : nx->t - it->t);
        double fa = (ta - it->t) / span, fb = (tb - it->t) / span, alpha = .5 * (fa + fb);
        V3 raww = (1 - alpha) * it->w + alpha * nx->w, rawa = (1 - alpha) * it->a + alpha * nx->a;
        double temp = (1 - alpha) * it->temp + alpha * nx->temp;
        V3 w = calcw(raww, rawa, temp), ac = calca(rawa);
        if (first) {
            wstart = calcw((1 - fa) * it->w + fa * nx->w, (1 - fa) * it->a + fa * nx->a,
                           (1 - fa) * it->temp + fa * nx->temp);
            astart = calca((1 - fa) * it->a + fa * nx->a);
            first = false;
        }
        wend = calcw((1 - fb) * it->w + fb * nx->w, (1 - fb) * it->a + fb * nx->a,
                     (1 - fb) * it->temp + fb * nx->temp);
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
        auto addacc = [&](auto &Jp, auto &Jv, const auto &A) {
            Jp += Jv * dt + .5 * R * A * dt * dt;
            Jv += R * A * dt;
        };
        addacc(z.Jpa, z.Jva, -M3::Identity());
        addacc(z.Jarp, z.Jarv, -skew(rawa));
        addacc(z.Jscp, z.Jscv, accDesign(rawa));
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
Pre integ(double t0, double t1, V3 bg, V3 ba, V3 ar = V3::Zero(), V6 sc = V6::Zero(),
          const double *gm = nullptr, const double *tm = nullptr, const double *gt = nullptr) {
    if (getenv("MIDPOINT"))
        return integ_mid(t0, t1, bg, ba, ar, sc, gm, tm, gt);
    return integ_left(t0, t1, bg, ba, ar, sc, gm, tm, gt);
}

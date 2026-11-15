// Exact derivatives of the preintegrated residual, including quaternion manifolds.
class InertialAnalytic
    : public ceres::SizedCostFunction<9, 4, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 9, 2, 3> {
    Pre z;
    Matrix<double, 9, 9> W = Matrix<double, 9, 9>::Zero();
    static Matrix<double, 4, 3> plus(const double *q) {
        Matrix<double, 4, 3> J;
        J << -q[1], -q[2], -q[3], q[0], q[3], -q[2], -q[3], q[0], q[1], q[2], -q[1], q[0];
        return J;
    }

  public:
    InertialAnalytic(Pre p, double gs, double as) : z(p) {
        double sr = gs * sqrt(z.dt) + .0001, sv = as * sqrt(z.dt),
               sp = .5 * as * sqrt(z.dt) * z.dt + .001;
        W.block<3, 3>(0, 0) = M3::Identity() / sr;
        W.block<3, 3>(3, 3) = M3::Identity() / sv;
        W.block<3, 3>(6, 6) = M3::Identity() / sp;
    }
    bool Evaluate(double const *const *a, double *r, double **jac) const override {
        Q qi(a[0][0], a[0][1], a[0][2], a[0][3]), qj(a[3][0], a[3][1], a[3][2], a[3][3]);
        M3 B = qi.toRotationMatrix().transpose();
        V3 db = Map<const V3>(a[6]) - z.bg, da = Map<const V3>(a[7]) - z.ba,
           g = Map<const V3>(a[8]), dar = Map<const V3>(a[9]) - z.ar,
           dsc = Map<const V3>(a[10]) - z.sc, dgt = Map<const V3>(a[13]) - z.gt;
        Matrix<double, 9, 1> dg = Map<const Matrix<double, 9, 1>>(a[11]) - z.gm;
        Vector2d dtime = Map<const Vector2d>(a[12]) - z.tm;
        V3 dr = z.Jr * db + z.Jrg * dg + z.Jtr * dtime + z.Jrt * dgt;
        Q dq(1., dr[0] * .5, dr[1] * .5, dr[2] * .5), D = z.r * dq,
                                                      er = D.conjugate() * (qi.conjugate() * qj);
        double sn = er.w() < 0 ? -1. : 1.;
        V3 U = Map<const V3>(a[4]) - Map<const V3>(a[1]) - Map<const V3>(a[2]) * z.dt -
               g * (z.dt * z.dt * .5),
           V = Map<const V3>(a[5]) - Map<const V3>(a[2]) - g * z.dt;
        V3 ep = B * U - z.p - z.Jpg * db - z.Jpa * da - z.Jarp * dar - z.Jscp * dsc - z.Jpgm * dg -
                z.Jtp * dtime - z.Jpt * dgt;
        V3 ev = B * V - z.v - z.Jvg * db - z.Jva * da - z.Jarv * dar - z.Jscv * dsc - z.Jvgm * dg -
                z.Jtv * dtime - z.Jvt * dgt;
        Matrix<double, 9, 1> raw;
        raw.segment<3>(0) = 2 * sn * er.vec();
        raw.segment<3>(3) = ev;
        raw.segment<3>(6) = ep;
        Map<Matrix<double, 9, 1>> out(r);
        out = W * raw;
        if (!jac)
            return true;
        using M93 = Matrix<double, 9, 3>;
        M93 J = M93::Zero();
        Q c = D.conjugate() * qi.conjugate();
        M3 Z;
        for (int k = 0; k < 3; k++) {
            Q b(0., k == 0 ? 1. : 0., k == 1 ? 1. : 0., k == 2 ? 1. : 0.);
            Z.col(k) = 2 * sn * (c * b * qj).vec();
        }
        if (jac[0]) {
            J.setZero();
            J.block<3, 3>(0, 0) = -Z;
            J.block<3, 3>(3, 0) = 2 * B * skew(V);
            J.block<3, 3>(6, 0) = 2 * B * skew(U);
            Map<Matrix<double, 9, 4, RowMajor>>{jac[0]} = W * J * plus(a[0]).transpose();
        }
        if (jac[3]) {
            J.setZero();
            J.block<3, 3>(0, 0) = Z;
            Map<Matrix<double, 9, 4, RowMajor>>{jac[3]} = W * J * plus(a[3]).transpose();
        }
        auto emit = [&](int k, const M93 &j) {
            if (jac[k])
                Map<Matrix<double, 9, 3, RowMajor>>{jac[k]} = W * j;
        };
        J.setZero();
        J.block<3, 3>(6, 0) = -B;
        emit(1, J);
        J.block<3, 3>(6, 0) = B;
        emit(4, J);
        J.setZero();
        J.block<3, 3>(3, 0) = -B;
        J.block<3, 3>(6, 0) = -B * z.dt;
        emit(2, J);
        J.setZero();
        J.block<3, 3>(3, 0) = B;
        emit(5, J);
        J.setZero();
        J.block<3, 3>(3, 0) = -B * z.dt;
        J.block<3, 3>(6, 0) = -B * (z.dt * z.dt * .5);
        emit(8, J);
        Q h = z.r.conjugate() * (qi.conjugate() * qj);
        M3 H = -sn * (h.w() * M3::Identity() - skew(h.vec()));
        J.setZero();
        J.block<3, 3>(0, 0) = H * z.Jr;
        J.block<3, 3>(3, 0) = -z.Jvg;
        J.block<3, 3>(6, 0) = -z.Jpg;
        emit(6, J);
        J.setZero();
        J.block<3, 3>(3, 0) = -z.Jva;
        J.block<3, 3>(6, 0) = -z.Jpa;
        emit(7, J);
        J.setZero();
        J.block<3, 3>(3, 0) = -z.Jarv;
        J.block<3, 3>(6, 0) = -z.Jarp;
        emit(9, J);
        J.setZero();
        J.block<3, 3>(3, 0) = -z.Jscv;
        J.block<3, 3>(6, 0) = -z.Jscp;
        emit(10, J);
        J.setZero();
        J.block<3, 3>(0, 0) = H * z.Jrt;
        J.block<3, 3>(3, 0) = -z.Jvt;
        J.block<3, 3>(6, 0) = -z.Jpt;
        emit(13, J);
        if (jac[11]) {
            Matrix<double, 9, 9> G;
            G.block<3, 9>(0, 0) = H * z.Jrg;
            G.block<3, 9>(3, 0) = -z.Jvgm;
            G.block<3, 9>(6, 0) = -z.Jpgm;
            Map<Matrix<double, 9, 9, RowMajor>>{jac[11]} = W * G;
        }
        if (jac[12]) {
            Matrix<double, 9, 2> T;
            T.block<3, 2>(0, 0) = H * z.Jtr;
            T.block<3, 2>(3, 0) = -z.Jtv;
            T.block<3, 2>(6, 0) = -z.Jtp;
            Map<Matrix<double, 9, 2, RowMajor>>{jac[12]} = W * T;
        }
        return true;
    }
};

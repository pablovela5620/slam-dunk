// Small world-frame angular rate, estimated without a known world orientation.
// The regularizer has the scale of Earth's rotation; no location is assumed.
class EarthInertial : public ceres::CostFunction {
    unique_ptr<ceres::CostFunction> inner;
    double dt;

  public:
    EarthInertial(ceres::CostFunction *c, double t) : inner(c), dt(t) {
        *mutable_parameter_block_sizes() = c->parameter_block_sizes();
        mutable_parameter_block_sizes()->push_back(3);
        set_num_residuals(9);
    }
    bool Evaluate(double const *const *a, double *r, double **jac) const override {
        V3 phi = Map<const V3>(a[14]) * dt;
        Q qe = expq(phi), qj(a[3][0], a[3][1], a[3][2], a[3][3]), q = qe * qj;
        double qv[4] = {q.w(), q.x(), q.y(), q.z()};
        const double *par[14];
        double *jj[14];
        for (int k = 0; k < 14; k++) {
            par[k] = a[k];
            if (jac)
                jj[k] = jac[k];
        }
        par[3] = qv;
        Matrix<double, 9, 4, RowMajor> Jq;
        if (jac && (jac[3] || jac[14]))
            jj[3] = Jq.data();
        bool ok = inner->Evaluate(par, r, jac ? jj : nullptr);
        if (jac && jac[3]) {
            Matrix4d L;
            L << qe.w(), -qe.x(), -qe.y(), -qe.z(), qe.x(), qe.w(), -qe.z(), qe.y(), qe.y(), qe.z(),
                qe.w(), -qe.x(), qe.z(), -qe.y(), qe.x(), qe.w();
            Map<Matrix<double, 9, 4, RowMajor>>{jac[3]} = Jq * L;
        }
        if (jac && jac[14])
            Map<Matrix<double, 9, 3, RowMajor>>{jac[14]} =
                .5 * dt * Jq * VisualAnalytic::plus(qv) * rightJac(-phi);
        return ok;
    }
};

class PoseVisual : public ceres::SizedCostFunction<2, 7, 3, 4, 3, 2, 6> {
    unique_ptr<ceres::CostFunction> inner;

  public:
    explicit PoseVisual(ceres::CostFunction *c) : inner(c) {
    }
    bool Evaluate(double const *const *a, double *r, double **jac) const override {
        const double *par[7] = {a[0], a[0] + 4, a[1], a[2], a[3], a[4], a[5]};
        double *jj[7];
        Matrix<double, 2, 4, RowMajor> Jq;
        Matrix<double, 2, 3, RowMajor> Jp;
        if (jac) {
            jj[0] = jac[0] ? Jq.data() : nullptr;
            jj[1] = jac[0] ? Jp.data() : nullptr;
            for (int k = 2; k < 7; k++)
                jj[k] = jac[k - 1];
        }
        bool ok = inner->Evaluate(par, r, jac ? jj : nullptr);
        if (jac && jac[0]) {
            Map<Matrix<double, 2, 7, RowMajor>> J(jac[0]);
            J.leftCols<4>() = Jq;
            J.rightCols<3>() = Jp;
        }
        return ok;
    }
};
class PoseInertial : public ceres::SizedCostFunction<9, 7, 3, 7, 3, 3, 3, 3, 3, 6, 9, 2, 3, 3> {
    unique_ptr<ceres::CostFunction> inner;

  public:
    explicit PoseInertial(ceres::CostFunction *c) : inner(c) {
    }
    bool Evaluate(double const *const *a, double *r, double **jac) const override {
        const double *par[15] = {a[0], a[0] + 4, a[1], a[2], a[2] + 4, a[3],  a[4], a[5],
                                 a[6], a[7],     a[8], a[9], a[10],    a[11], a[12]};
        double *jj[15];
        Matrix<double, 9, 4, RowMajor> Jq[2];
        Matrix<double, 9, 3, RowMajor> Jp[2];
        if (jac) {
            jj[0] = jac[0] ? Jq[0].data() : nullptr;
            jj[1] = jac[0] ? Jp[0].data() : nullptr;
            jj[2] = jac[1];
            jj[3] = jac[2] ? Jq[1].data() : nullptr;
            jj[4] = jac[2] ? Jp[1].data() : nullptr;
            jj[5] = jac[3];
            for (int k = 6; k < 15; k++)
                jj[k] = jac[k - 2];
        }
        bool ok = inner->Evaluate(par, r, jac ? jj : nullptr);
        if (jac)
            for (int k = 0; k < 2; k++)
                if (jac[2 * k]) {
                    Map<Matrix<double, 9, 7, RowMajor>> J(jac[2 * k]);
                    J.leftCols<4>() = Jq[k];
                    J.rightCols<3>() = Jp[k];
                }
        return ok;
    }
};

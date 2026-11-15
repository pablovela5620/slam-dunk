// MEMS thermal drift is continuous, so bias knots represent interval centers.
class LinearGyroBias : public ceres::CostFunction {
    unique_ptr<ceres::CostFunction> inner;
    double weight;

  public:
    LinearGyroBias(ceres::CostFunction *c, double w) : inner(c), weight(w) {
        *mutable_parameter_block_sizes() = c->parameter_block_sizes();
        mutable_parameter_block_sizes()->push_back(3);
        set_num_residuals(9);
    }
    bool Evaluate(double const *const *a, double *r, double **jac) const override {
        const double *par[13];
        double *jj[13];
        for (int k = 0; k < 13; k++) {
            par[k] = a[k];
            if (jac)
                jj[k] = jac[k];
        }
        V3 b = (1 - weight) * Map<const V3>(a[4]) + weight * Map<const V3>(a[13]);
        par[4] = b.data();
        Matrix<double, 9, 3, RowMajor> J;
        if (jac && (jac[4] || jac[13]))
            jj[4] = J.data();
        bool ok = inner->Evaluate(par, r, jac ? jj : nullptr);
        if (jac && jac[4])
            Map<Matrix<double, 9, 3, RowMajor>>{jac[4]} = (1 - weight) * J;
        if (jac && jac[13])
            Map<Matrix<double, 9, 3, RowMajor>>{jac[13]} = weight * J;
        return ok;
    }
};

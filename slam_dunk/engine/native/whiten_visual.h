#pragma once
class WhitenVisual : public ceres::CostFunction {
    unique_ptr<ceres::CostFunction> inner;
    double factor;

  public:
    WhitenVisual(ceres::CostFunction *cost, double precision)
        : inner(cost), factor(sqrt(precision)) {
        *mutable_parameter_block_sizes() = inner->parameter_block_sizes();
        set_num_residuals(inner->num_residuals());
    }
    bool Evaluate(double const *const *a, double *r, double **jac) const override {
        if (!inner->Evaluate(a, r, jac))
            return false;
        for (int i = 0; i < num_residuals(); i++)
            r[i] *= factor;
        if (jac)
            for (size_t j = 0; j < parameter_block_sizes().size(); j++)
                if (jac[j])
                    for (int k = 0; k < num_residuals() * parameter_block_sizes()[j]; k++)
                        jac[j][k] *= factor;
        return true;
    }
};

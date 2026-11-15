#pragma once
// Exact scalar whitening through the loss: g(s)=family*rho(precision*s).
// This preserves objective, gradient and robust normal equations while leaving
// raw pixel residuals/Jacobians in the optimized camera implementation.
class PrecisionLoss : public ceres::LossFunction {
    const ceres::LossFunction *base;
    double precision, family;

  public:
    PrecisionLoss(const ceres::LossFunction *b, double p, double w)
        : base(b), precision(p), family(w) {
    }
    void Evaluate(double s, double *rho) const override {
        base->Evaluate(precision * s, rho);
        rho[0] *= family;
        rho[1] *= family * precision;
        rho[2] *= family * precision * precision;
    }
};

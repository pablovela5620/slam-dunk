#pragma once

// T_B0 is fixed. Chunk states are T_01 = (stereo_q, stereo_t), mapping
// camera 1 into camera 0. Only projection factors compose T_B1 = T_B0 T_01.
struct StereoReference {
    Q rotation;
    V3 translation;
    explicit StereoReference(const double *body_from_camera0)
        : rotation(body_from_camera0[0], body_from_camera0[1], body_from_camera0[2],
                   body_from_camera0[3]),
          translation(Map<const V3>(body_from_camera0 + 4)) {
    }
    void compose(const double *relative_q, const double *relative_t, double *body_q,
                 double *body_t) const {
        Q q = rotation * Q(relative_q[0], relative_q[1], relative_q[2], relative_q[3]);
        body_q[0] = q.w();
        body_q[1] = q.x();
        body_q[2] = q.y();
        body_q[3] = q.z();
        Map<V3> t(body_t);
        t = translation + rotation * Map<const V3>(relative_t);
    }
    void relative(const double *body_q, const double *body_t, double *relative_q,
                  double *relative_t) const {
        Q q = rotation.conjugate() * Q(body_q[0], body_q[1], body_q[2], body_q[3]);
        relative_q[0] = q.w();
        relative_q[1] = q.x();
        relative_q[2] = q.y();
        relative_q[3] = q.z();
        Map<V3> t(relative_t);
        t = rotation.conjugate() * (Map<const V3>(body_t) - translation);
    }
};

// Compose the fixed reference with the relative stereo state, including the
// ambient quaternion and translation Jacobians for Ceres' quaternion manifold.
class StereoVisual : public ceres::SizedCostFunction<2, 4, 3, 3, 4, 3, 2, 6> {
    unique_ptr<ceres::CostFunction> inner;
    StereoReference reference;

  public:
    StereoVisual(ceres::CostFunction *cost, const StereoReference &frame)
        : inner(cost), reference(frame) {
    }
    bool Evaluate(double const *const *a, double *r, double **jac) const override {
        double body_q[4], body_t[3];
        reference.compose(a[3], a[4], body_q, body_t);
        const double *par[7] = {a[0], a[1], a[2], body_q, body_t, a[5], a[6]};
        double *jj[7];
        Matrix<double, 2, 4, RowMajor> Jq;
        Matrix<double, 2, 3, RowMajor> Jt;
        if (jac) {
            copy(jac, jac + 7, jj);
            jj[3] = jac[3] ? Jq.data() : nullptr;
            jj[4] = jac[4] ? Jt.data() : nullptr;
        }
        bool ok = inner->Evaluate(par, r, jac ? jj : nullptr);
        if (jac && jac[3]) {
            const Q &q = reference.rotation;
            Matrix4d left;
            left << q.w(), -q.x(), -q.y(), -q.z(),
                    q.x(), q.w(), -q.z(), q.y(),
                    q.y(), q.z(), q.w(), -q.x(),
                    q.z(), -q.y(), q.x(), q.w();
            Map<Matrix<double, 2, 4, RowMajor>> out(jac[3]);
            out = Jq * left;
        }
        if (jac && jac[4]) {
            Map<Matrix<double, 2, 3, RowMajor>> out(jac[4]);
            out = Jt * reference.rotation.toRotationMatrix();
        }
        return ok;
    }
};

// Express local increments in body axes while storing states in stereo axes.
// This keeps Ceres' coordinate-dependent LM damping identical to the composed
// camera-to-body formulation, as well as preserving the mathematical objective.
class StereoRotationManifold : public ceres::Manifold {
    M3 body_from_stereo;
    ceres::QuaternionManifold quaternion;

  public:
    explicit StereoRotationManifold(const Q &rotation)
        : body_from_stereo(rotation.toRotationMatrix()) {
    }
    int AmbientSize() const override { return 4; }
    int TangentSize() const override { return 3; }
    bool Plus(const double *x, const double *delta, double *out) const override {
        V3 local = body_from_stereo.transpose() * Map<const V3>(delta);
        return quaternion.Plus(x, local.data(), out);
    }
    bool PlusJacobian(const double *x, double *out) const override {
        Matrix<double, 4, 3, RowMajor> J;
        if (!quaternion.PlusJacobian(x, J.data())) return false;
        Map<Matrix<double, 4, 3, RowMajor>> result(out);
        result = J * body_from_stereo.transpose();
        return true;
    }
    bool Minus(const double *y, const double *x, double *out) const override {
        V3 local;
        if (!quaternion.Minus(y, x, local.data())) return false;
        Map<V3> result(out);
        result = body_from_stereo * local;
        return true;
    }
    bool MinusJacobian(const double *x, double *out) const override {
        Matrix<double, 3, 4, RowMajor> J;
        if (!quaternion.MinusJacobian(x, J.data())) return false;
        Map<Matrix<double, 3, 4, RowMajor>> result(out);
        result = body_from_stereo * J;
        return true;
    }
};

class StereoTranslationManifold : public ceres::Manifold {
    M3 body_from_stereo;

  public:
    explicit StereoTranslationManifold(const Q &rotation)
        : body_from_stereo(rotation.toRotationMatrix()) {
    }
    int AmbientSize() const override { return 3; }
    int TangentSize() const override { return 3; }
    bool Plus(const double *x, const double *delta, double *out) const override {
        Map<V3> result(out);
        result = Map<const V3>(x) + body_from_stereo.transpose() * Map<const V3>(delta);
        return true;
    }
    bool PlusJacobian(const double *, double *out) const override {
        Map<M3r> result(out);
        result = body_from_stereo.transpose();
        return true;
    }
    bool Minus(const double *y, const double *x, double *out) const override {
        Map<V3> result(out);
        result = body_from_stereo * (Map<const V3>(y) - Map<const V3>(x));
        return true;
    }
    bool MinusJacobian(const double *, double *out) const override {
        Map<M3r> result(out);
        result = body_from_stereo;
        return true;
    }
};

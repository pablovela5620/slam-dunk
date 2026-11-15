// Compile with the release container's Ceres/Eigen toolchain; see README.md.
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <array>
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include "estimator/geometry.hpp"
#include "estimator/visual_factors.hpp"
#include "estimator/stereo.hpp"

void check(bool ok, const char *message) {
    if (!ok) {
        cerr << message << endl;
        exit(1);
    }
}
void quaternion(Q q, double *out) {
    out[0] = q.w(); out[1] = q.x(); out[2] = q.y(); out[3] = q.z();
}
int main() {
    double max_residual = 0, max_derivative = 0, max_prior = 0;
    ceres::QuaternionManifold manifold;
    for (int trial = 0; trial < 20; ++trial) {
        double fixed[7], q01[4], t01[3] = {.12, -.015, .003};
        quaternion(expq(V3(.08 * trial, -.3, .2)), fixed);
        fixed[4] = .03; fixed[5] = -.04; fixed[6] = .02;
        quaternion(expq(V3(.003 * trial, .02, -.01)), q01);
        StereoReference frame(fixed);
        double body_q[4], body_t[3], recovered_q[4], recovered_t[3];
        frame.compose(q01, t01, body_q, body_t);
        frame.relative(body_q, body_t, recovered_q, recovered_t);
        check((Map<Vector4d>(recovered_q)-Map<Vector4d>(q01)).norm() < 1e-14,
              "Relative rotation round trip failed");
        check((Map<V3>(recovered_t)-Map<V3>(t01)).norm() < 1e-14,
              "Relative translation round trip failed");
        StereoRotationManifold rotation_step(frame.rotation);
        StereoTranslationManifold translation_step(frame.rotation);
        double delta[3] = {.001,-.002,.003}, stepped_q[4], stepped_t[3],
               expected_q[4], composed_q[4], composed_t[3], recovered_delta[3];
        rotation_step.Plus(q01,delta,stepped_q);
        translation_step.Plus(t01,delta,stepped_t);
        frame.compose(stepped_q,stepped_t,composed_q,composed_t);
        manifold.Plus(body_q,delta,expected_q);
        check((Map<Vector4d>(composed_q)-Map<Vector4d>(expected_q)).norm()<1e-14,
              "Rotation step changed body-coordinate LM update");
        check((Map<V3>(composed_t)-Map<V3>(body_t)-Map<V3>(delta)).norm()<1e-14,
              "Translation step changed body-coordinate LM update");
        rotation_step.Minus(stepped_q,q01,recovered_delta);
        check((Map<V3>(recovered_delta)-Map<V3>(delta)).norm()<1e-14,"Rotation Minus failed");
        translation_step.Minus(stepped_t,t01,recovered_delta);
        check((Map<V3>(recovered_delta)-Map<V3>(delta)).norm()<1e-14,"Translation Minus failed");
        for (const ceres::Manifold *step : vector<const ceres::Manifold *>{&rotation_step,&translation_step}) {
            const double *state = step==&rotation_step ? q01 : t01;
            Matrix<double,Dynamic,Dynamic,RowMajor> plus(step->AmbientSize(),3), minus(3,step->AmbientSize());
            step->PlusJacobian(state,plus.data()); step->MinusJacobian(state,minus.data());
            check((minus*plus-M3::Identity()).norm()<1e-14,"Manifold Jacobians are inconsistent");
            for (int j=0;j<3;j++) {
                double d[3]={}, positive[4],negative[4]; d[j]=1e-6;
                step->Plus(state,d,positive);d[j]=-1e-6;step->Plus(state,d,negative);
                VectorXd numeric=(Map<VectorXd>(positive,step->AmbientSize())-
                                  Map<VectorXd>(negative,step->AmbientSize()))/(2e-6);
                check((numeric-plus.col(j)).norm()<1e-9,"Manifold Plus derivative failed");
            }
        }
        double qb[4], p[3] = {.5, -.1, .2}, x[3], ff[2] = {240, 241};
        double distortion[6] = {.02, -.01, .0001, -.0002, .00001, -.00001};
        quaternion(expq(V3(.02, .003 * trial, -.03)), qb);
        Map<V3> point(x);
        point = Q(qb[0],qb[1],qb[2],qb[3]) *
                    (Q(body_q[0],body_q[1],body_q[2],body_q[3]) * V3(.4,.1,3) +
                     Map<V3>(body_t)) + Map<V3>(p);
        double K[16] = {240,241,320,240,.01,-.002,.0001};
        double *args[7] = {qb,p,x,q01,t01,ff,distortion};
        const double *body_args[7] = {qb,p,x,body_q,body_t,ff,distortion};
        for (bool timed : {false, true}) {
            auto make = [&]() -> ceres::CostFunction * {
                if (timed) return new VisualTimed(K,330,250,.001,V3(.2,.1,-.3),V3(.4,.2,.1));
                return new VisualAnalytic(K,330,250);
            };
            StereoVisual cost(make(), frame);
            unique_ptr<ceres::CostFunction> original(make());
            double r[2], before[2];
            cost.Evaluate(args,r,nullptr);
            original->Evaluate(body_args,before,nullptr);
            max_residual = max(max_residual,(Map<Vector2d>(r)-Map<Vector2d>(before)).norm());
            // Test the analytic path and the intrinsics-autodiff fallback separately.
            for (bool intrinsics : {false,true}) {
                int sizes[7] = {4,3,3,4,3,2,6};
                double storage[7][12], *jac[7];
                for (int b=0;b<7;b++) jac[b] = b==6 && !intrinsics ? nullptr : storage[b];
                cost.Evaluate(args,r,jac);
                for (int b=0;b<7;b++) {
                    if (!jac[b]) continue;
                    int local = b==0 || b==3 ? 3 : sizes[b];
                    MatrixXd analytic = Map<Matrix<double,2,Dynamic,RowMajor>>(jac[b],2,sizes[b]);
                    if (local != sizes[b]) {
                        Matrix<double,4,3,RowMajor> plus;
                        manifold.PlusJacobian(args[b],plus.data());
                        analytic = (analytic * plus).eval();
                    }
                    for (int j=0;j<local;j++) {
                        double saved[6], delta[6] = {}, rp[2], rm[2];
                        copy(args[b],args[b]+sizes[b],saved);
                        constexpr double eps=1e-6;
                        for (int sign : {-1,1}) {
                            delta[j]=sign*eps;
                            if (local != sizes[b]) manifold.Plus(saved,delta,args[b]);
                            else { copy(saved,saved+sizes[b],args[b]); args[b][j]+=sign*eps; }
                            cost.Evaluate(args,sign==1 ? rp : rm,nullptr);
                        }
                        copy(saved,saved+sizes[b],args[b]);
                        Vector2d numeric=(Map<Vector2d>(rp)-Map<Vector2d>(rm))/(2*eps);
                        max_derivative=max(max_derivative,(numeric-analytic.col(j)).norm() /
                                           max(1.,numeric.norm()));
                    }
                }
            }
        }
        double next_q[4], next_t[3] = {.1202,-.0153,.0031}, next_body_q[4], next_body_t[3];
        quaternion(expq(V3(.0001,-.0002,.0003))*Q(q01[0],q01[1],q01[2],q01[3]),next_q);
        frame.compose(next_q,next_t,next_body_q,next_body_t);
        double old_r[3], new_r[3];
        RotDiff(.001)(body_q,next_body_q,old_r);
        RotDiff(.001)(q01,next_q,new_r);
        max_prior=max(max_prior,(Map<V3>(old_r)-Map<V3>(new_r)).norm());
        Diff(.0005)(body_t,next_body_t,old_r); Diff(.0005)(t01,next_t,new_r);
        max_prior=max(max_prior,abs(Map<V3>(old_r).squaredNorm()-Map<V3>(new_r).squaredNorm()));
        NormPrior(Map<V3>(fixed+4),.12,.0005)(body_t,old_r);
        NormPrior(V3::Zero(),.12,.0005)(t01,new_r);
        max_prior=max(max_prior,abs(old_r[0]-new_r[0]));
    }
    check(max_residual < 1e-10,"Projection changed");
    check(max_derivative < 2e-6,"Stereo projection Jacobian failed finite differences");
    check(max_prior < 1e-10,"Calibration prior changed");
    cout << "max residual difference " << max_residual << ", relative Jacobian error "
         << max_derivative << ", prior difference " << max_prior << endl;
}

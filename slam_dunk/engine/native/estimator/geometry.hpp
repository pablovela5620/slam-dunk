#pragma once

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
using V6 = Matrix<double, 6, 1>;
using M36 = Matrix<double, 3, 6>;
M3 symmetric(const V6 &s) {
    M3 m;
    m << s[0], s[3], s[4], s[3], s[1], s[5], s[4], s[5], s[2];
    return m;
}
M36 accDesign(const V3 &a) {
    M36 m;
    m << a.x(), 0, 0, a.y(), a.z(), 0, 0, a.y(), 0, a.x(), 0, a.z(), 0, 0, a.z(), 0, a.x(), a.y();
    return m;
}
struct Pose {
    double q[4], p[3], v[3];
    Q rot() const {
        return Q(q[0], q[1], q[2], q[3]);
    }
};

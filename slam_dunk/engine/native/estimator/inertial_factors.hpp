#pragma once

struct Inertial {
    Pre z;
    double gs, as;
    Inertial(Pre p, double g, double a) : z(p), gs(g), as(a) {
    }
    template <class T>
    bool operator()(const T *qi, const T *pi, const T *vi, const T *qj, const T *pj, const T *vj,
                    const T *bg, const T *ba, const T *gg, const T *ar, const T *sc, const T *gm,
                    const T *tm, const T *gt, T *r) const {
        Matrix<T, 2, 1> dtime(tm[0] - T(z.tm[0]), tm[1] - T(z.tm[1]));
        using V = Matrix<T, 3, 1>;
        V dgt(gt[0] - T(z.gt[0]), gt[1] - T(z.gt[1]), gt[2] - T(z.gt[2]));
        Matrix<T, 9, 1> dg;
        for (int k = 0; k < 9; k++)
            dg[k] = gm[k] - T(z.gm[k]);
        Quaternion<T> ai(qi[0], qi[1], qi[2], qi[3]), aj(qj[0], qj[1], qj[2], qj[3]);
        V db(bg[0] - T(z.bg[0]), bg[1] - T(z.bg[1]), bg[2] - T(z.bg[2])),
            da(ba[0] - T(z.ba[0]), ba[1] - T(z.ba[1]), ba[2] - T(z.ba[2])), g(gg[0], gg[1], gg[2]);
        Matrix<T, 6, 1> dsc;
        for (int k = 0; k < 6; k++)
            dsc[k] = sc[k] - T(z.sc[k]);
        V dar(ar[0] - T(z.ar[0]), ar[1] - T(z.ar[1]), ar[2] - T(z.ar[2]));
        V dr = z.Jr.cast<T>() * db + z.Jrg.cast<T>() * dg + z.Jtr.cast<T>() * dtime +
               z.Jrt.cast<T>() * dgt;
        Quaternion<T> dq(T(1), dr[0] / T(2), dr[1] / T(2), dr[2] / T(2)),
            er = (z.r.cast<T>() * dq).conjugate() * (ai.conjugate() * aj);
        T sn = er.w() < T(0) ? T(-1) : T(1);
        double sr = gs * sqrt(z.dt) + .0001, sv = as * sqrt(z.dt),
               sp = .5 * as * sqrt(z.dt) * z.dt + .001;
        for (int k = 0; k < 3; k++)
            r[k] = T(2 / sr) * sn * er.vec()[k];
        V dpi(pj[0] - pi[0] - vi[0] * T(z.dt), pj[1] - pi[1] - vi[1] * T(z.dt),
              pj[2] - pi[2] - vi[2] * T(z.dt)),
            dvi(vj[0] - vi[0], vj[1] - vi[1], vj[2] - vi[2]);
        V ep = ai.conjugate() * (dpi - g * T(z.dt * z.dt / 2)) - z.p.cast<T>() -
               z.Jpg.cast<T>() * db - z.Jpa.cast<T>() * da - z.Jarp.cast<T>() * dar -
               z.Jscp.cast<T>() * dsc - z.Jpgm.cast<T>() * dg - z.Jtp.cast<T>() * dtime -
               z.Jpt.cast<T>() * dgt,
          ev = ai.conjugate() * (dvi - g * T(z.dt)) - z.v.cast<T>() - z.Jvg.cast<T>() * db -
               z.Jva.cast<T>() * da - z.Jarv.cast<T>() * dar - z.Jscv.cast<T>() * dsc -
               z.Jvgm.cast<T>() * dg - z.Jtv.cast<T>() * dtime - z.Jvt.cast<T>() * dgt;
        for (int k = 0; k < 3; k++) {
            r[3 + k] = ev[k] / T(sv);
            r[6 + k] = ep[k] / T(sp);
        }
        return true;
    }
};
#include "inertial_cross.h"

import numpy as np
from numba import njit


@njit(cache=True)
def mul(a, b):
    return np.array(
        [
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        ]
    )


@njit(cache=True)
def ex(w, dt):
    a = np.sqrt((w * w).sum()) * dt
    fac = dt * 0.5 if abs(a) < 1e-10 else np.sin(a * 0.5) / (a / dt)
    return np.array([w[0] * fac, w[1] * fac, w[2] * fac, np.cos(a * 0.5)])


@njit(cache=True)
def integrate(t, w, ft):
    ans = np.zeros((len(ft), 4))
    q = np.array([0.0, 0.0, 0.0, 1.0])
    j = 0
    while j < len(ft) and ft[j] <= t[0]:
        ans[j] = q
        j += 1
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        if dt <= 0:
            continue
        wm = (w[i] + w[i + 1]) * 0.5
        while j < len(ft) and ft[j] <= t[i + 1]:
            dd = max(0.0, ft[j] - t[i])
            v = (w[i] + (w[i] + (w[i + 1] - w[i]) * (dd / dt))) * 0.5
            ans[j] = mul(q, ex(v, dd))
            j += 1
        q = mul(q, ex(wm, dt))
        q /= np.sqrt((q * q).sum())
        if j == len(ft):
            break
    while j < len(ft):
        ans[j] = q
        j += 1
    return ans

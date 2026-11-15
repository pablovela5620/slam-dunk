// Reconsider unused correspondences only after global pose and calibration refinement.
void recover_tracks(const string &path, vector<Pose> &poses, vector<array<double, 3>> &points,
                    vector<array<double, 5>> &obs, const vector<int> &ci,
                    const vector<array<double, 4>> &stereo_q, const vector<array<double, 3>> &stereo_t,
                    const vector<array<double, 4>> &fcs, double *K, double *ext, double *intr,
                    const vector<array<double, 7>> &metadata, double *heights, bool usepls,
                    double pls, const vector<Pose> &initialposes, vector<bool> &used,
                    vector<double> &observation_weights, bool refresh_velocity = false) {
    const StereoReference reference(ext);
    ifstream f(path + ".candidates", ios::binary);
    if (!f)
        return;
    int64_t H[2];
    f.read((char *)H, sizeof(H));
    int M = H[0], O = H[1];
    if (M < 0 || M > 2000000 || O < 0 || O > 10000000)
        return;
    vector<int64_t> old(M);
    f.read((char *)old.data(), M * sizeof(int64_t));
    vector<array<double, 8>> rows(O);
    for (auto &x : rows)
        rd(f, x.data(), 8);
    if (!f)
        return;
    vector<double> candidate_precision(O, 1.);
    if (getenv("SCALE_WEIGHT")) {
        ifstream wf(path + ".candidateweights", ios::binary);
        if (wf)
            rd(wf, candidate_precision.data(), O);
    }
    for (double &w : candidate_precision)
        if (!isfinite(w) || w <= 0)
            w = 1.;
    vector<vector<int>> groups(M);
    for (int n = 0; n < O; n++) {
        int k = int(rows[n][0]);
        if (k >= 0 && k < M)
            groups[k].push_back(n);
    }
    struct Ray {
        int i, c;
        V3 o, d;
        M3 R;
        double u, v, precision = 1.;
    };
    auto ray = [&](const array<double, 8> &x) {
        Ray z;
        z.i = int(x[1]);
        z.c = int(x[2]);
        z.u = x[3];
        z.v = x[4];
        int k = ci[z.i], c = z.c;
        double body_q[4], body_t[3];
        const double *eq = ext, *et = ext + 4;
        if (c == 1) {
            reference.compose(stereo_q[k].data(), stereo_t[k].data(), body_q, body_t);
            eq = body_q;
            et = body_t;
        }
        Q rc(eq[0], eq[1], eq[2], eq[3]), rb = poses[z.i].rot();
        V3 pp = Map<V3>(poses[z.i].p);
        if (usepls) {
            double ro = .00912 * clamp(z.v / heights[c], 0., 1.), ex = max(metadata[z.i][c], 1e-6),
                   delay = .5 * ro * .01 * (1 + ro / ex) / (1 + .01 * ro / ex) * pls;
            V3 w(metadata[z.i][2], metadata[z.i][3], metadata[z.i][4]);
            rb = rb * expq(w * delay);
            pp += Map<const V3>(refresh_velocity ? poses[z.i].v : initialposes[z.i].v) * delay;
        }
        V3 local(x[5], x[6], 1.);
        Visual m(K + 16 * c, z.u, z.v);
        for (int it = 0; it < 4; it++) {
            double r[2];
            Matrix<double, 2, 3> J;
            manualProject(m, local, fcs[k].data() + 2 * c, intr + 6 * c, r, &J);
            double det = J(0, 0) * J(1, 1) - J(0, 1) * J(1, 0);
            if (abs(det) < 1e-8)
                break;
            local[0] -= (J(1, 1) * r[0] - J(0, 1) * r[1]) / det;
            local[1] -= (-J(1, 0) * r[0] + J(0, 0) * r[1]) / det;
        }
        z.R = (rb * rc).toRotationMatrix();
        z.o = pp + rb * Map<const V3>(et);
        z.d = z.R * local.normalized();
        return z;
    };
    double maxdepth = envd("RECOVER_DEPTH", 250);
    auto error = [&](const Ray &z, const V3 &X) {
        V3 p = z.R.transpose() * (X - z.o);
        if (p.z() <= .2 || p.norm() > maxdepth)
            return 1e9;
        double r[2];
        manualProject(Visual(K + 16 * z.c, z.u, z.v), p, fcs[ci[z.i]].data() + 2 * z.c,
                      intr + 6 * z.c, r, nullptr);
        return hypot(r[0], r[1]);
    };
    // Each existing track has at most one observation in an image.
    vector<vector<int>> present(points.size());
    for (auto &o : obs)
        present[int(o[2])].push_back(2 * int(o[0]) + int(o[1]));
    for (auto &v : present)
        sort(v.begin(), v.end());
    int added = 0, newpoints = 0;
    double gate = envd("RECOVER_GATE", 1.5);
    for (int k = 0; k < M; k++) {
        auto &ix = groups[k];
        if (ix.size() < 4)
            continue;
        if (getenv("RECOVER_OLD_ONLY") && old[k] < 0)
            continue;
        if (getenv("RECOVER_NEW_ONLY") && old[k] >= 0)
            continue;
        vector<Ray> rays;
        for (int n : ix) {
            Ray z = ray(rows[n]);
            z.precision = candidate_precision[n];
            rays.push_back(z);
        }
        int p = old[k];
        V3 X;
        if (p >= 0 && p < int(points.size())) {
            if (present[p].size() < 2)
                continue;
            X = Map<V3>(points[p].data());
            for (auto &z : rays)
                if (!binary_search(present[p].begin(), present[p].end(), 2 * z.i + z.c) &&
                    error(z, X) < gate) {
                    obs.push_back({double(z.i), double(z.c), double(p), z.u, z.v});
                    observation_weights.push_back(z.precision);
                    used[p] = true;
                    added++;
                }
        } else {
            int low = 10000000, high = -1;
            double baseline = 0;
            for (auto &z : rays) {
                low = min(low, z.i);
                high = max(high, z.i);
                baseline = max(baseline, (z.o - rays[0].o).norm());
            }
            if (high - low < 2 || baseline < .3)
                continue;
            M3 A = M3::Zero();
            V3 b = V3::Zero();
            for (auto &z : rays) {
                M3 a = M3::Identity() - z.d * z.d.transpose();
                A += a;
                b += a * z.o;
            }
            if (A.determinant() < 1e-10)
                continue;
            X = A.ldlt().solve(b);
            if (!X.allFinite())
                continue;
            for (int it = 0; it < 3; it++) {
                A.setZero();
                b.setZero();
                for (auto &z : rays) {
                    double d = (X - z.o).squaredNorm(), e = error(z, X);
                    double w = 1 / max(d, .1) * min(1., gate / max(e, 1e-5));
                    M3 a = w * (M3::Identity() - z.d * z.d.transpose());
                    A += a;
                    b += a * z.o;
                }
                X = A.ldlt().solve(b);
            }
            vector<Ray> valid;
            for (auto &z : rays)
                if (error(z, X) < gate)
                    valid.push_back(z);
            if (valid.size() < 4 || valid.size() < .7 * rays.size())
                continue;
            low = 10000000;
            high = -1;
            for (auto &z : valid) {
                low = min(low, z.i);
                high = max(high, z.i);
            }
            if (high - low < 2)
                continue;
            double parallax = 0;
            for (auto &z : valid)
                parallax = max(parallax, z.d.cross(valid[0].d).norm());
            if (parallax < envd("RECOVER_PARALLAX", .01))
                continue;
            p = points.size();
            points.push_back({X[0], X[1], X[2]});
            used.push_back(true);
            newpoints++;
            for (auto &z : valid) {
                obs.push_back({double(z.i), double(z.c), double(p), z.u, z.v});
                observation_weights.push_back(z.precision);
                added++;
            }
        }
    }
    cerr << "RECOVERED " << added << " observations " << newpoints << " points" << endl;
}

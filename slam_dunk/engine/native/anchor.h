#pragma once
struct Anchor {
    static constexpr int N = 64;
    Eigen::Matrix<float, 6, N> Ji;
    Eigen::Matrix<float, N, 1> ref;
    Eigen::Matrix2f warp = Eigen::Matrix2f::Identity();
    bool valid = false;
    static float pix(const cv::Mat &im, float x, float y) {
        int ix = int(x), iy = int(y);
        float a = x - ix, b = y - iy;
        const uchar *p = im.ptr<uchar>(iy) + ix;
        float v = p[0] * (1 - a) + p[1] * a;
        const uchar *q = p + im.step;
        return v * (1 - b) + (q[0] * (1 - a) + q[1] * a) * b;
    }
    static Eigen::Vector2f pattern(int i) {
        return {float((i % 8) * 2 - 7), float((i / 8) * 2 - 7)};
    }
    Anchor() = default;
    Anchor(const cv::Mat &im, cv::Point2f p) {
        if (p.x < 10 || p.y < 10 || p.x > im.cols - 11 || p.y > im.rows - 11)
            return;
        Eigen::Matrix<float, N, 6> J;
        float mean = 0;
        for (int i = 0; i < N; i++) {
            auto d = pattern(i);
            float x = p.x + d.x(), y = p.y + d.y();
            float v = pix(im, x, y), gx = .5f * (pix(im, x + 1, y) - pix(im, x - 1, y)),
                  gy = .5f * (pix(im, x, y + 1) - pix(im, x, y - 1));
            ref[i] = v;
            mean += v;
            J.row(i) << gx, gy, gx * d.x(), gx * d.y(), gy * d.x(), gy * d.y();
        }
        mean /= N;
        if (mean < 3)
            return;
        Eigen::Matrix<float, 1, 6> jmean = J.colwise().mean();
        for (int i = 0; i < N; i++)
            J.row(i) = (J.row(i) - jmean * (ref[i] / mean)) / mean;
        ref /= mean;
        Eigen::Matrix<float, 6, 6> H = J.transpose() * J;
        H.diagonal().array() += 1e-7;
        Ji = H.ldlt().solve(J.transpose());
        valid = Ji.array().isFinite().all();
    }
    bool refine(const cv::Mat &im, cv::Point2f &p, float maxerr) {
        if (!valid)
            return false;
        Eigen::Vector2f pos(p.x, p.y);
        auto A = warp;
        float err = 10;
        for (int it = 0; it < 7; it++) {
            Eigen::Matrix<float, N, 1> v;
            float mean = 0;
            for (int i = 0; i < N; i++) {
                auto x = A * pattern(i) + pos;
                if (x.x() < 2 || x.y() < 2 || x.x() >= im.cols - 3 || x.y() >= im.rows - 3)
                    return false;
                v[i] = pix(im, x.x(), x.y());
                mean += v[i];
            }
            mean /= N;
            if (mean < 2)
                return false;
            auto residual = (v / mean - ref).eval();
            err = sqrt(residual.squaredNorm() / N);
            Eigen::Matrix<float, 6, 1> inc = -Ji * residual;
            if (!inc.array().isFinite().all() || inc.head<2>().norm() > 3 ||
                inc.tail<4>().norm() > .5)
                return false;
            Eigen::Matrix2f B;
            B << 1 + inc[2], inc[3], inc[4], 1 + inc[5];
            pos += A * inc.head<2>();
            A = A * B;
            if (inc.head<2>().norm() < .005 && inc.tail<4>().norm() < .001)
                break;
        }
        if (err > maxerr || A.determinant() < .3 || A.determinant() > 3)
            return false;
        warp = A;
        p = {pos.x(), pos.y()};
        return true;
    }
};

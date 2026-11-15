#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>
extern "C" int64_t union_tracks(int64_t n, int64_t nv, const int64_t *off, int64_t ne,
                                const int32_t *edges, int32_t *parent, int check) {
    std::iota(parent, parent + n, 0);
    std::vector<uint8_t> rank(n, 0);
    std::vector<int32_t> view(n);
    for (int c = 0; c < nv; c++)
        std::fill(view.begin() + off[c], view.begin() + off[c + 1], c);
    std::vector<std::vector<int32_t>> seen(check ? n : 0);
    auto find = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    int64_t rejected = 0;
    for (int64_t i = 0; i < ne; i++) {
        int u = find(edges[2 * i]), v = find(edges[2 * i + 1]);
        if (u == v)
            continue;
        if (check) {
            if (seen[u].empty())
                seen[u].push_back(view[u]);
            if (seen[v].empty())
                seen[v].push_back(view[v]);
            bool conflict = false;
            size_t a = 0, b = 0;
            while (a < seen[u].size() && b < seen[v].size()) {
                if (seen[u][a] == seen[v][b]) {
                    conflict = true;
                    break;
                }
                if (seen[u][a] < seen[v][b])
                    a++;
                else
                    b++;
            }
            if (conflict) {
                rejected++;
                continue;
            }
        }
        if (rank[u] < rank[v])
            std::swap(u, v);
        parent[v] = u;
        if (rank[u] == rank[v])
            rank[u]++;
        if (check) {
            std::vector<int32_t> merged;
            merged.reserve(seen[u].size() + seen[v].size());
            std::merge(seen[u].begin(), seen[u].end(), seen[v].begin(), seen[v].end(),
                       std::back_inserter(merged));
            seen[u].swap(merged);
            std::vector<int32_t>().swap(seen[v]);
        }
    }
    for (int64_t i = 0; i < n; i++)
        parent[i] = find(i);
    return rejected;
}

// ─────────────────────────────────────────────────────────────────────────────
// echo-agent · C++ Cognitive Engine (pybind11 bindings)
//
// Exposes two hot-path primitives to the Python microkernel with near-zero
// call overhead:
//
//   1. VectorIndex  — flat vector store with cached L2 norms and an exact
//                      cosine-similarity top-k query (SM retrieval kernel).
//   2. CauseGraph   — directed weighted cause-and-effect graph (episodic
//                      structure) with BFS reachability and shortest paths.
//
// Everything is pure STL: no external deps beyond pybind11, deterministic
// performance, and POD-level memory locality for cache-friendly scans.
// ─────────────────────────────────────────────────────────────────────────────

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// VectorIndex: exact cosine top-k over a flat store.
// Norms are cached on upsert so queries never recompute magnitudes; the scan
// is a tight dot-product loop over contiguous float blocks.
// ─────────────────────────────────────────────────────────────────────────────
class VectorIndex {
public:
    /// Insert or replace a vector. Idempotent per id.
    void upsert(const std::string& id, const std::vector<float>& vec) {
        std::lock_guard<std::mutex> lock(mu_);
        float norm = 0.0f;
        for (float v : vec) norm += v * v;
        norms_[id] = std::sqrt(norm);
        vectors_[id] = vec;   // copy keeps the store self-contained
    }

    void remove(const std::string& id) {
        std::lock_guard<std::mutex> lock(mu_);
        vectors_.erase(id);
        norms_.erase(id);
    }

    /// Exact top-k by cosine similarity. Ties break by insertion hash order —
    /// acceptable for recall purposes; callers wanting stability may sort ids.
    std::vector<std::pair<std::string, double>> top_k(
        const std::vector<float>& query, size_t k) const {
        std::lock_guard<std::mutex> lock(mu_);
        float qnorm = 0.0f;
        for (float v : query) qnorm += v * v;
        qnorm = std::sqrt(qnorm);
        if (qnorm == 0.0f) return {};

        // Min-heap of (similarity, id); pop worst as better candidates arrive.
        using Scored = std::pair<double, const std::string*>;
        std::priority_queue<Scored, std::vector<Scored>, std::greater<Scored>> heap;

        for (const auto& [id, vec] : vectors_) {
            float denom = norms_.at(id) * qnorm;
            if (denom == 0.0f) continue;
            const size_t n = std::min(vec.size(), query.size());
            float dot = 0.0f;
            for (size_t i = 0; i < n; ++i) dot += vec[i] * query[i];
            double sim = static_cast<double>(dot) / static_cast<double>(denom);
            if (heap.size() < k) {
                heap.push({sim, &id});
            } else if (sim > heap.top().first) {
                heap.pop();
                heap.push({sim, &id});
            }
        }

        std::vector<std::pair<std::string, double>> out;
        out.reserve(heap.size());
        while (!heap.empty()) {
            const auto& [sim, idp] = heap.top();
            out.emplace_back(*idp, sim);
            heap.pop();
        }
        std::reverse(out.begin(), out.end());   // best-first
        return out;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return vectors_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        vectors_.clear();
        norms_.clear();
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<std::string, std::vector<float>> vectors_;
    std::unordered_map<std::string, float> norms_;
};

// ─────────────────────────────────────────────────────────────────────────────
// CauseGraph: directed weighted cause→effect edges with BFS traversal.
// Maintains forward and reverse adjacency for O(1) degree queries.
// ─────────────────────────────────────────────────────────────────────────────
class CauseGraph {
public:
    /// Create/replace an edge cause→effect with a strength weight.
    void add_edge(const std::string& cause, const std::string& effect, double weight = 1.0) {
        std::lock_guard<std::mutex> lock(mu_);
        adj_[cause][effect] = weight;
        radj_[effect][cause] = weight;
    }

    void remove_edge(const std::string& cause, const std::string& effect) {
        std::lock_guard<std::mutex> lock(mu_);
        if (auto it = adj_.find(cause); it != adj_.end()) {
            it->second.erase(effect);
            if (it->second.empty()) adj_.erase(it);
        }
        if (auto it = radj_.find(effect); it != radj_.end()) {
            it->second.erase(cause);
            if (it->second.empty()) radj_.erase(it);
        }
    }

    void remove_node(const std::string& node) {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& [_, out] : adj_) out.erase(node);
        for (auto& [_, in] : radj_) in.erase(node);
        adj_.erase(node);
        radj_.erase(node);
    }

    std::vector<std::string> nodes() const {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<std::string> out;
        out.reserve(adj_.size());
        for (const auto& [n, _] : adj_) out.push_back(n);
        for (const auto& [n, _] : radj_)
            if (!adj_.count(n)) out.push_back(n);
        return out;
    }

    /// Forward neighbors with edge weights.
    std::vector<std::pair<std::string, double>> neighbors(const std::string& node) const {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<std::pair<std::string, double>> out;
        if (auto it = adj_.find(node); it != adj_.end())
            for (const auto& [nxt, w] : it->second) out.emplace_back(nxt, w);
        return out;
    }

    /// BFS from `start` following forward edges up to `max_depth`.
    std::vector<std::string> bfs(const std::string& start, size_t max_depth = 4) const {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<std::string> order;
        if (!adj_.count(start)) return order;
        std::unordered_set<std::string> visited{start};
        std::queue<std::pair<std::string, size_t>> q;
        q.push({start, 0});
        while (!q.empty()) {
            auto [node, depth] = q.front();
            q.pop();
            order.push_back(node);
            if (depth >= max_depth) continue;
            auto it = adj_.find(node);
            if (it == adj_.end()) continue;
            for (const auto& [nxt, _] : it->second) {
                if (visited.insert(nxt).second) q.push({nxt, depth + 1});
            }
        }
        return order;
    }

    /// Shortest path (fewest hops) start→goal; empty when unreachable.
    std::vector<std::string> shortest_path(const std::string& start, const std::string& goal) const {
        std::lock_guard<std::mutex> lock(mu_);
        if (start == goal) return {start};
        std::unordered_map<std::string, std::string> parent;
        std::queue<std::string> q;
        q.push(start);
        while (!q.empty()) {
            std::string node = q.front();
            q.pop();
            auto it = adj_.find(node);
            if (it == adj_.end()) continue;
            for (const auto& [nxt, _] : it->second) {
                if (parent.emplace(nxt, node).second) {
                    if (nxt == goal) {
                        std::vector<std::string> path{goal};
                        std::string cur = goal;
                        while (cur != start) {
                            cur = parent.at(cur);
                            path.push_back(cur);
                        }
                        std::reverse(path.begin(), path.end());
                        return path;
                    }
                    q.push(nxt);
                }
            }
        }
        return {};
    }

    size_t node_count() const {
        std::lock_guard<std::mutex> lock(mu_);
        std::unordered_set<std::string> seen;
        for (const auto& [n, _] : adj_) seen.insert(n);
        for (const auto& [n, _] : radj_) seen.insert(n);
        return seen.size();
    }

    size_t edge_count() const {
        std::lock_guard<std::mutex> lock(mu_);
        size_t n = 0;
        for (const auto& [_, out] : adj_) n += out.size();
        return n;
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<std::string, std::unordered_map<std::string, double>> adj_;
    std::unordered_map<std::string, std::unordered_map<std::string, double>> radj_;
};

// ─────────────────────────────────────────────────────────────────────────────
// pybind11 module surface — mirrors the Python-native fallback API in
// python_core/plugins/memory/storage.py so the memory plugin can swap this in
// transparently when the extension is importable.
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(cpp_memory, m) {
    m.doc() = "echo-agent C++ cognitive engine: exact vector top-k + cause-effect graph";
    m.attr("__version__") = "0.1.0";

    py::class_<VectorIndex>(m, "VectorIndex")
        .def(py::init<>())
        .def("upsert", &VectorIndex::upsert, py::arg("id"), py::arg("vector"),
             "Insert/replace a vector under an id")
        .def("remove", &VectorIndex::remove, py::arg("id"))
        .def("top_k", &VectorIndex::top_k, py::arg("query"), py::arg("k") = 5,
             "Exact cosine-similarity top-k; returns [(id, similarity)] best-first")
        .def("size", &VectorIndex::size)
        .def("clear", &VectorIndex::clear);

    py::class_<CauseGraph>(m, "CauseGraph")
        .def(py::init<>())
        .def("add_edge", &CauseGraph::add_edge, py::arg("cause"), py::arg("effect"),
             py::arg("weight") = 1.0)
        .def("remove_edge", &CauseGraph::remove_edge, py::arg("cause"), py::arg("effect"))
        .def("remove_node", &CauseGraph::remove_node, py::arg("node"))
        .def("nodes", &CauseGraph::nodes)
        .def("neighbors", &CauseGraph::neighbors, py::arg("node"))
        .def("bfs", &CauseGraph::bfs, py::arg("start"), py::arg("max_depth") = 4)
        .def("shortest_path", &CauseGraph::shortest_path, py::arg("start"), py::arg("goal"))
        .def("node_count", &CauseGraph::node_count)
        .def("edge_count", &CauseGraph::edge_count);
}

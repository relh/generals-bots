// Classic-only batched opponent derived from expander_python at cee053c.
// Public type/owner/army grids only. Memory is explicit and caller-owned.
// Frontier ties use row-major order (Python source uses set iteration).
#include <algorithm>
#include <array>
#include <cstdint>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>

namespace {
constexpr int N = 441;
using Action = std::array<int32_t, 5>;
const Action PASS{1, 0, 0, 0, 0};
struct Routes {
  std::array<int, N> distance, toward;
  std::vector<int> order;
  Routes() {
    distance.fill(-1);
    toward.fill(-1);
  }
};
struct Agent {
  int h, w, turn;
  const int32_t *type, *owner, *army;
  int32_t *memory; // city, remembered enemy general, spearhead; -1 = absent
  std::vector<int> owned;
  std::array<std::vector<int>, N> neighbors;
  Agent(int H, int W, int T, const int32_t *grid, int32_t *mem)
      : h(H), w(W), turn(T), type(grid), owner(grid + N), army(grid + 2 * N),
        memory(mem) {
    for (int r = 0; r < h; ++r)
      for (int c = 0; c < w; ++c) {
        int p = r * 21 + c;
        if (owner[p] == 1)
          owned.push_back(p);
        const int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
        for (int d = 0; d < 4; ++d) {
          int nr = r + dr[d], nc = c + dc[d], q = nr * 21 + nc;
          if (nr >= 0 && nr < h && nc >= 0 && nc < w && type[q] != 2 &&
              type[q] != 5)
            neighbors[p].push_back(q);
        }
      }
  }
  Action move(int p, int q, int split = 0) const {
    int delta = q - p, d = delta == -21  ? 0
                           : delta == 21 ? 1
                           : delta == -1 ? 2
                                         : 3;
    return {0, p / 21, p % 21, d, split};
  }
  Routes routes(const std::vector<int> &roots, int limit = N,
                bool all = false) const {
    Routes out;
    for (int p : roots) {
      out.distance[p] = 0;
      out.order.push_back(p);
    }
    for (size_t i = 0; i < out.order.size(); ++i) {
      int p = out.order[i];
      if (out.distance[p] >= limit)
        continue;
      for (int q : neighbors[p]) {
        if (out.distance[q] >= 0 || (!all && owner[q] != 1))
          continue;
        out.distance[q] = out.distance[p] + 1;
        out.toward[q] = p;
        out.order.push_back(q);
      }
    }
    return out;
  }
  Action gather(int target) const {
    auto route = routes({target});
    int best = -1;
    for (int p : route.order)
      if (p != target && army[p] > 1) {
        auto key = std::make_tuple(army[p], route.distance[p], -p);
        if (best < 0 ||
            key > std::make_tuple(army[best], route.distance[best], -best))
          best = p;
      }
    return best < 0 ? PASS
                    : move(best, route.toward[best],
                           turn >= 800 && (type[best] == 3 || type[best] == 4));
  }
  bool siege(Action &action) {
    int target = memory[1];
    if (target < 0)
      return false;
    if (type[target] == 3) {
      memory[1] = memory[2] = -1;
      return false;
    }
    int adjacent = -1, attacker = -1;
    for (int p : owned)
      if (std::abs(p / 21 - target / 21) + std::abs(p % 21 - target % 21) ==
          1) {
        if (adjacent < 0 || army[p] > army[adjacent])
          adjacent = p;
        if (army[p] > army[target] + 1 &&
            (attacker < 0 || army[p] > army[attacker]))
          attacker = p;
      }
    if (attacker >= 0) {
      action = move(attacker, target);
      return true;
    }
    if (adjacent >= 0) {
      memory[2] = adjacent;
      action = gather(adjacent);
      return true;
    }
    auto distance = routes({target}, N, true).distance;
    // Connected owned component surplus is invariant within a component.
    std::array<int, N> surplus;
    surplus.fill(-1);
    for (int p : owned)
      if (surplus[p] < 0) {
        auto component = routes({p});
        int total = 0;
        for (int q : component.order)
          total += std::max(0, army[q] - 1);
        for (int q : component.order)
          surplus[q] = total;
      }
    int source = -1, dest = -1;
    for (int p : owned) {
      if (distance[p] < 0 || surplus[p] <= 0)
        continue;
      int next = -1;
      for (int q : neighbors[p])
        if (distance[q] >= 0 && distance[q] < distance[p]) {
          auto key = std::make_tuple(distance[q], owner[q] != 1, army[q]);
          if (next < 0 || key < std::make_tuple(distance[next],
                                                owner[next] != 1, army[next]))
            next = q;
        }
      if (next < 0)
        continue;
      auto key =
          std::make_tuple(-distance[p], surplus[p], army[p] - army[next]);
      if (source < 0 ||
          key > std::make_tuple(-distance[source], surplus[source],
                                army[source] - army[dest])) {
        source = p;
        dest = next;
      }
    }
    if (source < 0)
      return false;
    memory[2] = source;
    if ((owner[dest] == 1 && army[source] > 1) ||
        (owner[dest] != 1 && army[source] > army[dest] + 1)) {
      memory[2] = dest;
      action = move(source, dest);
    } else
      action = gather(source);
    return true;
  }
  int openings(int p) const {
    int n = 0;
    for (int q : neighbors[p])
      n += owner[q] != 1;
    return n;
  }
  int edge(int p) const {
    return std::min({p / 21, h - 1 - p / 21, p % 21, w - 1 - p % 21});
  }
  Action act() {
    for (int r = 0; r < h; ++r) {
      bool found = false;
      for (int c = 0; c < w; ++c) {
        int p = r * 21 + c;
        if (type[p] == 4 && owner[p] == 2) {
          memory[1] = p;
          found = true;
          break;
        }
      }
      if (found)
        break;
    }
    Action action;
    if (siege(action))
      return action;
    if (turn >= 800) {
      int strongest = -1;
      for (int p : owned) {
        bool border = false;
        for (int q : neighbors[p])
          border |= owner[q] == 2;
        if (border && (strongest < 0 || army[p] > army[strongest]))
          strongest = p;
      }
      if (strongest >= 0) {
        bool advance = false;
        for (int q : neighbors[strongest])
          if (owner[q] == 2 && army[strongest] > army[q] + 1)
            advance = true;
        if (!advance) {
          action = gather(strongest);
          if (action[0] == 0) {
            memory[2] = strongest;
            return action;
          }
        }
      }
    }
    std::array<bool, N> frontier{}, cities{};
    using Key = std::tuple<int, int, int, int, int, Action, int, int>;
    Key best{};
    bool captures = false;
    for (int p : owned)
      for (int q : neighbors[p]) {
        if (owner[q] == 1)
          continue;
        if (type[q] == 3)
          cities[q] = true;
        if (type[q] == 0 || type[q] == 1 || owner[q] == 2)
          frontier[p] = true;
        if (army[p] <= army[q] + 1)
          continue;
        int priority = type[q] == 3 ? 2 : owner[q] == 2 ? 1 : 0;
        int rank = turn >= 800 && owner[q] == 2 ? army[p] : -army[p];
        Key key{priority, rank,       -army[q], openings(q),
                edge(q),  move(p, q), q,        owner[q]};
        if (!captures || key > best) {
          best = key;
          captures = true;
        }
      }
    if (captures && std::get<0>(best) >= 2) {
      memory[0] = -1;
      return std::get<5>(best);
    }
    std::vector<int> city_list;
    for (int p = 0; p < N; ++p)
      if (cities[p])
        city_list.push_back(p);
    std::stable_sort(city_list.begin(), city_list.end(), [&](int a, int b) {
      return std::make_pair(a != memory[0], a) <
             std::make_pair(b != memory[0], b);
    });
    for (int city : city_list) {
      std::vector<int> roots;
      for (int q : neighbors[city])
        if (owner[q] == 1)
          roots.push_back(q);
      std::stable_sort(roots.begin(), roots.end(),
                       [&](int a, int b) { return army[a] > army[b]; });
      for (int root : roots) {
        auto route = routes({root}, 6);
        int surplus = 0;
        for (int q : route.order)
          surplus += std::max(0, army[q] - 1);
        if (surplus <= army[city] + 2)
          continue;
        int source = -1;
        for (int q : route.order)
          if (q != root && army[q] > 1) {
            auto key = std::make_pair(army[q] - 1, -route.distance[q]);
            if (source < 0 ||
                key > std::make_pair(army[source] - 1, -route.distance[source]))
              source = q;
          }
        if (source >= 0) {
          memory[0] = city;
          return move(source, route.toward[source]);
        }
      }
    }
    memory[0] = -1;
    if (captures) {
      if (turn >= 800 && std::get<7>(best) == 2)
        memory[2] = std::get<6>(best);
      return std::get<5>(best);
    }
    std::vector<int> front;
    for (int p = 0; p < N; ++p)
      if (frontier[p])
        front.push_back(p);
    if (turn >= 800 && !front.empty()) {
      int source = memory[2];
      if (source < 0 || owner[source] != 1) {
        source = front[0];
        for (int p : front)
          if (army[p] > army[source])
            source = p;
        memory[2] = source;
      }
      int dest = -1;
      for (int q : neighbors[source])
        if (owner[q] != 1 && army[source] > army[q] + 1) {
          if (dest < 0 || std::make_pair(openings(q), edge(q)) >
                              std::make_pair(openings(dest), edge(dest)))
            dest = q;
        }
      if (dest >= 0) {
        memory[2] = dest;
        return move(source, dest);
      }
      action = gather(source);
      if (action[0] == 0)
        return action;
    }
    auto route = routes(front);
    int source = -1;
    for (int p : route.order)
      if (route.toward[p] >= 0 && army[p] > 1) {
        if (source < 0 ||
            std::make_pair(army[p], route.distance[p]) >
                std::make_pair(army[source], route.distance[source]))
          source = p;
      }
    return source < 0 ? PASS : move(source, route.toward[source]);
  }
};
} // namespace
extern "C" int classic_siege_batch(int count, const int32_t *dimensions,
                                   const int32_t *turns, const int32_t *grids,
                                   const int32_t *memories, int32_t *actions,
                                   int32_t *next_memories) {
  if (count < 0)
    return 1;
  for (int i = 0; i < count; ++i) {
    int h = dimensions[2 * i], w = dimensions[2 * i + 1];
    if (h < 1 || h > 21 || w < 1 || w > 21)
      return 2;
    for (int k = 0; k < 3; ++k) {
      int m = memories[3 * i + k];
      if (m < -1 || m >= N || (m >= 0 && (m / 21 >= h || m % 21 >= w)))
        return 3;
      next_memories[3 * i + k] = m;
    }
    Agent agent(h, w, turns[i], grids + i * 3 * N, next_memories + 3 * i);
    Action a = agent.act();
    std::copy(a.begin(), a.end(), actions + 5 * i);
  }
  return 0;
}

extern "C" int
classic_siege_batch_parallel(int count, const int32_t *dimensions,
                             const int32_t *turns, const int32_t *grids,
                             const int32_t *memories, int32_t *actions,
                             int32_t *next_memories, int workers) {
  if (count < 0 || workers < 1 || workers > 8)
    return 4;
  if (!count)
    return 0;
  workers = std::min(workers, count);
  std::array<int, 8> status{};
  // Each worker owns disjoint rows. Joining precedes every return, including
  // rejected inputs, so a callback never exposes buffers still being written.
  auto work = [&](int worker) {
    int begin =
        static_cast<int>(static_cast<int64_t>(count) * worker / workers);
    int end =
        static_cast<int>(static_cast<int64_t>(count) * (worker + 1) / workers);
    try {
      status[worker] = classic_siege_batch(
          end - begin, dimensions + 2 * begin, turns + begin,
          grids + 3 * N * begin, memories + 3 * begin, actions + 5 * begin,
          next_memories + 3 * begin);
    } catch (...) {
      status[worker] = 5;
    }
  };
  std::array<std::thread, 7> threads;
  for (int i = 1; i < workers; ++i) {
    try {
      threads[i - 1] = std::thread(work, i);
    } catch (const std::system_error &) {
      // Finish this chunk on the caller if the OS cannot create a thread.
      work(i);
    }
  }
  work(0);
  for (auto &thread : threads)
    if (thread.joinable())
      thread.join();
  for (int i = 0; i < workers; ++i)
    if (status[i])
      return status[i];
  return 0;
}

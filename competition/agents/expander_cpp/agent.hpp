// Edit this file to implement your agent.
//
// `Observation` is the parsed per-turn view from main.cpp.
// `Agent::act` is called once per turn and must return an `Action`:
//
//     pass:      0 to move, 1 to skip the turn, 2 to build a castle
//     row, col:  source cell of a move, or the cell to build on
//     dir:       0=up, 1=down, 2=left, 3=right (ignored for pass/build)
//     split:     0=move all-but-one armies, 1=move half (floor; ignored for pass/build)
//
// To build a castle on one of your own plain cells, return {2, r, c, 0, 0} —
// the price is paid from the army standing on (r, c) (see the rules page).
//
// Tile-type codes in obs.type_grid:
//     0=fog, 1=plain, 2=mountain, 3=castle, 4=general, 5=structure-in-fog
//
// Owner codes in obs.owner_grid (perspective-relative):
//     0=neutral/unknown, 1=me, 2=opp
#pragma once

#include <algorithm>
#include <tuple>
#include <vector>

struct Observation {
    int H, W, turn;
    int my_land, my_army, opp_land, opp_army;
    std::vector<int> type_grid;    // H*W row-major: type_grid[r*W + c]
    std::vector<int> owner_grid;
    std::vector<int> army_grid;
};

struct Action {
    int pass, row, col, dir, split;
};

class Agent {
public:
    Agent(int player_id, int H, int W)
        : player_id_(player_id), H_(H), W_(W) {}

    // Expander: each turn pick the move that maximizes
    //   score = src_army * (10 if expansion else 1) * (2 if opponent else 1)
    // among captures (src_army > dest_army + 1). Otherwise take the first
    // valid move; otherwise pass.
    Action act(const Observation& obs) {
        static const int dr[4] = {-1, 1, 0, 0};   // up, down, left, right
        static const int dc[4] = { 0, 0, -1, 1};

        const Action PASS{1, 0, 0, 0, 0};
        Action best = PASS;
        Action fallback = PASS;
        bool has_valid = false;
        std::tuple<int, int, int> best_key{-1, -1, -1};
        std::pair<int, int> fallback_key{-1, -1};

        for (int r = 0; r < obs.H; ++r) {
            for (int c = 0; c < obs.W; ++c) {
                int idx = r * obs.W + c;
                if (obs.owner_grid[idx] != 1) continue;
                int src_army = obs.army_grid[idx];
                if (src_army <= 1) continue;

                for (int d = 0; d < 4; ++d) {
                    int nr = r + dr[d], nc = c + dc[d];
                    if (nr < 0 || nr >= obs.H || nc < 0 || nc >= obs.W) continue;
                    int didx = nr * obs.W + nc;
                    int dtype = obs.type_grid[didx];
                    if (dtype == 2 || dtype == 5) continue;   // impassable

                    Action move{0, r, c, d, 0};
                    int frontier = 0;
                    for (int nd = 0; nd < 4; ++nd) {
                        int rr = nr + dr[nd], cc = nc + dc[nd];
                        if (rr < 0 || rr >= obs.H || cc < 0 || cc >= obs.W) continue;
                        int nidx = rr * obs.W + cc;
                        int ntype = obs.type_grid[nidx];
                        if (ntype != 2 && ntype != 5 && obs.owner_grid[nidx] != 1)
                            ++frontier;
                    }
                    int edge_clearance = std::min(std::min(nr, obs.H - 1 - nr),
                                                  std::min(nc, obs.W - 1 - nc));
                    std::pair<int, int> geometry{frontier, edge_clearance};
                    if (!has_valid || geometry > fallback_key) {
                        fallback = move;
                        fallback_key = geometry;
                        has_valid = true;
                    }

                    int dest_owner = obs.owner_grid[didx];
                    int dest_army = obs.army_grid[didx];
                    if (src_army <= dest_army + 1) continue;

                    bool is_opp = (dest_owner == 2);
                    bool is_expansion = dest_owner != 1;

                    int score = src_army;
                    if (is_expansion) score *= 10;
                    if (is_opp)       score *= 2;

                    std::tuple<int, int, int> key{score, frontier, edge_clearance};
                    if (key > best_key) {
                        best_key = key;
                        best = move;
                    }
                }
            }
        }

        if (std::get<0>(best_key) > 0) return best;
        if (has_valid)                 return fallback;
        return PASS;
    }

private:
    int player_id_, H_, W_;
};

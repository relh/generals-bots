"""
Edit this file to implement your agent.

`Agent.act(obs)` is called once per turn. The `obs` argument is built by
`main.py` from the wire-protocol frame and has these fields:

    obs.H, obs.W            board dimensions (constant for the whole game)
    obs.turn                current turn number, increments each step
    obs.my_land             total cells you own
    obs.my_army             total armies summed over your cells
    obs.opp_land            opponent's land count (visible at all times)
    obs.opp_army            opponent's army total (visible at all times)
    obs.type_grid[r][c]     0=fog, 1=plain, 2=mountain, 3=castle, 4=general, 5=structure-in-fog
    obs.owner_grid[r][c]    0=neutral/unknown, 1=me, 2=opp  (perspective-relative)
    obs.army_grid[r][c]     army count, 0 in fog or empty

`act` must return a 5-tuple `(pass, row, col, direction, split)`:

    pass:       0 to move, 1 to skip the turn, 2 to build a castle
    row, col:   source cell of a move (must be owned by you and have
                army > 1), or the cell to build a castle on
    direction:  0=up, 1=down, 2=left, 3=right (ignored for pass/build)
    split:      0=move all-but-one armies, 1=move half (floor division;
                ignored for pass/build)

To build a castle on one of your own plain cells, return (2, r, c, 0, 0) —
the price is paid from the army standing on (r, c) (see the rules page).

Invalid moves and builds are silently treated as a pass by the engine.
"""

# A no-op action — used when no valid move exists or as a safe default.
PASS = (1, 0, 0, 0, 0)

# (dr, dc) offsets for direction codes 0..3
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _is_passable(t):
    # Mountains (2) and fogged-structures (5) are impassable; everything else
    # can be entered (including fog — you just don't know what's there).
    return t != 2 and t != 5


class Agent:
    """Capture territory, gather for cities, and route without idle shuffling."""

    def __init__(self, player_id, H, W):
        self.player_id, self.H, self.W = player_id, H, W
        self.city = None

    def neighbors(self, obs, cell):
        r, c = cell
        for d, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < obs.H and 0 <= nc < obs.W and _is_passable(obs.type_grid[nr][nc]):
                yield d, (nr, nc)

    def routes(self, obs, roots, limit=None):
        from collections import deque
        distance = {cell: 0 for cell in roots}
        toward = {}
        queue = deque(roots)
        while queue:
            cell = queue.popleft()
            if limit is not None and distance[cell] >= limit:
                continue
            for _, nxt in self.neighbors(obs, cell):
                if nxt in distance or obs.owner_grid[nxt[0]][nxt[1]] != 1:
                    continue
                distance[nxt] = distance[cell] + 1
                toward[nxt] = cell
                queue.append(nxt)
        return distance, toward

    def move(self, source, destination):
        r, c = source
        return (0, r, c, DIRECTIONS.index((destination[0]-r, destination[1]-c)), 0)

    def act(self, obs):
        owned = [(r, c) for r in range(obs.H) for c in range(obs.W) if obs.owner_grid[r][c] == 1]
        captures, frontier, cities = [], set(), set()
        for cell in owned:
            r, c = cell
            for _, nxt in self.neighbors(obs, cell):
                nr, nc = nxt
                if obs.owner_grid[nr][nc] == 1:
                    continue
                kind, defense = obs.type_grid[nr][nc], obs.army_grid[nr][nc]
                if kind == 3:
                    cities.add(nxt)
                if kind in (0, 1) or obs.owner_grid[nr][nc] == 2:
                    frontier.add(cell)
                if obs.army_grid[r][c] <= defense + 1:
                    continue
                openings = sum(obs.owner_grid[a][b] != 1 for _, (a, b) in self.neighbors(obs, nxt))
                edge = min(nr, obs.H-1-nr, nc, obs.W-1-nc)
                # All captures beat friendly transfers. Cheap single-army
                # expansion uses distributed growth instead of wasting turns.
                priority = 3 if kind == 4 else 2 if kind == 3 else 1 if obs.owner_grid[nr][nc] == 2 else 0
                key = (priority, -defense, -obs.army_grid[r][c], openings, edge)
                captures.append((key, self.move(cell, nxt)))
        if captures and max(captures)[0][0] >= 2:
            self.city = None
            return max(captures)[1]

        # Commit only when an owned path connects enough nearby surplus to a
        # staging square. Transfers strictly approach that square; no ping-pong.
        candidates = sorted(cities, key=lambda cell: (cell != self.city, cell))
        for city in candidates:
            defense = obs.army_grid[city[0]][city[1]]
            roots = [n for _, n in self.neighbors(obs, city) if obs.owner_grid[n[0]][n[1]] == 1]
            for root in sorted(roots, key=lambda n: -obs.army_grid[n[0]][n[1]]):
                distance, toward = self.routes(obs, [root], limit=6)
                surplus = sum(max(0, obs.army_grid[r][c]-1) for r, c in distance)
                if surplus <= defense + 2:
                    continue
                sources = [n for n in toward if obs.army_grid[n[0]][n[1]] > 1]
                if sources:
                    self.city = city
                    source = max(sources, key=lambda n: (obs.army_grid[n[0]][n[1]] - 1, -distance[n]))
                    return self.move(source, toward[source])
        self.city = None
        if captures:
            return max(captures)[1]
        distance, toward = self.routes(obs, sorted(frontier))
        sources = [n for n in toward if obs.army_grid[n[0]][n[1]] > 1]
        if sources:
            source = max(sources, key=lambda n: (obs.army_grid[n[0]][n[1]], distance[n]))
            return self.move(source, toward[source])
        return PASS

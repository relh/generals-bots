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
    """Expand early, then concentrate one army into a persistent breakthrough."""

    PRESSURE_TURN = 800

    def __init__(self, player_id, H, W):
        self.player_id, self.H, self.W = player_id, H, W
        self.city = None
        self.enemy_general = None
        self.spearhead = None
        self.pressure_anchor = None

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

    def move(self, source, destination, split=0):
        r, c = source
        return (0, r, c, DIRECTIONS.index((destination[0]-r, destination[1]-c)), split)

    def gather(self, obs, target):
        """Move the strongest reachable surplus one owned step toward target."""
        distance, toward = self.routes(obs, [target])
        sources = [
            cell for cell in toward
            if cell != target and obs.army_grid[cell[0]][cell[1]] > 1
        ]
        if not sources:
            return None
        source = max(
            sources,
            key=lambda cell: (
                obs.army_grid[cell[0]][cell[1]],
                distance[cell],
                -cell[0],
                -cell[1],
            ),
        )
        # Once combat starts, do not empty a productive structure just to feed
        # the attack. Half moves still build the rally stack every turn.
        split = int(obs.turn >= self.PRESSURE_TURN and obs.type_grid[source[0]][source[1]] in (3, 4))
        return self.move(source, toward[source], split)

    def distances(self, obs, roots, forced=()):
        """Shortest visible-map distances through passable or forced cells."""
        from collections import deque

        forced = set(forced)
        distance = {cell: 0 for cell in roots}
        queue = deque(roots)
        while queue:
            cell = queue.popleft()
            for _, nxt in self.neighbors(obs, cell):
                if nxt in distance:
                    continue
                distance[nxt] = distance[cell] + 1
                queue.append(nxt)
            # A remembered general is reported as a structure in fog. It is a
            # legal destination and must remain a route root while hidden.
            r, c = cell
            for dr, dc in DIRECTIONS:
                nxt = (r + dr, c + dc)
                if (
                    nxt in forced
                    and 0 <= nxt[0] < obs.H
                    and 0 <= nxt[1] < obs.W
                    and nxt not in distance
                ):
                    distance[nxt] = distance[cell] + 1
                    queue.append(nxt)
        return distance

    def siege(self, obs, owned):
        """Advance toward a remembered general and preserve an understrength siege."""
        target = self.enemy_general
        if target is None:
            return None

        adjacent = [
            cell for cell in owned
            if abs(cell[0] - target[0]) + abs(cell[1] - target[1]) == 1
        ]
        if adjacent:
            defense = obs.army_grid[target[0]][target[1]]
            attackers = [
                cell for cell in adjacent
                if obs.army_grid[cell[0]][cell[1]] > defense + 1
            ]
            if attackers:
                source = max(attackers, key=lambda cell: obs.army_grid[cell[0]][cell[1]])
                return self.move(source, target)

            # The old policy abandoned exactly this position. Keep the largest
            # adjacent stack in place and feed it until it can take the crown.
            stage = max(adjacent, key=lambda cell: obs.army_grid[cell[0]][cell[1]])
            self.spearhead = stage
            return self.gather(obs, stage) or PASS

        distance = self.distances(obs, [target], forced=[target])
        advances = []
        for source in owned:
            if source not in distance:
                continue
            next_steps = [
                destination for _, destination in self.neighbors(obs, source)
                if destination in distance and distance[destination] < distance[source]
            ]
            if not next_steps:
                continue
            component, _ = self.routes(obs, [source])
            surplus = sum(
                max(0, obs.army_grid[cell[0]][cell[1]] - 1)
                for cell in component
            )
            if surplus <= 0:
                continue
            destination = min(
                next_steps,
                key=lambda cell: (
                    distance[cell],
                    obs.owner_grid[cell[0]][cell[1]] != 1,
                    obs.army_grid[cell[0]][cell[1]],
                ),
            )
            defense = obs.army_grid[destination[0]][destination[1]]
            advances.append((distance[source], surplus, source, destination, defense))
        if not advances:
            return None

        best_distance = min(row[0] for row in advances)
        near = [row for row in advances if row[0] == best_distance]
        _, _, source, destination, defense = max(
            near,
            key=lambda row: (
                row[1],
                obs.army_grid[row[2][0]][row[2][1]] - row[4],
            ),
        )
        self.spearhead = source
        if obs.owner_grid[destination[0]][destination[1]] == 1 and obs.army_grid[source[0]][source[1]] > 1:
            self.spearhead = destination
            return self.move(source, destination)
        if (
            obs.owner_grid[destination[0]][destination[1]] != 1
            and obs.army_grid[source[0]][source[1]] > defense + 1
        ):
            self.spearhead = destination
            return self.move(source, destination)
        return self.gather(obs, source) or PASS

    def late_explore(self, obs, owned, frontier, *, reset_interior=False):
        """After the opening, explore with one reinforced column, not scattered probes."""
        if not frontier:
            return None
        # A captured frontier tile becomes interior on the following turn. If
        # we keep it as the spearhead, gather() just shuttles armies back into
        # that dead end forever -- the exact failure seen in FFA turn-limit
        # draws after two opponents had already been eliminated.
        valid_spearheads = frontier if reset_interior else owned
        if self.spearhead not in valid_spearheads:
            self.spearhead = max(frontier, key=lambda cell: obs.army_grid[cell[0]][cell[1]])
        source = self.spearhead
        destinations = [
            nxt for _, nxt in self.neighbors(obs, source)
            if obs.owner_grid[nxt[0]][nxt[1]] != 1
        ]
        capturable = [
            cell for cell in destinations
            if obs.army_grid[source[0]][source[1]] > obs.army_grid[cell[0]][cell[1]] + 1
        ]
        if capturable:
            destination = max(
                capturable,
                key=lambda cell: (
                    sum(obs.owner_grid[n[0]][n[1]] != 1 for _, n in self.neighbors(obs, cell)),
                    min(cell[0], obs.H - 1 - cell[0], cell[1], obs.W - 1 - cell[1]),
                ),
            )
            self.spearhead = destination
            return self.move(source, destination)
        return self.gather(obs, source)

    def ffa_pressure(self, obs, owned, enemy_cells):
        """Keep one endgame rally point until it breaks through enemy land."""
        if obs.turn < self.PRESSURE_TURN or not enemy_cells:
            return None
        border = [
            cell for cell in owned
            if any(obs.owner_grid[n[0]][n[1]] == 2 for _, n in self.neighbors(obs, cell))
        ]
        if not border:
            self.pressure_anchor = None
            return None
        if self.pressure_anchor not in border:
            self.pressure_anchor = max(
                border,
                key=lambda cell: (obs.army_grid[cell[0]][cell[1]], -cell[0], -cell[1]),
            )
        source = self.pressure_anchor
        blockers = [
            nxt for _, nxt in self.neighbors(obs, source)
            if obs.owner_grid[nxt[0]][nxt[1]] == 2
        ]
        capturable = [
            cell for cell in blockers
            if obs.army_grid[source[0]][source[1]] > obs.army_grid[cell[0]][cell[1]] + 1
        ]
        if capturable:
            destination = max(
                capturable,
                key=lambda cell: (
                    obs.type_grid[cell[0]][cell[1]] == 4,
                    obs.type_grid[cell[0]][cell[1]] == 3,
                    -obs.army_grid[cell[0]][cell[1]],
                ),
            )
            self.pressure_anchor = destination
            self.spearhead = destination
            return self.move(source, destination)
        self.spearhead = source
        return self.gather(obs, source) or PASS

    def castle_opening(self, obs, owned):
        """Fund one early productive castle when the variant permits builds."""
        costs = getattr(obs, "build_cost_grid", None)
        if costs is None:
            return None
        if any(obs.type_grid[r][c] == 3 for r, c in owned):
            return None

        affordable = [
            (costs[r][c], r, c)
            for r, c in owned
            if obs.type_grid[r][c] == 1
            and costs[r][c] > 0
            and obs.army_grid[r][c] >= costs[r][c]
        ]
        if affordable:
            _, r, c = min(affordable)
            return (2, r, c, 0, 0)

        # The general reaches 50 armies around turn 98. Moving all but one to
        # an adjacent plain leaves enough to pay the initial 47-army cost on
        # the following turn, matching the source-published Builder baseline.
        if obs.turn <= 102:
            generals = [cell for cell in owned if obs.type_grid[cell[0]][cell[1]] == 4]
            if generals:
                general = generals[0]
                if obs.army_grid[general[0]][general[1]] < 50:
                    return PASS
                destinations = [
                    nxt for _, nxt in self.neighbors(obs, general)
                    if obs.type_grid[nxt[0]][nxt[1]] == 1
                    and obs.owner_grid[nxt[0]][nxt[1]] in (0, 1)
                    and obs.army_grid[nxt[0]][nxt[1]] == 0
                ]
                if destinations:
                    return self.move(general, destinations[0])
        return None

    def act(self, obs):
        owned = [(r, c) for r in range(obs.H) for c in range(obs.W) if obs.owner_grid[r][c] == 1]
        is_ffa = len(getattr(obs, "players", ())) == 4 or len(getattr(obs, "public_scores", ())) == 4
        castle_move = self.castle_opening(obs, owned)
        if castle_move is not None:
            return castle_move
        visible_generals = [
            (r, c) for r in range(obs.H) for c in range(obs.W)
            if obs.type_grid[r][c] == 4 and obs.owner_grid[r][c] == 2
        ]
        if visible_generals:
            self.enemy_general = visible_generals[0]
        siege_move = self.siege(obs, owned)
        if siege_move is not None:
            return siege_move

        enemy_cells = [
            (r, c) for r in range(obs.H) for c in range(obs.W)
            if obs.owner_grid[r][c] == 2
        ]
        if is_ffa:
            pressure_move = self.ffa_pressure(obs, owned, enemy_cells)
            if pressure_move is not None:
                return pressure_move
        if enemy_cells and obs.turn >= self.PRESSURE_TURN:
            border = [
                cell for cell in owned
                if any(obs.owner_grid[n[0]][n[1]] == 2 for _, n in self.neighbors(obs, cell))
            ]
            if border:
                strongest = max(border, key=lambda cell: obs.army_grid[cell[0]][cell[1]])
                blockers = [
                    nxt for _, nxt in self.neighbors(obs, strongest)
                    if obs.owner_grid[nxt[0]][nxt[1]] == 2
                ]
                can_advance = any(
                    obs.army_grid[strongest[0]][strongest[1]] > obs.army_grid[nxt[0]][nxt[1]] + 1
                    for nxt in blockers
                )
                if blockers and not can_advance:
                    reinforcement = self.gather(obs, strongest)
                    if reinforcement is not None:
                        self.spearhead = strongest
                        return reinforcement

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
                priority = 2 if kind == 3 else 1 if obs.owner_grid[nr][nc] == 2 else 0
                source_army = obs.army_grid[r][c]
                source_rank = (
                    source_army
                    if obs.turn >= self.PRESSURE_TURN and obs.owner_grid[nr][nc] == 2
                    else -source_army
                )
                key = (priority, source_rank, -defense, openings, edge)
                captures.append((key, self.move(cell, nxt), nxt, obs.owner_grid[nr][nc]))
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
        if is_ffa and obs.turn >= self.PRESSURE_TURN and not enemy_cells:
            late_move = self.late_explore(obs, owned, frontier, reset_interior=True)
            if late_move is not None:
                return late_move
        if captures:
            best = max(captures)
            if obs.turn >= self.PRESSURE_TURN and best[3] == 2:
                self.spearhead = best[2]
            return best[1]
        if obs.turn >= self.PRESSURE_TURN:
            late_move = self.late_explore(obs, owned, frontier)
            if late_move is not None:
                return late_move
        distance, toward = self.routes(obs, sorted(frontier))
        sources = [n for n in toward if obs.army_grid[n[0]][n[1]] > 1]
        if sources:
            source = max(sources, key=lambda n: (obs.army_grid[n[0]][n[1]], distance[n]))
            return self.move(source, toward[source])
        return PASS

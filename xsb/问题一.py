import math

cells = [(r, c) for r in range(4) for c in range(4)]

def inside(r, c):
    return 0 <= r < 4 and 0 <= c < 4

placements = []

for r in range(4):
    for c in range(4):
        placements.append({
            'type': 'mono',
            'cells': [(r, c)]
        })

for r in range(4):
    for c in range(4):
        if c + 1 < 4:
            placements.append({
                'type': 'domino',
                'cells': [(r, c), (r, c+1)]
            })
        if r + 1 < 4:
            placements.append({
                'type': 'domino',
                'cells': [(r, c), (r+1, c)]
            })


L_shapes = [
    [(0,0), (1,0), (1,1)],  # └
    [(0,0), (0,1), (1,0)],  # ┌
    [(0,0), (0,1), (1,1)],  # ┐
    [(0,1), (1,0), (1,1)],  # ┘
]

for r in range(3):
    for c in range(3):
        for shape in L_shapes:
            shape_cells = []
            ok = True
            for dr, dc in shape:
                rr, cc = r + dr, c + dc
                if not inside(rr, cc):
                    ok = False
                    break
                shape_cells.append((rr, cc))
            if ok:
                placements.append({
                    'type': 'L',
                    'cells': shape_cells
                })

print("Total placements:", len(placements))

# 每个格子 -> 能覆盖它的所有放置 index
cell_to_placements = {cell: [] for cell in cells}
for idx, pl in enumerate(placements):
    for cell in pl['cells']:
        cell_to_placements[cell].append(idx)

solutions = []

def search(covered, used_indices):
    if len(covered) == 16:
        if len(used_indices) == 6:
            solutions.append(list(used_indices))
        return
    if len(used_indices) >= 6:
        return

    remaining = 16 - len(covered)

    if len(used_indices) + math.ceil(remaining / 3) > 6:
        return

    for cell in cells:
        if cell not in covered:
            next_cell = cell
            break

    for pl_idx in cell_to_placements[next_cell]:
        pl = placements[pl_idx]
        pl_cells = pl['cells']

        if any(c in covered for c in pl_cells):
            continue
        new_covered = covered.union(pl_cells)
        used_indices.append(pl_idx)
        search(new_covered, used_indices)
        used_indices.pop()

search(set(), [])
print("Number of 6-tile tilings:", len(solutions))


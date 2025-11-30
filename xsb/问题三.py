import numpy as np
from ortools.linear_solver import pywraplp
import sys
import io
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

polyominoes = {
    1: [[(0, 0)]],  
    2: [[(0, 0), (1, 0)], [(0, 0), (0, 1)]],  
    3: [[(0, 0), (1, 0), (2, 0)], [(0, 0), (0, 1), (0, 2)]],  
    4: [[(0, 0), (0, 1), (1, 0)], [(0, 0), (0, 1), (1, 1)], [(0, 1), (1, 0), (1, 1)]], 
    5: [[(0, 0), (0, 1), (1, 0), (1, 1)]],  
    6: [[(0, 0), (0, 1), (0, 2), (1, 1)], [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (1, 1)], [(0, 1), (1, 0), (1, 1), (2, 1)]],  
    7: [[(0, 0), (0, 1), (1, 1), (1, 2)], [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)], [(0, 1), (1, 0), (1, 1), (2, 0)]], 
    8: [[(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 0), (0, 1), (0, 2), (0, 3)]],  
    9: [[(0, 0), (1, 0), (2, 0), (0, 1)], [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)], [(0, 2), (1, 0), (1, 1), (1, 2)]]  
}

polyomino_names = {
    1: "1×1", 2: "1×2", 3: "I3", 4: "L3", 5: "2×2", 6: "T4", 7: "Z4", 8: "I4", 9: "L4"
}

polyomino_colors = {
    1: "#F896FF",  
    2: "#FF9F43",  
    3: "#FFD1D8", 
    4: "#27AE60",  
    5: "#FDF99B",  
    6: "#FF8383",  
    7: "#A3FD7B",  
    8: "#A6BEFF",  
    9: "#97FCFC",  
}

polyomino_costs = {1: 1.0, 2: 1.5, 3: 2.0, 4: 2.0, 5: 2.5, 6: 2.5, 7: 2.5, 8: 2.5, 9: 2.5}

min_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
max_counts = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None}

SOLVER_CONFIG = {'solver_type': 'SAT', 'time_limit_seconds': 600, 'enable_output': True}

GRID_CONFIG = {'rows': 12, 'cols': 11}
COVERAGE_CONFIG = {'max_uncovered_cells':27}

weights = [0.1650,0.003,0.5376,0.2944]

def get_grid_dimensions():
    return GRID_CONFIG['rows'], GRID_CONFIG['cols']

def generate_placements_with_step(poly_id, shapes, step=1):
    rows, cols = get_grid_dimensions()
    placements = []
    for shape in shapes:
        max_r = max(cell[0] for cell in shape)
        max_c = max(cell[1] for cell in shape)
        for start_r in range(0, rows - max_r, step):
            for start_c in range(0, cols - max_c, step):
                covered_cells = [(start_r + dr, start_c + dc) for dr, dc in shape]
                if all(0 <= r < rows and 0 <= c < cols for r, c in covered_cells):
                    cell_set = set(covered_cells)
                    internal_edges = 0
                    for (r, c) in cell_set:
                        if (r, c + 1) in cell_set:
                            internal_edges += 1
                        if (r + 1, c) in cell_set:
                            internal_edges += 1
                    placements.append({
                        'type': poly_id,  
                        'start_pos': (start_r, start_c),  
                        'shape_index': shapes.index(shape),  
                        'covered_cells': covered_cells,  
                        'internal_edges': internal_edges
                    })
    return placements

def solve_basic_ilp(weights):
    rows, cols = get_grid_dimensions()
    all_placements = []
    placement_indices = {}
    for poly_id in range(1, 10):
        step = 2 if poly_id <= 2 else 1
        placements = generate_placements_with_step(poly_id, polyominoes[poly_id], step)
        all_placements.extend(placements)
        placement_indices[poly_id] = list(range(len(all_placements) - len(placements), len(all_placements)))
    area = rows * cols
    max_cost_per_piece = max(polyomino_costs.values())
    cost_ref = max_cost_per_piece * area
    count_ref = area
    stability_ref = 2 * area
    solver = pywraplp.Solver.CreateSolver(SOLVER_CONFIG['solver_type'])
    if SOLVER_CONFIG.get('enable_output', False):
        solver.EnableOutput()
    x = {p: solver.IntVar(0, 1, f'x[{p}]') for p in range(len(all_placements))}
    y = {t: solver.IntVar(min_counts[t],
                          solver.infinity() if max_counts[t] is None else max_counts[t],
                          f'y[{t}]') for t in range(1, 10)}
    cost_objective = solver.Sum(
        x[p] * polyomino_costs[all_placements[p]['type']]
        for p in range(len(all_placements))
    )
    count_objective = solver.Sum(y[t] for t in range(1, 10))
    stability_objective = solver.Sum(
        all_placements[p]['internal_edges'] * x[p]
        for p in range(len(all_placements))
    )
    cell_coverage = {}
    for r in range(rows):
        for c in range(cols):
            cell_coverage[(r, c)] = []
    for p, placement in enumerate(all_placements):
        for cell in placement['covered_cells']:
            cell_coverage[cell].append(p)
    cover = {}
    for r in range(rows):
        for c in range(cols):
            cover[(r, c)] = solver.IntVar(0, 1, f'cover[{r},{c}]')
    for cell, placements_list in cell_coverage.items():
        if placements_list:
            solver.Add(solver.Sum(x[p] for p in placements_list) == cover[cell])
        else:
            solver.Add(cover[cell] == 0)
    num_covered_cells = solver.Sum(cover[cell] for cell in cover)
    uncovered_cells = area - num_covered_cells
    max_uncovered = COVERAGE_CONFIG['max_uncovered_cells']
    solver.Add(uncovered_cells <= max_uncovered)
    normalized_cost = cost_objective*(1.0/cost_ref)
    normalized_stability= stability_objective * (1.0 / stability_ref)
    normalized_count = count_objective*(1.0/count_ref)
    normalized_uncovered = uncovered_cells*(1.0 / area)
    total_objective = (
            weights[0] * normalized_cost +
            weights[1] * normalized_count +
            weights[2] * normalized_stability +
            weights[3] * normalized_uncovered
    )
    solver.Minimize(total_objective)
    for t in range(1, 10):
        solver.Add(y[t] == solver.Sum([x[p] for p in placement_indices[t]]))
    support_side_vars = {}
    for p, placement in enumerate(all_placements):
        if placement['type'] != 5: 
            continue
        cells = placement['covered_cells']
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        for side in ['top', 'bottom', 'left', 'right']:
            neighbor_placements = set()
            if side == 'top':
                if min_r == 0:
                    continue
                for r, c in cells:
                    if r == min_r:
                        neighbor_cell = (r - 1, c)
                        neighbor_placements.update(cell_coverage[neighbor_cell])
            elif side == 'bottom':
                if max_r == rows - 1:
                    continue
                for r, c in cells:
                    if r == max_r:
                        neighbor_cell = (r + 1, c)
                        neighbor_placements.update(cell_coverage[neighbor_cell])
            elif side == 'left':
                if min_c == 0:
                    continue
                for r, c in cells:
                    if c == min_c:
                        neighbor_cell = (r, c - 1)
                        neighbor_placements.update(cell_coverage[neighbor_cell])
            elif side == 'right':
                if max_c == cols - 1:
                    continue
                for r, c in cells:
                    if c == max_c:
                        neighbor_cell = (r, c + 1)
                        neighbor_placements.update(cell_coverage[neighbor_cell])
            if not neighbor_placements:
                continue
            w = solver.IntVar(0, 1, f'w_support[{p},{side}]')
            support_side_vars[(p, side)] = (w, list(neighbor_placements))
            solver.Add(w <= x[p])
            solver.Add(w <= solver.Sum(x[q] for q in neighbor_placements))
    for p, placement in enumerate(all_placements):
        if placement['type'] != 5:
            continue
        side_vars = [
            support_side_vars[(p, side)][0]
            for side in ['top', 'bottom', 'left', 'right']
            if (p, side) in support_side_vars
        ]
        if side_vars:
            solver.Add(solver.Sum(side_vars) >= 2 * x[p])
    solver.set_time_limit(SOLVER_CONFIG['time_limit_seconds'] * 1000)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        solution = {}
        total_pieces = 0
        total_cost = 0
        for poly_id in range(1, 10):
            used_placements = []
            for p in placement_indices[poly_id]:
                if x[p].solution_value() > 0.5:  
                    placement = all_placements[p].copy()
                    used_placements.append(placement)
                    total_pieces += 1
                    total_cost += polyomino_costs[poly_id]
            solution[poly_id] = {
                'name': polyomino_names[poly_id],
                'count': len(used_placements),
                'placements': used_placements,
                'cost': sum(polyomino_costs[poly_id] for _ in used_placements)
            }
        shared_edges = calculate_shared_edges(solution, rows, cols)
        print(f"骨牌总数: {total_pieces}, 总成本: {total_cost}, 共享边数: {shared_edges}")
        return solution, total_pieces, total_cost, shared_edges, solver.Objective().Value()
    else:
        if status == pywraplp.Solver.INFEASIBLE:
            print("问题不可行")
        elif status == pywraplp.Solver.UNBOUNDED:
            print("问题无界")
        return None, None, None, None, None

def calculate_shared_edges(solution, rows, cols):
    grid = np.zeros((rows, cols), dtype=int)
    poly_counter = 1
    for t in range(1, 10):
        for i, placement in enumerate(solution[t]['placements']):
            for r, c in placement['covered_cells']:
                grid[r, c] = poly_counter
            poly_counter += 1
    shared_edges = 0
    for r in range(rows):
        for c in range(cols):
            current_poly = grid[r, c]
            if current_poly == 0: continue
            if c < cols - 1 and grid[r, c + 1] != 0 and grid[r, c + 1] != current_poly:
                shared_edges += 1
            if r < rows - 1 and grid[r + 1, c] != 0 and grid[r + 1, c] != current_poly:
                shared_edges += 1
    return shared_edges

def check_connectivity(solution, rows, cols):
    covered = np.zeros((rows, cols), dtype=bool)
    for t in range(1, 10):
        for placement in solution[t]['placements']:
            for r, c in placement['covered_cells']:
                covered[r, c] = True
    if not np.any(covered):
        return False
    visited = np.zeros((rows, cols), dtype=bool)
    start_r, start_c = -1, -1
    for r in range(rows):
        for c in range(cols):
            if covered[r, c]:
                start_r, start_c = r, c
                break
        if start_r != -1:
            break
    queue = deque([(start_r, start_c)])
    visited[start_r, start_c] = True
    connected_count = 0
    while queue:
        r, c = queue.popleft()
        connected_count += 1
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                    covered[nr, nc] and not visited[nr, nc]):
                visited[nr, nc] = True
                queue.append((nr, nc))
    total_covered = np.sum(covered)
    return connected_count == total_covered

def check_corner_coverage(solution):
    rows, cols = get_grid_dimensions()
    corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
    covered_corners = 0
    for corner in corners:
        r, c = corner
        for t in range(1, 10):
            for placement in solution[t]['placements']:
                if (r, c) in placement['covered_cells']:
                    covered_corners += 1
                    break
    return covered_corners == 4

def check_2x2_support(solution, rows, cols):
    grid = np.zeros((rows, cols), dtype=int)
    poly_counter = 1
    for t in range(1, 10):
        for i, placement in enumerate(solution[t]['placements']):
            for r, c in placement['covered_cells']:
                grid[r, c] = poly_counter
            poly_counter += 1
    for t in range(1, 10):
        if t != 5:
            continue

        for placement in solution[t]['placements']:
            cells = placement['covered_cells']
            if len(cells) != 4:
                continue
            min_r = min(r for r, c in cells)
            max_r = max(r for r, c in cells)
            min_c = min(c for r, c in cells)
            max_c = max(c for r, c in cells)
            supported_edges = 0
            if min_r > 0:
                top_neighbors = [(min_r - 1, c) for c in range(min_c, max_c + 1)]
                if any(grid[r, c] != 0 and grid[r, c] != grid[cells[0][0], cells[0][1]]
                       for r, c in top_neighbors):
                    supported_edges += 1
            if max_r < rows - 1:
                bottom_neighbors = [(max_r + 1, c) for c in range(min_c, max_c + 1)]
                if any(grid[r, c] != 0 and grid[r, c] != grid[cells[0][0], cells[0][1]]
                       for r, c in bottom_neighbors):
                    supported_edges += 1
            if min_c > 0:
                left_neighbors = [(r, min_c - 1) for r in range(min_r, max_r + 1)]
                if any(grid[r, c] != 0 and grid[r, c] != grid[cells[0][0], cells[0][1]]
                       for r, c in left_neighbors):
                    supported_edges += 1
            if max_c < cols - 1:
                right_neighbors = [(r, max_c + 1) for r in range(min_r, max_r + 1)]
                if any(grid[r, c] != 0 and grid[r, c] != grid[cells[0][0], cells[0][1]]
                       for r, c in right_neighbors):
                    supported_edges += 1
            if supported_edges < 2:
                return False
    return True

def validate_solution(solution):
    rows, cols = get_grid_dimensions()
    coverage_validation = np.zeros((rows, cols), dtype=bool)
    total_cells_covered = 0
    for t in range(1, 10):
        for placement in solution[t]['placements']:
            for cell in placement['covered_cells']:
                r, c = cell
                if coverage_validation[r, c]:
                    return False
                coverage_validation[r, c] = True
                total_cells_covered += 1
    constraint_violations = 0
    for t in range(1, 10):
        count = solution[t]['count']
        if count < min_counts[t]:
            constraint_violations += 1
        if max_counts[t] is not None and count > max_counts[t]:
            constraint_violations += 1
    if not check_corner_coverage(solution):
        constraint_violations += 1
    if not check_connectivity(solution, rows, cols):
        constraint_violations += 1
    if not check_2x2_support(solution, rows, cols):
        constraint_violations += 1
    max_uncovered = COVERAGE_CONFIG['max_uncovered_cells']
    total_cells = rows * cols
    uncovered = total_cells - total_cells_covered
    if uncovered <= max_uncovered and constraint_violations == 0:
        print("验证通过 ")
        return True
    else:
        if uncovered > max_uncovered:
            print("验证失败")
        if constraint_violations > 0:
            print("约束验证")
        return False

def main():
    solution, total_pieces, total_cost, shared_edges, objective_score = solve_basic_ilp(weights)
    if solution is not None:
        if validate_solution(solution):
            print("求解结果摘要:")
            for t in range(1, 10):
                info = solution[t]
                print(f"{info['name']}: {info['count']} 个, 成本: {info['cost']}")
            print(f"\n总骨牌数: {total_pieces}")
            print(f"总成本: {total_cost}")
            print(f"共享边数: {shared_edges}")
            print(f"模型评分: {objective_score:.6f}")
        else:
            print("解验证失败!")
    else:
        print("求解失败!")

if __name__ == "__main__":
    main()
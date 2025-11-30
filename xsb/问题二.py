import numpy as np
from ortools.linear_solver import pywraplp
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

polyominoes = {
    1: [[(0,0)]],
    2: [[(0,0), (1,0)], [(0,0), (0,1)]],
    3: [[(0,0), (1,0), (2,0)], [(0,0), (0,1), (0,2)]],
    4: [[(0,0), (0,1), (1,0)], [(0,0), (0,1), (1,1)], [(0,1), (1,0), (1,1)]],
    5: [[(0,0), (0,1), (1,0), (1,1)]],
    6: [[(0,0), (0,1), (0,2), (1,1)], [(0,1), (1,0), (1,1), (1,2)],
        [(0,0), (1,0), (2,0), (1,1)], [(0,1), (1,0), (1,1), (2,1)]],
    7: [[(0,0), (0,1), (1,1), (1,2)], [(0,1), (0,2), (1,0), (1,1)],
        [(0,0), (1,0), (1,1), (2,1)], [(0,1), (1,0), (1,1), (2,0)]],
    8: [[(0,0), (1,0), (2,0), (3,0)], [(0,0), (0,1), (0,2), (0,3)]],
    9: [[(0,0), (1,0), (2,0), (0,1)], [(0,0), (0,1), (0,2), (1,0)],
        [(0,0), (0,1), (1,1), (2,1)], [(0,2), (1,0), (1,1), (1,2)]]
}

polyomino_names = {
    1: "单格骨牌", 2: "双格骨牌", 3: "I型三格", 4: "L型三格", 
    5: "田字四格", 6: "T型四格", 7: "Z型四格", 8: "I型四格", 9: "L型四格"
}

polyomino_colors = {
    1: '#F896FF', 2: '#FF9F43', 3: '#FFD1D8', 4: '#27AE60', 5: '#FDF99B',
    6: '#FF8383', 7: '#A3FD7B', 8: '#A6BEFF', 9: '#97FCFC'
}

min_counts = {
    1: 0,  2: 0,  3: 0,  4: 0,  5: 0,  
    6: 0,  7: 0,  8: 0,  9: 0
}

max_counts = {
    1: 18, 2: 15,  3: 12,  4: 12,  5: 9,
    6: 9,  7: 9,  8: 9,  9: 9
}

SOLVER_CONFIG = {
    'solver_type': 'SAT',  
    'time_limit_seconds': 3000000,
    'enable_output': True
}

GRID_CONFIG = {
    'rows': 12,
    'cols': 11
}

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
                covered_cells = []
                for dr, dc in shape:
                    r, c = start_r + dr, start_c + dc
                    if 0 <= r < rows and 0 <= c < cols:
                        covered_cells.append((r, c))
                if len(covered_cells) == len(shape):
                    placements.append({
                        'type': poly_id,
                        'start_pos': (start_r, start_c),
                        'shape_index': shapes.index(shape),
                        'covered_cells': covered_cells
                    })
    return placements

def solve_basic_ilp():
    rows, cols = get_grid_dimensions()
    all_placements = []
    placement_indices = {}
    for poly_id in range(1, 10):
        step = 1
        if poly_id <= 2:
            step = 2
        elif poly_id <= 4:
            step = 1
        else:
            step = 1
        placements = generate_placements_with_step(poly_id, polyominoes[poly_id], step)
        all_placements.extend(placements)
        placement_indices[poly_id] = list(range(
            len(all_placements) - len(placements),
            len(all_placements)
        ))
    solver_type = SOLVER_CONFIG['solver_type']
    solver = pywraplp.Solver.CreateSolver(solver_type)
    x = {}
    for p in range(len(all_placements)):
        x[p] = solver.IntVar(0, 1, f'x[{p}]')
    y = {}
    for t in range(1, 10):
        y[t] = solver.IntVar(min_counts[t], max_counts[t], f'y[{t}]')
    solver.Minimize(solver.Sum([y[t] for t in range(1, 10)]))
    cell_coverage = {}
    for r in range(rows):
        for c in range(cols):
            cell_coverage[(r, c)] = []
    for p, placement in enumerate(all_placements):
        for cell in placement['covered_cells']:
            cell_coverage[cell].append(p)
    for cell, placements_list in cell_coverage.items():
        solver.Add(solver.Sum([x[p] for p in placements_list]) == 1)
    for t in range(1, 10):
        solver.Add(y[t] == solver.Sum([x[p] for p in placement_indices[t]]))
    solver.set_time_limit(SOLVER_CONFIG['time_limit_seconds'] * 1000)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        solution = {}
        total_pieces = 0
        for poly_id in range(1, 10):
            used_placements = []
            for p in placement_indices[poly_id]:
                if x[p].solution_value() > 0.5:
                    placement = all_placements[p].copy()
                    used_placements.append(placement)
                    total_pieces += 1
            solution[poly_id] = {
                'name': polyomino_names[poly_id],
                'count': len(used_placements),
                'placements': used_placements
            }
        return solution, total_pieces
    
    else:
        print("未能找到可行解")
        if status == pywraplp.Solver.INFEASIBLE:
            print("问题不可行")
        elif status == pywraplp.Solver.UNBOUNDED:
            print("问题无界")
        elif status == pywraplp.Solver.ABNORMAL:
            print("求解异常")
        elif status == pywraplp.Solver.NOT_SOLVED:
            print("问题未解决")
        return None, None

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
        if count < min_counts[t] or count > max_counts[t]:
            constraint_violations += 1
    
    if total_cells_covered == rows * cols and constraint_violations == 0:
        print("验证通过")
        return True
    else:
        print("验证失败")
        return False

def main():
    solution, total_pieces = solve_basic_ilp()
    if solution is not None:
        if validate_solution(solution):
            for t in range(1, 10):
                info = solution[t]
                print(f"{info['name']}: {info['count']} 个 ")

if __name__ == "__main__":
    main()
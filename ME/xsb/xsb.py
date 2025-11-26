import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ortools.linear_solver import pywraplp
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 骨牌形状定义
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

# 骨牌名称和颜色
polyomino_names = {
    1: "单格骨牌", 2: "双格骨牌", 3: "I型三格", 4: "L型三格", 
    5: "田字四格", 6: "T型四格", 7: "Z型四格", 8: "I型四格", 9: "L型四格"
}

polyomino_colors = {
    1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1', 4: '#96CEB4', 5: '#FFEAA7',
    6: '#DDA0DD', 7: '#FFA07A', 8: '#98D8C8', 9: '#F7DC6F'
}

# 骨牌数量约束配置
min_counts = {
    1: 0,  2: 0,  3: 0,  4: 0,  5: 0,  
    6: 0,  7: 0,  8: 0,  9: 0
}

max_counts = {
    1: 100, 2: 80,  3: 70,  4: 70,  5: 50,  
    6: 50,  7: 50,  8: 50,  9: 50
}

# 求解器配置
SOLVER_CONFIG = {
    'solver_type': 'SAT',  # 可选: 'SCIP', 'SAT', 'BOP'
    'time_limit_seconds': 3000000,
    'enable_output': True
}

# 网格尺寸配置
GRID_CONFIG = {
    'rows': 30,
    'cols': 30
}

def get_grid_dimensions():
    """获取网格尺寸"""
    return GRID_CONFIG['rows'], GRID_CONFIG['cols']

def generate_placements(poly_id, shapes):
    """生成某种骨牌的所有合法放置方式"""
    rows, cols = get_grid_dimensions()
    placements = []
    
    for shape in shapes:
        max_r = max(cell[0] for cell in shape)
        max_c = max(cell[1] for cell in shape)
        
        for start_r in range(rows - max_r):
            for start_c in range(cols - max_c):
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
    """使用基本ILP方法求解骨牌覆盖问题"""
    start_time = time.time()
    rows, cols = get_grid_dimensions()
    
    print("生成所有骨牌的放置方式...")
    
    all_placements = []
    placement_indices = {}
    
    for poly_id in range(1, 10):
        placements = generate_placements(poly_id, polyominoes[poly_id])
        all_placements.extend(placements)
        placement_indices[poly_id] = list(range(
            len(all_placements) - len(placements),
            len(all_placements)
        ))
        print(f"{polyomino_names[poly_id]}: {len(placements)} 种放置方式")
    
    print(f"总共 {len(all_placements)} 种放置方式")
    
    # 创建模型
    solver_type = SOLVER_CONFIG['solver_type']
    solver = pywraplp.Solver.CreateSolver(solver_type)
    
    # 创建变量
    x = {}
    for p in range(len(all_placements)):
        x[p] = solver.IntVar(0, 1, f'x[{p}]')
    
    # 创建骨牌数量变量
    y = {}
    for t in range(1, 10):
        y[t] = solver.IntVar(min_counts[t], max_counts[t], f'y[{t}]')
    
    # 目标函数: 最小化总骨牌数
    solver.Minimize(solver.Sum([y[t] for t in range(1, 10)]))
    
    # 覆盖约束: 每个单元格恰好被覆盖一次
    cell_coverage = {}
    for r in range(rows):
        for c in range(cols):
            cell_coverage[(r, c)] = []
    
    for p, placement in enumerate(all_placements):
        for cell in placement['covered_cells']:
            cell_coverage[cell].append(p)
    
    for cell, placements_list in cell_coverage.items():
        solver.Add(solver.Sum([x[p] for p in placements_list]) == 1)
    
    # 数量约束
    for t in range(1, 10):
        solver.Add(y[t] == solver.Sum([x[p] for p in placement_indices[t]]))
    
    # 设置求解时间限制
    solver.set_time_limit(SOLVER_CONFIG['time_limit_seconds'] * 1000)
    
    # 求解
    print(f"\n开始求解基本ILP问题 (使用 {solver.SolverVersion()} 求解器)...")
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        success_msg = "最优解" if status == pywraplp.Solver.OPTIMAL else "可行解"
        objective_value = solver.Objective().Value()
        print(f"\n找到{success_msg}!")
        print(f"骨牌总数: {objective_value}")
        
        # 提取解
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
        
        end_time = time.time()
        print(f"求解时间: {end_time - start_time:.2f} 秒")
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

def visualize_solution(solution, total_pieces):
    """可视化解决方案"""
    rows, cols = get_grid_dimensions()
    total_cells = rows * cols
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 绘制骨牌覆盖图
    ax1.set_xlim(-0.5, cols - 0.5)
    ax1.set_ylim(-0.5, rows - 0.5)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    ax1.set_title(f'骨牌覆盖方案\n(总数: {total_pieces})', fontsize=16, pad=20)
    ax1.set_xlabel('列')
    ax1.set_ylabel('行')
    
    # 为每个骨牌绘制矩形
    for t in range(1, 10):
        for placement in solution[t]['placements']:
            color = polyomino_colors[t]
            for r, c in placement['covered_cells']:
                rect = patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    linewidth=1, edgecolor='black',
                    facecolor=color, alpha=0.8
                )
                ax1.add_patch(rect)
    
    # 添加网格线
    ax1.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax1.grid(which="minor", color="black", linestyle='-', linewidth=0.3)
    ax1.tick_params(which="minor", size=0)
    
    # 统计信息
    ax2.axis('off')
    stats_text = "骨牌使用统计: \n\n"
    total_used = 0
    
    for t in range(1, 10):
        count = solution[t]['count']
        min_count = min_counts[t]
        max_count = max_counts[t]
        constraint_status = "✓" if min_count <= count <= max_count else "✗"
        stats_text += f"{polyomino_names[t]}: {count} 个 {constraint_status}\n"
        stats_text += f"    (约束: {min_count} ≤ {count} ≤ {max_count})\n"
        total_used += count
    
    stats_text += f"\n总计: {total_used} 个骨牌"
    stats_text += f"\n网格: {rows}x{cols} = {rows*cols} 个单元格"
    stats_text += f"\n理论最小: {int(np.ceil(total_cells / 4))} 个"
    stats_text += f"\n效率: {total_used / np.ceil(total_cells / 4):.2f} 倍理论最优"
    
    # 添加求解器信息
    stats_text += f"\n\n求解器: {SOLVER_CONFIG['solver_type']}"
    stats_text += f"\n时间限制: {SOLVER_CONFIG['time_limit_seconds']} 秒"
    
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=12,
              verticalalignment='top', linespacing=1.5)
    
    # 添加图例
    legend_elements = []
    for t in range(1, 10):
        legend_elements.append(
            patches.Patch(facecolor=polyomino_colors[t],
                         edgecolor='black',
                         label=f'{polyomino_names[t]}')
        )
    
    ax2.legend(handles=legend_elements, loc='lower left',
               bbox_to_anchor=(0, 0), fontsize=9,
               ncol=2, framealpha=0.7)
    
    plt.tight_layout()
    filename = 'polyomino_solution_basic.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def validate_solution(solution):
    """验证解的正确性"""
    rows, cols = get_grid_dimensions()
    coverage_validation = np.zeros((rows, cols), dtype=bool)
    total_cells_covered = 0
    
    # 检查覆盖完整性
    for t in range(1, 10):
        for placement in solution[t]['placements']:
            for cell in placement['covered_cells']:
                r, c = cell
                if coverage_validation[r, c]:
                    print(f"错误: 单元格 ({r}, {c}) 被重复覆盖!")
                    return False
                coverage_validation[r, c] = True
                total_cells_covered += 1
    
    # 检查数量约束
    constraint_violations = 0
    for t in range(1, 10):
        count = solution[t]['count']
        if count < min_counts[t] or count > max_counts[t]:
            print(f"数量约束违反: {polyomino_names[t]} 数量为 {count}，应在 [{min_counts[t]}, {max_counts[t]}] 范围内")
            constraint_violations += 1
    
    if total_cells_covered == rows * cols and constraint_violations == 0:
        print("✓ 验证通过: 所有单元格都被正确覆盖且满足数量约束")
        return True
    else:
        if total_cells_covered != rows * cols:
            print(f"✗ 覆盖验证失败: 只有 {total_cells_covered}/{rows*cols} 个单元格被覆盖")
        if constraint_violations > 0:
            print(f"✗ 数量约束验证失败: {constraint_violations} 个约束被违反")
        return False

def print_configuration():
    """打印当前配置信息"""
    rows, cols = get_grid_dimensions()
    print("=" * 60)
    print("骨牌覆盖优化问题求解系统 - 基本ILP方法")
    print("=" * 60)
    print(f"网格大小: {rows} x {cols} = {rows*cols} 单元格")
    print(f"求解器: {SOLVER_CONFIG['solver_type']}")
    print(f"时间限制: {SOLVER_CONFIG['time_limit_seconds']} 秒")
    
    print("\n骨牌数量约束:")
    for t in range(1, 10):
        print(f"  {polyomino_names[t]}: {min_counts[t]} ≤ 数量 ≤ {max_counts[t]}")
    
    print("\n骨牌形状信息:")
    for t in range(1, 10):
        print(f"  {polyomino_names[t]}: {len(polyominoes[t])} 种朝向, {len(polyominoes[t][0])} 格")
    
    print("=" * 60)

def main():
    """主函数"""
    # 打印配置信息
    print_configuration()
    
    # 求解问题
    solution, total_pieces = solve_basic_ilp()
    
    if solution is not None:
        # 验证解
        if validate_solution(solution):
            # 可视化结果
            visualize_solution(solution, total_pieces)
            
            # 打印详细结果
            rows, cols = get_grid_dimensions()
            print("\n" + "=" * 60)
            print("求解结果摘要:")
            print("=" * 60)
            for t in range(1, 10):
                info = solution[t]
                constraint_status = "✓" if min_counts[t] <= info['count'] <= max_counts[t] else "✗"
                print(f"{info['name']}: {info['count']} 个 {constraint_status}")
            
            print(f"\n总骨牌数: {total_pieces}")
            print(f"理论最小骨牌数: {int(np.ceil(rows * cols / 4))}")
            print(f"实际效率: {total_pieces / np.ceil(rows * cols / 4):.2f} 倍理论最优")
            
            # 保存配置摘要
            with open('solution_summary.txt', 'w', encoding='utf-8') as f:
                f.write("骨牌覆盖问题求解结果\n")
                f.write("=" * 40 + "\n")
                f.write(f"网格尺寸: {rows} x {cols}\n")
                f.write(f"总骨牌数: {total_pieces}\n\n")
                for t in range(1, 10):
                    info = solution[t]
                    f.write(f"{info['name']}: {info['count']} 个\n")
        else:
            print("解验证失败!")
    else:
        print("求解失败!")

if __name__ == "__main__":
    main()
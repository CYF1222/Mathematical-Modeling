import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from ortools.linear_solver import pywraplp
import itertools
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 定义骨牌形状（相对坐标）
polyominoes = {
    # 单格骨牌 (t=1)
    1: [[(0,0)]],
    
    # 双格骨牌 (t=2)
    2: [[(0,0), (0,1)]],  # I型
    
    # 三格骨牌
    3: [[(0,0), (0,1), (0,2)],  # I型三格
        [(0,0), (1,0), (2,0)]],
    
    4: [[(0,0), (0,1), (1,0)],  # L型三格
        [(0,0), (0,1), (1,1)],
        [(0,0), (1,0), (1,1)],
        [(0,1), (1,0), (1,1)]],
    
    # 四格骨牌
    5: [[(0,0), (0,1), (1,0), (1,1)]],  # 田字型
    
    6: [[(0,0), (0,1), (0,2), (1,1)],  # T型四格
        [(0,1), (1,0), (1,1), (1,2)],
        [(0,0), (1,0), (2,0), (1,1)],
        [(0,1), (1,0), (1,1), (2,1)]],
    
    7: [[(0,0), (0,1), (1,1), (1,2)],  # Z型四格
        [(0,1), (0,2), (1,0), (1,1)],
        [(0,0), (1,0), (1,1), (2,1)],
        [(0,1), (1,0), (1,1), (2,0)]],
    
    8: [[(0,0), (0,1), (0,2), (0,3)],  # I型四格
        [(0,0), (1,0), (2,0), (3,0)]],
    
    9: [[(0,0), (0,1), (0,2), (1,0)],  # L型四格
        [(0,0), (0,1), (0,2), (1,2)],
        [(0,0), (1,0), (2,0), (2,1)],
        [(0,1), (1,1), (2,0), (2,1)]]
}

# 骨牌名称
polyomino_names = {
    1: "单格骨牌",
    2: "双格骨牌", 
    3: "I型三格",
    4: "L型三格",
    5: "田字四格",
    6: "T型四格",
    7: "Z型四格",
    8: "I型四格",
    9: "L型四格"
}

# 骨牌数量上限
max_counts = {
    1: 100,  # 单格
    2: 80,  # 双格
    3: 70,  # I三格
    4: 70,  # L三格
    5: 50,   # 田四格
    6: 50,   # T四格
    7: 50,   # Z四格
    8: 50,   # I四格
    9: 50    # L四格
}

# 网格尺寸
ROWS, COLS = 30, 30
total_cells = ROWS * COLS

def generate_placements(poly_id, shapes):
    """生成某种骨牌的所有合法放置方式"""
    placements = []
    
    for shape in shapes:
        # 找到骨牌的边界
        max_r = max(cell[0] for cell in shape)
        max_c = max(cell[1] for cell in shape)
        
        # 遍历所有可能的起始位置
        for start_r in range(ROWS - max_r):
            for start_c in range(COLS - max_c):
                # 计算实际覆盖的单元格
                covered_cells = []
                for dr, dc in shape:
                    r, c = start_r + dr, start_c + dc
                    if 0 <= r < ROWS and 0 <= c < COLS:
                        covered_cells.append((r, c))
                
                # 如果所有单元格都在网格内，则这是一个合法放置
                if len(covered_cells) == len(shape):
                    placements.append({
                        'type': poly_id,
                        'start_pos': (start_r, start_c),
                        'shape_index': shapes.index(shape),
                        'covered_cells': covered_cells
                    })
    
    return placements

def solve_tiling_problem():
    """求解骨牌覆盖问题"""
    
    # 1. 生成所有可能的放置方式
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
    
    # 2. 创建模型 - 使用OR-Tools
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("SCIP solver not available, using SAT")
        solver = pywraplp.Solver.CreateSolver('SAT')
    
    # 3. 创建变量
    # x[p]: 是否选择第p种放置方式
    x = {}
    for p in range(len(all_placements)):
        x[p] = solver.IntVar(0, 1, f'x[{p}]')
    
    # y[t]: 第t种骨牌的使用数量
    y = {}
    for t in range(1, 10):
        y[t] = solver.IntVar(0, max_counts[t], f'y[{t}]')
    
    # 4. 目标函数：最小化总骨牌数
    solver.Minimize(solver.Sum([y[t] for t in range(1, 10)]))
    
    # 5. 约束条件
    
    # (a) 覆盖约束：每个单元格恰好被覆盖一次
    cell_coverage = {}
    for r in range(ROWS):
        for c in range(COLS):
            cell_coverage[(r, c)] = []
    
    for p, placement in enumerate(all_placements):
        for cell in placement['covered_cells']:
            cell_coverage[cell].append(p)
    
    for cell, placements_list in cell_coverage.items():
        solver.Add(solver.Sum([x[p] for p in placements_list]) == 1)
    
    # (b) 数量关系约束：y_t = sum(x_p for p in type t)
    for t in range(1, 10):
        solver.Add(y[t] == solver.Sum([x[p] for p in placement_indices[t]]))
    
    # 6. 设置求解时间限制（秒）
    solver.set_time_limit(30000000)  # 5分钟
    
    # 7. 求解
    print("\n开始求解...")
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        print(f"\n找到最优解！")
        print(f"最小骨牌总数: {solver.Objective().Value()}")
        
        # 提取解
        solution = extract_solution(solver, all_placements, placement_indices, x)
        return solution, solver.Objective().Value(), all_placements
    elif status == pywraplp.Solver.FEASIBLE:
        print(f"\n找到可行解（可能不是最优）")
        print(f"骨牌总数: {solver.Objective().Value()}")
        
        solution = extract_solution(solver, all_placements, placement_indices, x)
        return solution, solver.Objective().Value(), all_placements
    else:
        print("未找到可行解")
        return None, None, all_placements

def extract_solution(solver, all_placements, placement_indices, x):
    """从模型中提取解"""
    solution = {}
    
    for t in range(1, 10):
        used_placements = []
        for p in placement_indices[t]:
            if x[p].solution_value() > 0.5:
                placement = all_placements[p].copy()
                used_placements.append(placement)
        
        solution[t] = {
            'name': polyomino_names[t],
            'count': len(used_placements),
            'placements': used_placements
        }
    
    return solution

def visualize_solution(solution, total_pieces, all_placements):
    """可视化解决方案"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 创建颜色映射
    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    cmap = ListedColormap(colors[1:])
    
    # 绘制网格覆盖图
    grid = np.zeros((ROWS, COLS), dtype=int)
    placement_info = []
    
    # 重建placement_info
    for t, info in solution.items():
        for placement in info['placements']:
            for r, c in placement['covered_cells']:
                grid[r, c] = t
            # 记录放置信息
            placement_info.append({
                'type': t,
                'name': polyomino_names[t],
                'start_pos': placement['start_pos'],
                'shape_index': placement['shape_index'],
                'covered_cells': placement['covered_cells']
            })
    
    # 主图
    im = ax1.imshow(grid, cmap=cmap, vmin=1, vmax=9)
    ax1.set_title(f'骨牌覆盖方案 (总骨牌数: {total_pieces})', fontsize=14)
    ax1.set_xlabel('列')
    ax1.set_ylabel('行')
    
    # 添加网格线
    ax1.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax1.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax1.tick_params(which="minor", size=0)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_ticks(range(1, 10))
    cbar.set_ticklabels([polyomino_names[i] for i in range(1, 10)])
    
    # 统计信息
    ax2.axis('off')
    stats_text = "骨牌使用统计:\n\n"
    for t in range(1, 10):
        stats_text += f"{polyomino_names[t]}: {solution[t]['count']} 个\n"
    
    stats_text += f"\n总计: {total_pieces} 个骨牌"
    stats_text += f"\n覆盖: {ROWS}×{COLS} = {ROWS*COLS} 个单元格"
    
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=12, 
             verticalalignment='top', linespacing=1.5)
    
    plt.tight_layout()
    plt.savefig('polyomino_tiling_solution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return grid, placement_info

def save_to_excel(solution, placement_info, filename='polyomino_solution.xlsx'):
    """保存结果到Excel文件"""
    try:
        import pandas as pd
        
        # 创建数据框
        data = []
        for info in placement_info:
            data.append({
                '骨牌类型': info['name'],
                '起始行': info['start_pos'][0] + 1,  # 1-based indexing
                '起始列': info['start_pos'][1] + 1,
                '形状编号': info['shape_index'] + 1
            })
        
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False)
        print(f"\n结果已保存到 {filename}")
        
        # 也保存统计信息
        stats_data = []
        for t in range(1, 10):
            stats_data.append({
                '骨牌类型': polyomino_names[t],
                '使用数量': solution[t]['count'],
                '数量上限': max_counts[t]
            })
        
        stats_df = pd.DataFrame(stats_data)
        with pd.ExcelWriter(filename, mode='a', if_sheet_exists='replace') as writer:
            stats_df.to_excel(writer, sheet_name='统计信息', index=False)
            
    except ImportError:
        print("请安装 pandas 库来导出Excel文件: pip install pandas openpyxl")
        
        # 保存为CSV格式
        with open('polyomino_solution.csv', 'w', encoding='utf-8') as f:
            f.write('骨牌类型,起始行,起始列,形状编号\n')
            for info in placement_info:
                f.write(f"{info['name']},{info['start_pos'][0]+1},{info['start_pos'][1]+1},{info['shape_index']+1}\n")
        print("结果已保存到 polyomino_solution.csv")

def main():
    """主函数"""
    print("=" * 60)
    print("骨牌覆盖优化问题求解")
    print(f"网格大小: {ROWS} × {COLS} = {ROWS*COLS} 单元格")
    print("=" * 60)
    
    # 求解问题
    solution, total_pieces, all_placements = solve_tiling_problem()
    
    if solution is not None:
        # 可视化结果
        grid, placement_info = visualize_solution(solution, total_pieces, all_placements)
        
        # 保存结果
        save_to_excel(solution, placement_info)
        
        # 打印详细结果
        print("\n" + "=" * 60)
        print("详细结果:")
        print("=" * 60)
        for t in range(1, 10):
            info = solution[t]
            print(f"{info['name']}: {info['count']} 个")
        
        print(f"\n总骨牌数: {total_pieces}")
        print(f"理论最小骨牌数下限: {np.ceil(total_cells / 4)}")
        
        # 验证覆盖
        coverage_validation = np.zeros((ROWS, COLS), dtype=bool)
        for info in placement_info:
            for cell in info['covered_cells']:
                coverage_validation[cell] = True
        
        if np.all(coverage_validation):
            print("✓ 覆盖验证: 所有单元格都被正确覆盖")
        else:
            uncovered = np.where(~coverage_validation)
            print(f"✗ 覆盖验证: 存在 {len(uncovered[0])} 个未覆盖的单元格")

if __name__ == "__main__":
    main()
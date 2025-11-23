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

# 修正骨牌形状定义 - 确保所有形状都从(0,0)开始并正确连接
polyominoes = {
    # 单格骨牌 (t=1)
    1: [[(0,0)]],
    
    # 双格骨牌 (t=2)
    2: [[(0,0), (1,0)],  # 竖放
        [(0,0), (0,1)]], # 横放
    
    # 三格骨牌
    3: [[(0,0), (1,0), (2,0)],  # I型竖放
        [(0,0), (0,1), (0,2)]], # I型横放
    
    4: [[(0,0), (1,0), (1,1)],  # L型
        [(0,0), (0,1), (1,0)],
        [(0,0), (0,1), (1,1)],
        [(0,1), (1,0), (1,1)]],
    
    # 四格骨牌
    5: [[(0,0), (0,1), (1,0), (1,1)]],  # 田字型
    
    6: [[(0,0), (0,1), (0,2), (1,1)],  # T型
        [(0,1), (1,0), (1,1), (1,2)],
        [(0,0), (1,0), (2,0), (1,1)],
        [(0,1), (1,0), (1,1), (2,1)]],
    
    7: [[(0,0), (0,1), (1,1), (1,2)],  # Z型
        [(0,1), (0,2), (1,0), (1,1)],
        [(0,0), (1,0), (1,1), (2,1)],
        [(0,1), (1,0), (1,1), (2,0)]],
    
    8: [[(0,0), (1,0), (2,0), (3,0)],  # I型竖放
        [(0,0), (0,1), (0,2), (0,3)]], # I型横放
    
    9: [[(0,0), (1,0), (2,0), (2,1)],  # L型
        [(0,0), (0,1), (0,2), (1,0)],
        [(0,0), (0,1), (1,1), (2,1)],
        [(0,2), (1,0), (1,1), (1,2)]]
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

# 骨牌颜色
polyomino_colors = {
    1: '#FF6B6B',  # 红色
    2: '#4ECDC4',  # 青色
    3: '#45B7D1',  # 蓝色
    4: '#96CEB4',  # 绿色
    5: '#FFEAA7',  # 黄色
    6: '#DDA0DD',  # 紫色
    7: '#FFA07A',  # 橙色
    8: '#98D8C8',  # 薄荷色
    9: '#F7DC6F'   # 金色
}

# 骨牌数量上限
max_counts = {
    1: 50,  # 单格
    2: 50,  # 双格
    3: 40,  # I三格
    4: 40,  # L三格
    5: 20,  # 田四格
    6: 20,  # T四格
    7: 20,  # Z四格
    8: 20,  # I四格
    9: 20   # L四格
}

# 骨牌数量下限
min_counts = {
    1: 5,   # 单格骨牌最少5个
    2: 0,   # 双格骨牌最少0个
    3: 0,   # I型三格最少0个
    4: 0,   # L型三格最少0个
    5: 0,   # 田字四格最少0个
    6: 0,   # T型四格最少0个
    7: 0,   # Z型四格最少0个
    8: 0,   # I型四格最少0个
    9: 0    # L型四格最少0个
}

# 网格尺寸
ROWS, COLS = 25, 20
total_cells = ROWS * COLS

# 颜色数量 (C ≥ 2)
C_COLORS = 3

def apply_checkerboard_coloring():
    """应用棋盘着色到网格 (论文中的公式1)"""
    coloring = np.zeros((ROWS, COLS), dtype=int)
    
    # 使用论文中的公式: s_k(j', i') = (i' + j' + k - 1) mod C + 1
    # 我们使用 k = 0 (简化)
    k = 0
    
    for i in range(ROWS):
        for j in range(COLS):
            # 注意: 论文中参数顺序是 (j', i') 对应 (列, 行)
            color_index = (i + j + k) % C_COLORS + 1
            coloring[i, j] = color_index
    
    return coloring

def generate_colored_placements(poly_id, shapes, coloring):
    """生成某种骨牌的所有合法放置方式，考虑颜色匹配"""
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
                color_pattern = []
                
                for dr, dc in shape:
                    r, c = start_r + dr, start_c + dc
                    if 0 <= r < ROWS and 0 <= c < COLS:
                        covered_cells.append((r, c))
                        color_pattern.append(coloring[r, c])
                
                # 如果所有单元格都在网格内，则这是一个合法放置
                if len(covered_cells) == len(shape):
                    placements.append({
                        'type': poly_id,
                        'start_pos': (start_r, start_c),
                        'shape_index': shapes.index(shape),
                        'covered_cells': covered_cells,
                        'color_pattern': color_pattern
                    })
    
    return placements

def solve_colored_ilp():
    """使用单阶段ILP方法求解骨牌覆盖问题，结合颜色思想"""
    print("应用棋盘着色...")
    coloring = apply_checkerboard_coloring()
    
    print("生成所有骨牌的放置方式（考虑颜色）...")
    all_placements = []
    placement_indices = {}
    
    for poly_id in range(1, 10):
        placements = generate_colored_placements(poly_id, polyominoes[poly_id], coloring)
        all_placements.extend(placements)
        placement_indices[poly_id] = list(range(
            len(all_placements) - len(placements), 
            len(all_placements)
        ))
        print(f"{polyomino_names[poly_id]}: {len(placements)} 种放置方式")
    
    print(f"总共 {len(all_placements)} 种放置方式")
    
    # 创建模型 - 使用OR-Tools
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("SCIP solver not available, using SAT")
        solver = pywraplp.Solver.CreateSolver('SAT')
    
    # 创建变量
    # x[p]: 是否选择第p种放置方式
    x = {}
    for p in range(len(all_placements)):
        x[p] = solver.IntVar(0, 1, f'x[{p}]')
    
    # y[t]: 第t种骨牌的使用数量
    y = {}
    for t in range(1, 10):
        y[t] = solver.IntVar(min_counts[t], max_counts[t], f'y[{t}]')
    
    # 目标函数：最小化总骨牌数
    solver.Minimize(solver.Sum([y[t] for t in range(1, 10)]))
    
    # 约束条件
    
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
    
    # 设置求解时间限制（秒）
    solver.set_time_limit(300000)  # 5分钟
    
    # 求解
    print("\n开始求解...")
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        print(f"\n找到最优解！")
        print(f"最小骨牌总数: {solver.Objective().Value()}")
        
        # 提取解
        solution = extract_solution(solver, all_placements, placement_indices, x)
        total_pieces = int(solver.Objective().Value())
        return solution, total_pieces, all_placements, coloring
    elif status == pywraplp.Solver.FEASIBLE:
        print(f"\n找到可行解（可能不是最优）")
        print(f"骨牌总数: {solver.Objective().Value()}")
        
        solution = extract_solution(solver, all_placements, placement_indices, x)
        total_pieces = int(solver.Objective().Value())
        return solution, total_pieces, all_placements, coloring
    else:
        print("未找到可行解")
        return None, None, None, None

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

def visualize_solution(solution, total_pieces, coloring):
    """可视化解决方案 - 修复版本"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # 创建颜色映射
    colors_color = ['#FFFFFF', '#FF9999', '#99CCFF', '#99FF99']  # 白色, 红色, 蓝色, 绿色
    cmap_color = ListedColormap(colors_color[:C_COLORS + 1])
    
    # 绘制网格着色
    im1 = ax1.imshow(coloring, cmap=cmap_color, vmin=0, vmax=C_COLORS)
    ax1.set_title(f'棋盘着色 (C={C_COLORS})', fontsize=16, pad=20)
    ax1.set_xlabel('列')
    ax1.set_ylabel('行')
    
    # 添加网格线
    ax1.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax1.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.7)
    ax1.tick_params(which="minor", size=0)
    
    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_ticks(range(0, C_COLORS + 1))
    cbar1.set_ticklabels([''] + [f'颜色{i}' for i in range(1, C_COLORS + 1)])
    
    # 绘制骨牌覆盖图 - 使用patches正确绘制骨牌边界
    ax2.set_xlim(-0.5, COLS - 0.5)
    ax2.set_ylim(-0.5, ROWS - 0.5)
    ax2.set_aspect('equal')
    ax2.invert_yaxis()  # 让坐标原点在左上角
    ax2.set_title(f'骨牌覆盖方案 (总骨牌数: {total_pieces})', fontsize=16, pad=20)
    ax2.set_xlabel('列')
    ax2.set_ylabel('行')
    
    # 为每个骨牌绘制矩形
    for t, info in solution.items():
        for placement in info['placements']:
            color = polyomino_colors[t]
            
            # 绘制骨牌的每个单元格
            for r, c in placement['covered_cells']:
                rect = patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, 
                    linewidth=2, edgecolor='black', 
                    facecolor=color, alpha=0.8
                )
                ax2.add_patch(rect)
    
    # 添加网格线
    ax2.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax2.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax2.tick_params(which="minor", size=0)
    
    # 统计信息
    ax3.axis('off')
    stats_text = "骨牌使用统计:\n\n"
    total_used = 0
    for t in range(1, 10):
        count = solution[t]['count']
        stats_text += f"{polyomino_names[t]}: {count} 个\n"
        total_used += count
    
    stats_text += f"\n总计: {total_used} 个骨牌"
    stats_text += f"\n覆盖: {ROWS}×{COLS} = {ROWS*COLS} 个单元格"
    stats_text += f"\n使用颜色数: {C_COLORS}"
    stats_text += f"\n理论最小: {int(np.ceil(total_cells / 4))} 个"
    
    ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=14, 
             verticalalignment='top', linespacing=1.8)
    
    # 添加图例
    legend_elements = []
    for t in range(1, 10):
        legend_elements.append(
            patches.Patch(facecolor=polyomino_colors[t], 
                         edgecolor='black', 
                         label=f'{polyomino_names[t]}')
        )
    
    ax3.legend(handles=legend_elements, loc='lower left', 
               bbox_to_anchor=(0, 0), fontsize=10, 
               ncol=2, framealpha=0.7)
    
    plt.tight_layout()
    plt.savefig('colored_polyomino_solution_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return None, None

def save_to_excel(solution, placement_info, filename='colored_polyomino_solution.xlsx'):
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
        total_pieces = 0
        for t in range(1, 10):
            count = solution[t]['count']
            stats_data.append({
                '骨牌类型': polyomino_names[t],
                '使用数量': count,
                '数量下限': min_counts[t],
                '数量上限': max_counts[t]
            })
            total_pieces += count
        
        stats_data.append({
            '骨牌类型': '总计',
            '使用数量': total_pieces,
            '数量下限': '',
            '数量上限': ''
        })
        
        stats_df = pd.DataFrame(stats_data)
        with pd.ExcelWriter(filename, mode='a', if_sheet_exists='replace') as writer:
            stats_df.to_excel(writer, sheet_name='统计信息', index=False)
            
    except ImportError:
        print("请安装 pandas 库来导出Excel文件: pip install pandas openpyxl")
        
        # 保存为CSV格式
        with open('colored_polyomino_solution.csv', 'w', encoding='utf-8') as f:
            f.write('骨牌类型,起始行,起始列,形状编号\n')
            for info in placement_info:
                f.write(f"{info['name']},{info['start_pos'][0]+1},{info['start_pos'][1]+1},{info['shape_index']+1}\n")
        print("结果已保存到 colored_polyomino_solution.csv")

def main():
    """主函数"""
    print("=" * 60)
    print("结合棋盘着色思想的单阶段ILP方法 - 骨牌覆盖优化问题求解")
    print(f"网格大小: {ROWS} × {COLS} = {ROWS*COLS} 单元格")
    print(f"使用颜色数: {C_COLORS}")
    print("=" * 60)
    
    # 打印数量约束
    print("\n数量约束:")
    for poly_id in range(1, 10):
        print(f"{polyomino_names[poly_id]}: {min_counts[poly_id]} ≤ 数量 ≤ {max_counts[poly_id]}")
    
    # 求解问题
    solution, total_pieces, all_placements, coloring = solve_colored_ilp()
    
    if solution is not None:
        # 可视化结果
        placement_info = []
        for t, info in solution.items():
            for placement in info['placements']:
                placement_info.append({
                    'type': placement['type'],
                    'name': polyomino_names[placement['type']],
                    'start_pos': placement['start_pos'],
                    'shape_index': placement['shape_index'],
                    'covered_cells': placement['covered_cells']
                })
        
        grid, _ = visualize_solution(solution, total_pieces, coloring)
        
        # 保存结果
        save_to_excel(solution, placement_info)
        
        # 打印详细结果
        print("\n" + "=" * 60)
        print("详细结果:")
        print("=" * 60)
        for t in range(1, 10):
            info = solution[t]
            print(f"{info['name']}: {info['count']} 个 (约束: {min_counts[t]} ≤ {info['count']} ≤ {max_counts[t]})")
        
        print(f"\n总骨牌数: {total_pieces}")
        print(f"理论最小骨牌数下限: {np.ceil(total_cells / 4)}")
        
        # 验证覆盖
        coverage_validation = np.zeros((ROWS, COLS), dtype=bool)
        for t, info in solution.items():
            for placement in info['placements']:
                for cell in placement['covered_cells']:
                    coverage_validation[cell] = True
        
        if np.all(coverage_validation):
            print("✓ 覆盖验证: 所有单元格都被正确覆盖")
        else:
            uncovered = np.where(~coverage_validation)
            print(f"✗ 覆盖验证: 存在 {len(uncovered[0])} 个未覆盖的单元格")
            
        # 显示理论分析
        print(f"\n理论分析:")
        print(f"- 网格总面积: {ROWS}×{COLS} = {ROWS*COLS} 单元格")
        print(f"- 最大骨牌面积: 4 单元格")
        print(f"- 理论最小骨牌数: ⌈{ROWS*COLS}/4⌉ = {int(np.ceil(total_cells / 4))} 个")
        print(f"- 实际最小骨牌数: {total_pieces} 个")
        print(f"- 效率: {total_pieces / np.ceil(total_cells / 4):.2f} 倍理论最优")
        print(f"- 使用颜色方法: {C_COLORS} 色棋盘着色")
        
        # 验证数量约束
        print(f"\n数量约束验证:")
        for t in range(1, 10):
            count = solution[t]['count']
            if min_counts[t] <= count <= max_counts[t]:
                print(f"✓ {polyomino_names[t]}: {count} 个 (满足约束)")
            else:
                print(f"✗ {polyomino_names[t]}: {count} 个 (不满足约束)")
        
    else:
        print("未能找到可行解，请尝试调整参数或使用不同的求解器")

if __name__ == "__main__":
    main()
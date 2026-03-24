import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 中文字体设置：通过 font_manager 注册 Noto Sans CJK
from matplotlib.font_manager import FontProperties, fontManager
_cn_font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fontManager.addfont(_cn_font_path)
_cn_font_name = FontProperties(fname=_cn_font_path).get_name()
matplotlib.rcParams['font.family'] = [_cn_font_name, 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12

# ============================================================
# 数据定义
# ============================================================
datasets = ['Image-R', 'ArxivQA', 'Viz-cap', 'IconQA', 'CLEVR', 'Flickr30k']

# HiDe-LLaVA 结果矩阵 (task_i 行, dataset_j 列), NaN 表示未学
hide = np.array([
    [91.87, np.nan, np.nan, np.nan, np.nan, np.nan],
    [91.30, 93.27, np.nan, np.nan, np.nan, np.nan],
    [89.10, 92.33, 54.88, np.nan, np.nan, np.nan],
    [87.23, 90.83, 49.07, 81.67, np.nan, np.nan],
    [84.73, 91.57, 46.19, 66.67, 67.03, np.nan],
    [83.50, 90.60, 48.51, 66.60, 61.63, 55.32],
])

# HiDe-LLaVA + Task Prompt 结果矩阵
hybrid = np.array([
    [91.17, np.nan, np.nan, np.nan, np.nan, np.nan],
    [90.73, 93.23, np.nan, np.nan, np.nan, np.nan],
    [89.03, 93.07, 59.77, np.nan, np.nan, np.nan],
    [87.13, 90.77, 50.92, 84.67, np.nan, np.nan],
    [86.13, 91.87, 49.21, 73.00, 63.20, np.nan],
    [83.97, 92.43, 47.12, 71.63, 54.13, 56.40],
])

task_labels = [f'任务 {i+1}' for i in range(6)]


# ============================================================
# 图1: 准确率矩阵热力图
# ============================================================
def plot_heatmap(data, title, filename):
    fig, ax = plt.subplots(figsize=(9, 6))

    masked = np.ma.masked_where(np.isnan(data), data)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color='white')

    im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=40, vmax=95)

    ax.set_xticks(range(6))
    ax.set_xticklabels(datasets, rotation=30, ha='right')
    ax.set_yticks(range(6))
    ax.set_yticklabels(task_labels)

    for i in range(6):
        for j in range(6):
            if not np.isnan(data[i, j]):
                color = 'white' if data[i, j] > 80 else 'black'
                ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('准确率 (%)', fontsize=12)

    ax.set_title(title, fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('数据集', fontsize=13)
    ax.set_ylabel('训练阶段', fontsize=13)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {filename}')


plot_heatmap(hide, 'HiDe-LLaVA 准确率矩阵', 'charm/heatmap_hide.png')
plot_heatmap(hybrid, 'HiDe-LLaVA + Task Prompt 准确率矩阵', 'charm/heatmap_hybrid.png')

# 差值热力图
def plot_diff_heatmap():
    diff = hybrid - hide
    fig, ax = plt.subplots(figsize=(9, 6))

    masked = np.ma.masked_where(np.isnan(diff), diff)
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color='white')

    vmax = np.nanmax(np.abs(diff))
    im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(6))
    ax.set_xticklabels(datasets, rotation=30, ha='right')
    ax.set_yticks(range(6))
    ax.set_yticklabels(task_labels)

    for i in range(6):
        for j in range(6):
            if not np.isnan(diff[i, j]):
                val = diff[i, j]
                color = 'white' if abs(val) > 4 else 'black'
                sign = '+' if val > 0 else ''
                ax.text(j, i, f'{sign}{val:.1f}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('准确率差值 (%)', fontsize=12)

    ax.set_title('准确率差值（Task Prompt − HiDe-LLaVA）', fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('数据集', fontsize=13)
    ax.set_ylabel('训练阶段', fontsize=13)

    plt.tight_layout()
    plt.savefig('charm/heatmap_diff.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved: charm/heatmap_diff.png')

plot_diff_heatmap()


# ============================================================
# 图2: 遗忘曲线折线图
# ============================================================
def plot_forgetting_curves():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    for idx, (data, title) in enumerate([(hide, 'HiDe-LLaVA'), (hybrid, 'HiDe-LLaVA + Task Prompt')]):
        ax = axes[idx]
        for j in range(6):
            col = data[:, j]
            valid_idx = ~np.isnan(col)
            x = np.where(valid_idx)[0]
            y = col[valid_idx]
            ax.plot(x, y, color=colors[j], marker=markers[j], linewidth=2,
                    markersize=8, label=datasets[j], zorder=3)

        ax.set_xticks(range(6))
        ax.set_xticklabels(task_labels, fontsize=11)
        ax.set_xlabel('训练阶段', fontsize=13)
        ax.set_ylabel('准确率 (%)', fontsize=13)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(40, 98)

    plt.tight_layout()
    plt.savefig('charm/forgetting_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved: charm/forgetting_curves.png')

plot_forgetting_curves()

# 单图叠加对比（同一数据集两条线）
def plot_forgetting_overlay():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors_hide = '#e74c3c'
    colors_hybrid = '#3498db'

    for j in range(6):
        ax = axes[j]

        # HiDe
        col_h = hide[:, j]
        valid_h = ~np.isnan(col_h)
        x_h = np.where(valid_h)[0]
        y_h = col_h[valid_h]
        ax.plot(x_h, y_h, color=colors_hide, marker='o', linewidth=2.5,
                markersize=8, label='HiDe-LLaVA', zorder=3)

        # Hybrid
        col_hy = hybrid[:, j]
        valid_hy = ~np.isnan(col_hy)
        x_hy = np.where(valid_hy)[0]
        y_hy = col_hy[valid_hy]
        ax.plot(x_hy, y_hy, color=colors_hybrid, marker='s', linewidth=2.5,
                markersize=8, label='HiDe-LLaVA + Task Prompt', zorder=3)

        # 标注最终差值
        if len(y_h) > 0 and len(y_hy) > 0:
            diff = y_hy[-1] - y_h[-1]
            sign = '+' if diff > 0 else ''
            color = '#27ae60' if diff > 0 else '#c0392b'
            ax.annotate(f'{sign}{diff:.2f}%', xy=(x_h[-1], (y_h[-1]+y_hy[-1])/2),
                       fontsize=12, fontweight='bold', color=color,
                       xytext=(10, 0), textcoords='offset points')

        ax.set_title(datasets[j], fontsize=14, fontweight='bold')
        ax.set_xticks(range(6))
        ax.set_xticklabels([f'T{i+1}' for i in range(6)], fontsize=10)
        ax.set_ylabel('准确率 (%)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('遗忘曲线对比：HiDe-LLaVA vs HiDe-LLaVA + Task Prompt',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('charm/forgetting_overlay.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved: charm/forgetting_overlay.png')

plot_forgetting_overlay()


# ============================================================
# 图3: Task6 双柱状图对比
# ============================================================
def plot_bar_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(datasets))
    width = 0.35

    hide_task6 = hide[5, :]
    hybrid_task6 = hybrid[5, :]

    bars1 = ax.bar(x - width/2, hide_task6, width, label='HiDe-LLaVA',
                   color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, hybrid_task6, width, label='HiDe-LLaVA + Task Prompt',
                   color='#3498db', alpha=0.85, edgecolor='white', linewidth=1)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    for i in range(6):
        diff = hybrid_task6[i] - hide_task6[i]
        sign = '+' if diff > 0 else ''
        color = '#27ae60' if diff > 0 else '#c0392b'
        y_pos = max(hide_task6[i], hybrid_task6[i]) + 3
        ax.text(x[i], y_pos, f'{sign}{diff:.2f}%', ha='center', fontsize=10,
                fontweight='bold', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel('准确率 (%)', fontsize=13)
    ax.set_title('任务6 各数据集性能对比', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('charm/bar_task6.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved: charm/bar_task6.png')

plot_bar_comparison()


# ============================================================
# 图4: 雷达图
# ============================================================
def plot_radar():
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    hide_task6 = hide[5, :]
    hybrid_task6 = hybrid[5, :]

    N = len(datasets)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    hide_vals = hide_task6.tolist() + [hide_task6[0]]
    hybrid_vals = hybrid_task6.tolist() + [hybrid_task6[0]]

    ax.plot(angles, hide_vals, 'o-', linewidth=2.5, markersize=8,
            color='#e74c3c', label='HiDe-LLaVA')
    ax.fill(angles, hide_vals, alpha=0.15, color='#e74c3c')

    ax.plot(angles, hybrid_vals, 's-', linewidth=2.5, markersize=8,
            color='#3498db', label='HiDe-LLaVA + Task Prompt')
    ax.fill(angles, hybrid_vals, alpha=0.15, color='#3498db')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(datasets, fontsize=12, fontweight='bold')
    ax.set_ylim(40, 100)
    ax.set_yticks([50, 60, 70, 80, 90])
    ax.set_yticklabels(['50', '60', '70', '80', '90'], fontsize=9, color='gray')

    ax.set_title('任务6 雷达图对比', fontsize=15, fontweight='bold', pad=25)
    ax.legend(loc='lower right', bbox_to_anchor=(1.15, -0.05), fontsize=12)

    plt.tight_layout()
    plt.savefig('charm/radar.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved: charm/radar.png')

plot_radar()

print('\n所有图表已保存至 charm/')

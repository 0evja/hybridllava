import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 12

datasets = ['Image-R', 'ArxivQA', 'Viz-cap', 'IconQA', 'CLEVR', 'Flickr30k']
task_labels = [f'Task {i+1}' for i in range(6)]

hide = np.array([
    [91.87, np.nan, np.nan, np.nan, np.nan, np.nan],
    [91.30, 93.27, np.nan, np.nan, np.nan, np.nan],
    [89.10, 92.33, 54.88, np.nan, np.nan, np.nan],
    [87.23, 90.83, 49.07, 81.67, np.nan, np.nan],
    [84.73, 91.57, 46.19, 66.67, 67.03, np.nan],
    [83.50, 90.60, 48.51, 66.60, 61.63, 55.32],
])

hybrid = np.array([
    [91.17, np.nan, np.nan, np.nan, np.nan, np.nan],
    [90.73, 93.23, np.nan, np.nan, np.nan, np.nan],
    [89.03, 93.07, 59.77, np.nan, np.nan, np.nan],
    [87.13, 90.77, 50.92, 84.67, np.nan, np.nan],
    [86.13, 91.87, 49.21, 73.00, 63.20, np.nan],
    [83.97, 92.43, 47.12, 71.63, 54.13, 56.40],
])

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
markers = ['o', 's', '^', 'D', 'v', 'p']


def plot_single_forgetting(data, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))

    for j in range(6):
        col = data[:, j]
        valid_idx = ~np.isnan(col)
        x = np.where(valid_idx)[0]
        y = col[valid_idx]
        ax.plot(x, y, color=colors[j], marker=markers[j], linewidth=2.5,
                markersize=9, label=datasets[j], zorder=3)

        # 标注首尾数值
        ax.annotate(f'{y[0]:.1f}', xy=(x[0], y[0]),
                    xytext=(-5, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=colors[j])
        if len(y) > 1:
            ax.annotate(f'{y[-1]:.1f}', xy=(x[-1], y[-1]),
                        xytext=(5, -12), textcoords='offset points',
                        fontsize=9, fontweight='bold', color=colors[j])

    ax.set_xticks(range(6))
    ax.set_xticklabels(task_labels, fontsize=12)
    ax.set_xlabel('Training Stage', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(38, 98)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {filename}')


plot_single_forgetting(hide, 'HiDe-LLaVA Forgetting Curves', 'charm/forgetting_hide.png')
plot_single_forgetting(hybrid, 'HiDe-LLaVA + Task Prompt Forgetting Curves', 'charm/forgetting_hybrid.png')

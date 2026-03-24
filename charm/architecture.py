"""
HiDe-LLaVA + Task Prompt — Training Stage Architecture
Style: open layout, flowing arrows, icons, soft colors (following original Fig.3)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon
import matplotlib.patheffects as pe
import matplotlib
import numpy as np

from matplotlib.font_manager import FontProperties, fontManager
_cn_font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fontManager.addfont(_cn_font_path)
_cn_font_name = FontProperties(fname=_cn_font_path).get_name()
matplotlib.rcParams['font.family'] = [_cn_font_name, 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(15, 8.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.5)
ax.axis('off')
fig.patch.set_facecolor('white')

# ═══ 配色 ═══
C_llm      = '#b8e6d0'
C_llm_e    = '#6bc4a0'
C_proj     = '#f4a7a0'
C_proj_e   = '#e07060'
C_tok      = '#fce9a0'
C_tok_e    = '#d4b84a'
C_lora     = '#f2d0e0'
C_lora_e   = '#d48aac'
C_frozen   = '#e8e8e8'
C_frozen_e = '#bbb'
C_prompt   = '#e84040'
C_prompt_bg= '#fce4e4'
C_prompt_e = '#c0392b'
C_arrow    = '#6bc4a0'
C_gray     = '#999'
C_text     = '#333'

# ═══ 工具 ═══
def rbox(x, y, w, h, fc, ec='none', label='', fs=10, fc_t='#333',
         bold=False, lw=1.2, alpha=1.0, zorder=2, pad=0.1):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       alpha=alpha, zorder=zorder)
    ax.add_patch(b)
    if label:
        ax.text(x+w/2, y+h/2, label, ha='center', va='center',
                fontsize=fs, color=fc_t, fontweight='bold' if bold else 'normal',
                linespacing=1.2, zorder=zorder+1)

def arr(x1, y1, x2, y2, color=C_arrow, lw=1.5, rad=0, zorder=1):
    cs = f'arc3,rad={rad}'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>,head_width=0.25,head_length=0.15',
                                color=color, lw=lw, connectionstyle=cs),
                zorder=zorder)

def fire(cx, cy, s=0.1):
    xs = [cx, cx-s*0.5, cx-s*0.25, cx, cx+s*0.25, cx+s*0.5, cx]
    ys = [cy+s*1.4, cy+s*0.2, cy-s*0.3, cy+s*0.15, cy-s*0.3, cy+s*0.2, cy+s*1.4]
    ax.fill(xs, ys, color='#e85030', zorder=5, alpha=0.85)

def snowflake(cx, cy, s=0.07, color='#6ab8d8'):
    for a in [0, 60, 120]:
        r = np.radians(a)
        dx, dy = s*np.cos(r), s*np.sin(r)
        ax.plot([cx-dx, cx+dx], [cy-dy, cy+dy], color=color, lw=1.3, zorder=5)

def lora_hg(cx, cy, w=0.22, h=0.35, color=C_lora, ec=C_lora_e):
    ax.add_patch(Polygon([(cx-w,cy+h),(cx+w,cy+h),(cx,cy)], fc=color, ec=ec, lw=1, zorder=3))
    ax.add_patch(Polygon([(cx-w,cy-h),(cx+w,cy-h),(cx,cy)], fc=color, ec=ec, lw=1, zorder=3))

def diamond(cx, cy, s=0.13, color='#f0a0a0', ec='#d06060'):
    ax.add_patch(Polygon([(cx,cy+s),(cx+s,cy),(cx,cy-s),(cx-s,cy)],
                         fc=color, ec=ec, lw=1, zorder=4))

def circle(cx, cy, r=0.1, color='#f0a0a0', ec='#d06060'):
    ax.add_patch(plt.Circle((cx,cy), r, fc=color, ec=ec, lw=1, zorder=4))

def img_icon(cx, cy, s=0.35):
    ax.add_patch(FancyBboxPatch((cx-s,cy-s*0.7), s*2, s*1.4,
                 boxstyle="round,pad=0.02", fc='#f5f5f5', ec='#aaa', lw=1, zorder=2))
    ax.plot([cx-s*0.6,cx-s*0.15,cx+s*0.15,cx+s*0.45,cx+s*0.6],
            [cy-s*0.2,cy+s*0.2,cy-s*0.05,cy+s*0.3,cy-s*0.2],
            color='#6ab06a', lw=1.5, zorder=3)
    ax.add_patch(plt.Circle((cx+s*0.3,cy+s*0.15), s*0.1, fc='#ffd700', ec='none', zorder=3))

def txt_icon(cx, cy, s=0.3):
    ax.add_patch(FancyBboxPatch((cx-s,cy-s*0.7), s*2, s*1.4,
                 boxstyle="round,pad=0.02", fc='#fafafa', ec='#aaa', lw=1, zorder=2))
    for i in range(3):
        w = s*1.1 if i < 2 else s*0.6
        ax.plot([cx-s*0.55, cx-s*0.55+w], [cy+s*0.3-i*s*0.35]*2,
                color='#ccc', lw=1.8, zorder=3)

# ══════════════════════════════════════════════════════════
# 布局 — 左侧训练流程，右侧 Prompt Pool
# ══════════════════════════════════════════════════════════

# ── (1) 底部: 输入 icons ──
img_icon(0.9, 0.6)
txt_icon(3.0, 0.6)

# ── (2) 编码器 ──
rbox(0.1, 1.55, 1.5, 0.55, C_proj, C_proj_e, 'CLIP ViT-L/14\n@336px', fs=7.5, bold=True, fc_t='white')
rbox(1.8, 1.55, 1.2, 0.55, C_proj, C_proj_e, 'MLP\nProjector', fs=7.5, bold=True, fc_t='white')
rbox(3.3, 1.55, 1.3, 0.55, C_tok, C_tok_e, 'LLaMA\nTokenizer', fs=7.5, bold=True)
rbox(4.9, 1.55, 1.5, 0.55, C_tok, C_tok_e, 'CLIP Text\nEncoder', fs=7.5, bold=True)

fire(2.3, 2.22, s=0.08)

arr(0.9, 1.0, 0.85, 1.53, color=C_arrow, lw=1.2)
arr(1.2, 1.0, 2.3, 1.53, color=C_arrow, lw=1.2, rad=-0.15)
arr(3.0, 1.0, 3.95, 1.53, color=C_arrow, lw=1.2)

# ── (3) Token 序列 — 彩色方块 ──
sy = 2.7
sh = 0.5

# System (灰×2)
for i in range(2):
    rbox(0.4+i*0.35, sy, 0.28, sh, C_frozen, C_frozen_e, '', pad=0.03)

# Task Prompt (红×4 + 火焰)
for i in range(4):
    rbox(1.25+i*0.28, sy, 0.22, sh, C_prompt, C_prompt_e, '', pad=0.03)
fire(1.6, 2.58, s=0.07)

ax.text(1.6, sy+sh+0.2, 'Task Prompt\n(10×4096)', ha='center', fontsize=7.5,
        color=C_prompt, fontweight='bold')
ax.text(1.6, sy+sh+0.52, 'NEW', ha='center', fontsize=6.5, fontweight='bold',
        color='white', bbox=dict(boxstyle='round,pad=0.1', fc=C_prompt, ec=C_prompt_e, lw=1),
        zorder=5)

# dots
ax.text(2.55, sy+sh/2, '···', ha='center', va='center', fontsize=11, color=C_gray)

# Image tokens (蓝×5)
for i in range(5):
    rbox(2.85+i*0.3, sy, 0.24, sh, '#a8cce8', '#6a9fc7', '', pad=0.03)

# dots
ax.text(4.5, sy+sh/2, '···', ha='center', va='center', fontsize=11, color=C_gray)

# Text tokens (绿×3)
for i in range(3):
    rbox(4.75+i*0.3, sy, 0.24, sh, '#b8deb8', '#5a9e5f', '', pad=0.03)

# dots
ax.text(5.8, sy+sh/2, '···', ha='center', va='center', fontsize=11, color=C_gray)

# 编码器 → tokens
arr(2.3, 2.12, 3.3, 2.68, color=C_arrow, lw=1.1)
arr(3.95, 2.12, 5.1, 2.68, color=C_arrow, lw=1.1)

# ── (4) LLM ──
rbox(0.1, 4.0, 7.0, 3.6, C_llm, C_llm_e, '', lw=2, pad=0.2, alpha=0.45)
ax.text(3.6, 7.3, 'LLaMA-7B (Vicuna-v1.5)', ha='center', fontsize=12,
        fontweight='bold', color='#2a7a5a',
        path_effects=[pe.withStroke(linewidth=3, foreground='white')])

# tokens → LLM
arr(3.2, 3.22, 3.2, 3.98, color=C_arrow, lw=2)

# ── Frozen Weight ──
rbox(0.6, 5.3, 1.8, 1.1, C_frozen, C_frozen_e, 'Frozen\nPretrained\nWeight\n(LLaMA)', fs=8, bold=True, fc_t='#555')
snowflake(1.5, 6.55, s=0.09)

# ── LoRA 沙漏 + Remain / Top 标注 ──
# Remain Layers LoRA (左侧两个沙漏，黄色调)
remain_color = '#f5d88a'
remain_ec = '#c4a83a'
lora_hg(3.2, 5.6, w=0.22, h=0.35, color=remain_color, ec=remain_ec)
lora_hg(4.0, 5.6, w=0.22, h=0.35, color=remain_color, ec=remain_ec)
ax.text(3.6, 4.75, 'Remain Layers (1-31层)\n等权融合 LoRA, rank=8', ha='center', fontsize=7,
        color='#8a7a2a', style='italic')
ax.text(3.6, 6.25, r'$\epsilon=1.0$', ha='center', fontsize=7.5, color='#8a7a2a')

# Top Layer LoRA (右侧三个沙漏，粉色)
for i, lx in enumerate([5.2, 5.9, 6.6]):
    lora_hg(lx, 5.6, w=0.2, h=0.33)
    fire(lx, 5.15, s=0.06)

ax.text(5.9, 4.75, 'Top Layer (32层)\nMoE LoRA Experts', ha='center', fontsize=7,
        color=C_lora_e, style='italic')
ax.text(5.9, 6.25, r'$O_{top}=\sum d_i \cdot E_i(h)$', ha='center', fontsize=7, color=C_lora_e)

# 分隔线 (Remain | Top)
ax.plot([4.6, 4.6], [4.3, 6.7], color='#aaa', lw=0.8, ls='--', zorder=1)

# ── (5) Anchor 提取 (右下) ──
ax.text(7.8, 1.55, '训练时通过 CLIP 提取\nImage / Text Anchor',
        ha='center', fontsize=7.5, color=C_gray, style='italic')

diamond(7.2, 0.95, s=0.13, color='#f0a0a0', ec='#d06060')
ax.text(7.5, 0.95, 'Image Anchor', ha='left', fontsize=7.5, color='#b05050')
circle(7.2, 0.5, r=0.1, color='#f0a0a0', ec='#d06060')
ax.text(7.5, 0.5, 'Text Anchor', ha='left', fontsize=7.5, color='#b05050')

# 编码器→anchor 虚线
arr(1.5, 2.12, 7.1, 1.05, color='#ccc', lw=0.9, rad=-0.1)
arr(5.65, 2.12, 7.1, 0.6, color='#ccc', lw=0.9, rad=0.1)

# ── (6) Task Prompt Pool (右侧) ──
px, py = 9.2, 2.2
rbox(px, py, 4.5, 5.0, C_prompt_bg, C_prompt_e, '', lw=1.5, alpha=0.3, pad=0.15)
ax.text(px+2.25, py+5.2, 'Task Prompt Pool', ha='center', fontsize=12,
        fontweight='bold', color=C_prompt_e)

prompts = ['Prompt 1（当前任务）', 'Prompt 2', 'Prompt 3',
           'Prompt 4', 'Prompt 5', 'Prompt 6']
for i, label in enumerate(prompts):
    yi = py + 4.2 - i * 0.72
    if i == 0:
        fc, ec, tc, al, lw = C_prompt, C_prompt_e, 'white', 1.0, 1.8
    else:
        fc, ec, tc, al, lw = '#f5c8c8', '#e0a8a8', '#888', 0.5, 1.0
    rbox(px+0.4, yi, 3.6, 0.5, fc, ec, label, fs=9, fc_t=tc, alpha=al, lw=lw, bold=(i==0))
    if i == 0:
        fire(px+4.2, yi+0.28, s=0.08)
    else:
        snowflake(px+4.2, yi+0.25, s=0.06, color='#ccc')

ax.text(px+2.25, py-0.2, '仅训练当前 Prompt，历史冻结',
        ha='center', fontsize=7.5, color=C_prompt_e, style='italic')

# Pool → Task Prompt 弧线 (指向 token 序列中的红色方块)
arr(px, py+3.5, 1.6, 3.25, color=C_prompt, lw=2, rad=0.4, zorder=3)
ax.text(5.5, 3.7, '按 task_id 选取', ha='center', fontsize=8,
        color=C_prompt_e, style='italic',
        bbox=dict(boxstyle='round,pad=0.12', fc='white', ec=C_prompt_e, lw=0.7, alpha=0.9),
        zorder=4)

# ── (7) Output ──
arr(3.6, 7.62, 3.6, 8.1, color=C_arrow, lw=2)
ax.text(3.6, 8.25, 'Output', ha='center', fontsize=11, fontweight='bold', color=C_text)

# ── 标题 ──
ax.text(7.5, 8.25, '(a) HiDe-LLaVA + Task Prompt 训练阶段',
        ha='center', fontsize=14, fontweight='bold', color=C_text)

plt.tight_layout(pad=0.2)
plt.savefig('charm/architecture_hybrid.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: charm/architecture_hybrid.png')

# Project Context

## What is this project

在 HiDe-LLaVA（ACL 2025）基础上，通过添加 learnable task prompt 进行改进的多模态大模型持续指令微调框架。

- HiDe-LLaVA 原始论文：`HiDe-LLaVA.pdf`（已gitignore）
- 基线代码来自：https://github.com/Ghy0501/HiDe-LLaVA
- 我们的仓库：https://github.com/0evja/hybridllava
- 当前 tag：`v1.0-hybrid-prompt`（input-level task prompt 版本）

## Architecture Overview

```
HiDe-LLaVA 原始架构:
- Remain Layers (1-31): 所有已学任务的 LoRA 固定等权融合 (ε=1.0)
- Top Layer (32): MoE 加权选择 task-specific LoRA expert
- 路由: cosine_sim(CLIP_feature, anchor) 等权平均 image+text

v1.0 修改 (input-level task prompt):
- 在输入序列 system token 后插入 task_prompts[task_id] (10个4096维提示嵌入)
- ManualPromptUpdateCallback 绕过 DeepSpeed 手动 AdamW 更新 prompt (lr=6.4e-3)
- Prompt 位置: [sys][PROMPT][image][text]

P0 修改 (当前正在训练):
- Prompt 位置移到 image token 之后: [sys][image][PROMPT][text]
- Prompt lr 从 6.4e-3 降到 5e-4 (multiplier 从 32.0 改为 2.5)
- 删除了未使用的 task_keys 和正交损失代码
```

## Key Files

| File | Role |
|------|------|
| `llava/model/language_model/llava_hybrid.py` | Hybrid 模型定义 (task_prompts, anchors) |
| `llava/model/llava_arch.py` | Prompt 注入 & anchor 路由 (prepare_inputs_labels_for_multimodal) |
| `llava/train/train_hybrid.py` | 训练入口 + ManualPromptUpdateCallback |
| `llava/train/run.py` | 入口脚本，调用 train_hybrid.train() |
| `llava/train/train_MOE.py` | HiDe-LLaVA 原始训练入口 |
| `llava/train/llava_trainer.py` | LlavaHybridTrainer (optimizer 排除 prompt) |
| `llava/model/builder.py` | 模型加载 (训练 & 推理) |
| `llava/eval/model_answer.py` | 评估脚本 |
| `HiDe/peft/tuners/clitmoelora.py` | LoRA-MoE 实现 (HiDeMOELoraLinear) |
| `scripts/HiDe/Train_UCIT/train_all.sh` | 6任务连续训练脚本 |

## Experimental Results

### v1.0 (input-level prompt, lr=6.4e-3)

提升有限，核心原因：prompt 在 image 前干扰位置编码，与 MoE 路由功能冗余。

- 改善：ArxivQA (+1.83%), IconQA (+5.03%)
- 下降：CLEVR (-7.50%)
- Last Avg: 67.61% vs baseline 67.69%

### P0 (训练中)

改动：prompt 移到 image 后，lr 降到 5e-4。输出目录：`output/ucit_p0/`
[system Prompt] [Prompt] [image token] [text token] --->
[system Prompt] [image token] [Prompt] [text token]

## Planned Improvements

### P1 - 核心改进
<!-- 1. **经验回放**：每个已学任务保存 5-10% 样本，训练新任务时混合回放 -->
2. **非固定任务序列**：同一任务以不同数据子集交错重现，验证知识积累能力

### P2 - 进阶优化
<!-- 3. **可学习路由器**：用 nn.Linear(clip_dim*2, num_tasks) 替代 cosine similarity -->
4. **增大 LoRA rank**：总 rank 从 64 增到 96-128

## Git Workflow

- remote `origin`: 上游 HiDe-LLaVA 仓库
- remote `mygithub`: git@github.com:0evja/hybridllava.git
- 推送用 `git push mygithub main`
- 每个大版本打 tag（如 v1.0-hybrid-prompt, v2.0-adaptive-fusion）

## Dev Notes

- 训练脚本在 `scripts/HiDe/Train_UCIT/`，评估在 `scripts/HiDe/Eval_UCIT/`
- DeepSpeed ZeRO-2，3卡训练（GPU 0,3,4）
- batch_size=1 推理（每个样本独立路由）
- Python 环境：`conda activate hybrid`

## 数据集路径

| 数据集 | image_folder | 图片路径格式 |
|--------|-------------|-------------|
| ImageNet-R | `/data1/zms` | `ImageNet-R/train/xxx.jpg` |
| ArxivQA | `/data1/zms/datasets` | `ArxivQA/images/xxx.jpg` |
| VizWiz | `/data1/zms/datasets` | `VizWiz/...` |
| IconQA | `/data1/zms/datasets` | `IconQA/...` |
| CLEVR-Math | `/data1/zms/datasets` | `CLEVR/...` |
| Flickr30k | `/data1/zms/datasets` | `Flickr30k/...` |

模型权重：`models/llava-v1.5-7b`, `models/clip-vit-large-patch14-336`

## 训练命令

```bash
# 创建日志目录
mkdir -p logs

# 启动 tmux 会话
tmux new -s p0_train

# 在 tmux 内执行
cd /home/hechen/zms/MLLM_Factory/HiDe-LLaVA
conda activate hybrid
bash scripts/HiDe/Train_UCIT/train_all.sh 2>&1 | tee logs/p0_train.log

# Ctrl+B, D 退出 tmux
# tmux attach -t p0_train 重新进入
```

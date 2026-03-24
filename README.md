# Hybrid-LLaVA: Task Prompt Enhanced Continual Instruction Tuning for MLLM

Based on [HiDe-LLaVA](https://github.com/Ghy0501/HiDe-LLaVA) (ACL 2025), this repo explores adding **learnable task prompts** to the hierarchical decoupling framework for continual instruction tuning of Multimodal Large Language Models.

## Method

HiDe-LLaVA decouples the model into:
- **Top layer (Layer 32)**: Task-specific LoRA expansion via MoE with anchor-based routing
- **Remain layers (Layer 1-31)**: Task-general LoRA fusion

**Our modification**: We add a learnable task prompt (`prefix_len=10, hidden_size=4096`) for each task, injected into the input sequence after the system token. During training, prompts are updated via a manual AdamW callback (compatible with DeepSpeed ZeRO-2). An orthogonal loss encourages diversity across task prompts.

### Architecture

```
Input: [system_token] [task_prompt] [image_tokens] [text_tokens]
                          ↑
                   Learnable per-task
                   prefix embedding

LLM Layers 1-31:  Fused LoRA (all learned tasks merged)
LLM Layer 32:     MoE LoRA (weighted expert selection via image/text anchor similarity)
```

### Key Files

| File | Description |
|------|-------------|
| `llava/model/language_model/llava_hybrid.py` | Hybrid model definition (task_prompts, anchors) |
| `llava/model/llava_arch.py` | Prompt injection & anchor-based routing logic |
| `llava/train/train_hybrid.py` | Training pipeline with ManualPromptUpdateCallback |
| `llava/train/llava_trainer.py` | LlavaHybridTrainer with orthogonal loss |
| `llava/model/builder.py` | Model loading for training & inference |
| `llava/eval/model_answer.py` | Evaluation with expert routing |
| `test_CKA_sim.py` | CKA similarity analysis tool |
| `HiDe/peft/tuners/clitmoelora.py` | LoRA-MoE implementation |

## Results on UCIT Benchmark

### HiDe-LLaVA (Baseline Reproduction)

| Task | Image-R | ArxivQA | Viz-cap | IconQA | CLEVR | Flickr30k | Avg |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| task1 | 91.87% | | | | | | |
| task2 | 91.30% | 93.27% | | | | | |
| task3 | 89.10% | 92.33% | 54.88% | | | | |
| task4 | 87.23% | 90.83% | 49.07% | 81.67% | | | |
| task5 | 84.73% | 91.57% | 46.19% | 66.67% | 67.03% | | |
| task6 | 83.50% | 90.60% | 48.51% | 66.60% | 61.63% | 55.32% | 67.69% |

### Hybrid-LLaVA (Ours)

| Task | Image-R | ArxivQA | Viz-cap | IconQA | CLEVR | Flickr30k | Avg |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| task1 | 91.17% | | | | | | |
| task2 | 90.73% | 93.23% | | | | | |
| task3 | 89.03% | 93.07% | 59.77% | | | | |
| task4 | 87.13% | 90.77% | 50.92% | 84.67% | | | |
| task5 | 86.13% | 91.87% | 49.21% | 73.00% | 63.20% | | |
| task6 | 83.97% | 92.43% | 47.12% | 71.63% | 54.13% | 56.40% | 67.61% |

**Key observations**:
- Improved retention on text-heavy tasks: ArxivQA (+1.83%), IconQA (+5.03%) at task6
- Degradation on spatial reasoning: CLEVR (-7.50%) at task6
- Overall Last metric comparable (67.61% vs 67.69%)

## Roadmap

### Current Status (v1.0 — Input-level Task Prompt)

Initial attempt: inject learnable task prompt into input sequence. Results show improvement on semantic tasks (ArxivQA +1.83%, IconQA +5.03%) but degradation on spatial reasoning (CLEVR -7.50%). Overall 67.61% vs baseline 67.69%.

**Identified issues**:
1. Prompt inserted before image tokens disrupts visual position encoding → CLEVR drop
2. Input-level prompt overlaps functionally with top-layer MoE routing (both do task discrimination)

### Next Steps

#### 1. Task Prompt Architecture Optimization

- [ ] Move prompt position: `[sys][image][PROMPT][text]` — fix CLEVR degradation
- [ ] Lower prompt lr from 6.4e-3 to ~5e-4 (same order as LoRA lr 2e-4)
- [ ] **Core innovation**: Use task embedding to generate per-layer adaptive fusion coefficients for remain layers, replacing fixed ε=1.0. This complements the top-layer MoE (top layer selects experts, remain layers adapt fusion weights)

#### 2. Experience Replay

- [ ] Save 5-10% representative samples per learned task
- [ ] Mix replay data when training new tasks to provide gradient signal for old knowledge retention

#### 3. Non-stationary Task Sequence Evaluation

- [ ] Current UCIT benchmark: each task appears only once in fixed order — unrealistic
- [ ] Construct interleaved task sequences where the same task recurs with different data subsets
- [ ] Example: ImageNet-R(subset1) → ArxivQA(subset1) → IconQA(subset1) → ImageNet-R(subset2) → IconQA(subset2) → ...
- [ ] Evaluate whether the model can accumulate knowledge (not just resist forgetting) under task recurrence

#### 4. Ablation & Paper

- [ ] Full ablation: prompt position, fusion method, replay ratio
- [ ] Compare with HiDe-LLaVA, CoIN, O-LoRA baselines
- [ ] Write thesis

### Timeline

| Period | Goal |
|--------|------|
| 03.20 — 04.06 | Architecture optimization & initial experiments |
| 04.07 — 04.20 | Comparative experiments & ablation study |
| 04.21 — 04.27 | Thesis first draft |
| 04.28 — 05.15 | Revision & defense preparation |

## Installation

Same environment as [CoIN](https://github.com/zackschen/CoIN):

```bash
conda create -n hide python=3.10 -y
conda activate hide
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install nltk==3.9.1 pycocotools==2.0.8 pycocoevalcap==1.2
```

## Data Preparation

Please refer to the original [HiDe-LLaVA](https://github.com/Ghy0501/HiDe-LLaVA) repo for UCIT benchmark dataset setup, including image downloads and instruction organization.

Pre-trained weights: [LLaVA-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) and [CLIP-ViT-L/14-336](https://huggingface.co/openai/clip-vit-large-patch14-336).

## Training & Evaluation

```bash
# Train (example: Task 2 with hybrid model)
bash scripts/HiDe/Train_UCIT/Task2_hybrid.sh

# Evaluate
bash scripts/HiDe/Eval_UCIT/Eval_all.sh
```

> Modify paths in all `.sh` files to match your environment.

## Acknowledgement

- [HiDe-LLaVA](https://github.com/Ghy0501/HiDe-LLaVA) - Guo et al., ACL 2025
- [LLaVA](https://github.com/haotian-liu/LLaVA) - Liu et al.
- [CoIN](https://github.com/zackschen/CoIN) - Chen et al.

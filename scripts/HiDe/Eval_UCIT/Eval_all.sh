#!/bin/bash

# ============================================================
# Eval_all.sh - UCIT Benchmark Full Evaluation (P0)
# ============================================================
# Task order: 0=ImageNet-R, 1=ArxivQA, 2=VizWiz, 3=IconQA, 4=CLEVR-Math, 5=Flickr30k
#
# After training task N (0-indexed), evaluate on all tasks 0..N
# using the model checkpoint from TaskN+1_llava_lora_ours
#
# Usage:
#   sh Eval_all.sh [GPU] [OUTPUT_BASE_DIR]
# ============================================================

set -e

GPU=${1:-"0"}
OUTPUT_BASE="${2:-/home/hechen/zms/MLLM_Factory/HiDe-LLaVA/output/ucit_p0}"

EVAL_DIR="$(dirname "$0")"
TAG="p0"  # 结果文件前缀，与 baseline 区分
export RESULT_BASE="./results/UCIT_P0"  # 结果保存目录

echo "============================================"
echo " UCIT Full Evaluation (P0)"
echo " GPU: $GPU"
echo " Model dir: $OUTPUT_BASE"
echo "============================================"

# --- After Task 1: evaluate 1 dataset ---
echo ""
echo ">>> Evaluating Task1 model (expert_num=1) on 1 dataset..."
MODEL="${OUTPUT_BASE}/Task1_llava_lora_ours"
EXPERT_NUM=1
sh ${EVAL_DIR}/eval_imagenet.sh  ${TAG}-task1 "$MODEL" $GPU $EXPERT_NUM

# --- After Task 2: evaluate 2 datasets ---
echo ""
echo ">>> Evaluating Task2 model (expert_num=2) on 2 datasets..."
MODEL="${OUTPUT_BASE}/Task2_llava_lora_ours"
EXPERT_NUM=2
sh ${EVAL_DIR}/eval_imagenet.sh  ${TAG}-task2 "$MODEL" $GPU $EXPERT_NUM
sh ${EVAL_DIR}/eval_arxivqa.sh   ${TAG}-task2 "$MODEL" $GPU $EXPERT_NUM

# --- After Task 3: evaluate 3 datasets ---
echo ""
echo ">>> Evaluating Task3 model (expert_num=3) on 3 datasets..."
MODEL="${OUTPUT_BASE}/Task3_llava_lora_ours"
EXPERT_NUM=3
sh ${EVAL_DIR}/eval_imagenet.sh  ${TAG}-task3 "$MODEL" $GPU $EXPERT_NUM
sh ${EVAL_DIR}/eval_arxivqa.sh   ${TAG}-task3 "$MODEL" $GPU $EXPERT_NUM
sh ${EVAL_DIR}/eval_vizwiz.sh    ${TAG}-task3 "$MODEL" $GPU $EXPERT_NUM

# --- After Task 4: evaluate 4 datasets ---
echo ""
echo ">>> Evaluating Task4 model (expert_num=4) on 4 datasets..."
MODEL="${OUTPUT_BASE}/Task4_llava_lora_ours"
EXPERT_NUM=4
sh ${EVAL_DIR}/eval_imagenet.sh  ${TAG}-task4 "$MODEL" $GPU $EXPERT_NUM
sh ${EVAL_DIR}/eval_arxivqa.sh   ${TAG}-task4 "$MODEL" $GPU $EXPERT_NUM
sh ${EVAL_DIR}/eval_vizwiz.sh    ${TAG}-task4 "$MODEL" $GPU $EXPERT_NUM
sh ${EVAL_DIR}/eval_iconqa.sh    ${TAG}-task4 "$MODEL" $GPU $EXPERT_NUM

# --- After Task 5: evaluate 5 datasets ---
echo ""
echo ">>> Evaluating Task5 model (expert_num=5) on 5 datasets..."
MODEL="${OUTPUT_BASE}/Task5_llava_lora_ours"
EXPERT_NUM=5
sh ${EVAL_DIR}/eval_imagenet.sh  ${TAG}-task5 "$MODEL" $GPU $EXPERT_NUM
sh ${EVAL_DIR}/eval_arxivqa.sh   ${TAG}-task5 "$MODEL" $GPU $EXPERT_NUM
sh ${EVAL_DIR}/eval_vizwiz.sh    ${TAG}-task5 "$MODEL" $GPU $EXPERT_NUM
sh ${EVAL_DIR}/eval_iconqa.sh    ${TAG}-task5 "$MODEL" $GPU $EXPERT_NUM
sh ${EVAL_DIR}/eval_clevr.sh     ${TAG}-task5 "$MODEL" $GPU $EXPERT_NUM

# --- After Task 6: evaluate all 6 datasets ---
echo ""
echo ">>> Evaluating Task6 model (expert_num=6) on all 6 datasets..."
MODEL="${OUTPUT_BASE}/Task6_llava_lora_ours"
EXPERT_NUM=6
sh ${EVAL_DIR}/eval_imagenet.sh  ${TAG}-task6 "$MODEL" $GPU $EXPERT_NUM
sh ${EVAL_DIR}/eval_arxivqa.sh   ${TAG}-task6 "$MODEL" $GPU $EXPERT_NUM
sh ${EVAL_DIR}/eval_vizwiz.sh    ${TAG}-task6 "$MODEL" $GPU $EXPERT_NUM
sh ${EVAL_DIR}/eval_iconqa.sh    ${TAG}-task6 "$MODEL" $GPU $EXPERT_NUM
sh ${EVAL_DIR}/eval_clevr.sh     ${TAG}-task6 "$MODEL" $GPU $EXPERT_NUM
sh ${EVAL_DIR}/eval_flickr30k.sh ${TAG}-task6 "$MODEL" $GPU $EXPERT_NUM

echo ""
echo "============================================"
echo " All evaluations finished!"
echo " Results in: ${RESULT_BASE}/each_dataset/*/${TAG}-task*/"
echo "============================================"

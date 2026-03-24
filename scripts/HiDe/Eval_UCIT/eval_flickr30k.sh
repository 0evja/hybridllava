#!/bin/bash

# --- Usage ---
# sh eval_flickr30k.sh <STAGE> <MODEL_PATH> <GPU> <EXPERT_NUM>

STAGE=${1:-"hide"}
MODEL_PATH=${2:-"/home/hechen/zms/MLLM_Factory/HiDe-LLaVA/output/ucit/Task6_llava_lora_ours"}
GPU=${3:-"0"}
EXPERT_NUM=${4:-6}

# --- Fixed paths ---
BASE_DIR="/home/hechen/zms/MLLM_Factory/HiDe-LLaVA"
MODEL_BASE="${BASE_DIR}/models/llava-v1.5-7b"
TEXT_TOWER="${BASE_DIR}/models/clip-vit-large-patch14-336"
QUESTION_FILE="${BASE_DIR}/instructions/Flickr30k/test_3000.json"
ANNOTATION_FILE="${BASE_DIR}/instructions/Flickr30k/val_coco_type_3000.json"
IMAGE_FOLDER="/data1/zms/datasets"
RESULT_DIR="${RESULT_BASE:-./results/UCIT}/each_dataset/Flickr30k"

# --- Derived params ---
CUR_TASK=$((EXPERT_NUM - 1))

CHUNKS=1
IDX=0

mkdir -p "$RESULT_DIR/$STAGE"

echo "Starting evaluation: Flickr30k | expert_num=$EXPERT_NUM, cur_task=$CUR_TASK, GPU=$GPU"

CUDA_VISIBLE_DEVICES=$GPU python -m llava.eval.model_answer \
    --model-path "$MODEL_PATH" \
    --model-base "$MODEL_BASE" \
    --question-file "$QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --text-tower "$TEXT_TOWER" \
    --answers-file "$RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl" \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --cur-task $CUR_TASK \
    --expert-num $EXPERT_NUM \
    --conv-mode vicuna_v1 \
    --max_new_tokens 32 \
    --top_p 1.0 \
    --num_beams 1

# --- Merge & evaluate (caption metric) ---
output_file="$RESULT_DIR/$STAGE/merge.jsonl"
> "$output_file"
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

python -m llava.eval.eval_caption \
    --annotation-file "$ANNOTATION_FILE" \
    --result-file "$output_file" \
    --output-dir "$RESULT_DIR/$STAGE"

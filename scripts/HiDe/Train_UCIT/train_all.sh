#!/bin/bash
# ============================================================
# HiDe-LLaVA + Task Prompt (P0): 6-task sequential training
# GPU: 0,3,4  |  Entry: llava/train/run.py (-> train_hybrid.py)
# ============================================================

set -e  # 任一任务失败则停止

BASE_DIR="/home/hechen/zms/MLLM_Factory/HiDe-LLaVA"
DATA_DIR="/data1/zms/datasets"
GPUS="localhost:0,3,4"
PORT=29601
PROMPT_VERSION=v1

# 输出目录前缀（与 baseline 区分）
OUT_PREFIX="${BASE_DIR}/output/ucit_p0"

# 公共参数
COMMON_ARGS="
    --deepspeed ./scripts/zero2.json
    --lora_enable True --lora_r 48 --lora_alpha 96 --mm_projector_lr 2e-5
    --expert_num 6
    --num_tasks 6
    --model_name_or_path ${BASE_DIR}/models/llava-v1.5-7b
    --version ${PROMPT_VERSION}
    --vision_tower ${BASE_DIR}/models/clip-vit-large-patch14-336
    --text_tower ${BASE_DIR}/models/clip-vit-large-patch14-336
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio pad
    --group_by_modality_length True
    --bf16 False
    --fp16 True
    --num_train_epochs 1
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 2
    --gradient_accumulation_steps 16
    --evaluation_strategy no
    --save_strategy epoch
    --learning_rate 2e-4
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --logging_steps 1
    --tf32 False
    --model_max_length 2048
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --lazy_preprocess True
    --report_to none
"

# 任务定义: task_id | data_path | image_folder
TASKS=(
    "0|${BASE_DIR}/instructions/ImageNet-R/train.json|/data1/zms"
    "1|${BASE_DIR}/instructions/ArxivQA/train_4w.json|${DATA_DIR}"
    "2|${BASE_DIR}/instructions/VizWiz/train.json|${DATA_DIR}"
    "3|${BASE_DIR}/instructions/IconQA/train.json|${DATA_DIR}"
    "4|${BASE_DIR}/instructions/CLEVR-Math/train_4w.json|${DATA_DIR}"
    "5|${BASE_DIR}/instructions/Flickr30k/train_brief_4w.json|${DATA_DIR}"
)

PREV_MODEL=""

for task_entry in "${TASKS[@]}"; do
    IFS='|' read -r TASK_ID DATA_PATH IMAGE_FOLDER <<< "$task_entry"
    TASK_NUM=$((TASK_ID + 1))
    OUTPUT_DIR="${OUT_PREFIX}/Task${TASK_NUM}_llava_lora_ours"

    echo ""
    echo "============================================================"
    echo "  Training Task ${TASK_NUM} (cur_task=${TASK_ID})"
    echo "  Data: ${DATA_PATH}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "============================================================"
    echo ""

    PREV_ARG=""
    if [ -n "$PREV_MODEL" ]; then
        PREV_ARG="--previous_task_model_path ${PREV_MODEL}"
    fi

    deepspeed --include ${GPUS} --master_port ${PORT} llava/train/run.py \
        ${COMMON_ARGS} \
        --cur_task ${TASK_ID} \
        --data_path "${DATA_PATH}" \
        --image_folder "${IMAGE_FOLDER}" \
        --output_dir "${OUTPUT_DIR}" \
        ${PREV_ARG}

    PREV_MODEL="${OUTPUT_DIR}"
done

echo ""
echo "All 6 tasks completed! Output: ${OUT_PREFIX}/"

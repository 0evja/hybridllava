################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

BASE_DIR="/home/hechen/zms/MLLM_Factory/HiDe-LLaVA"
DATA_DIR="/data1/zms/datasets/"

deepspeed --include localhost:3,4,5 --master_port 29601 llava/train/run.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r 48 --lora_alpha 96 --mm_projector_lr 2e-5 \
    --expert_num 6 \
    --num_tasks 6 \
    --model_name_or_path "${BASE_DIR}"/models/llava-v1.5-7b \
    --previous_task_model_path "${BASE_DIR}"/output/ucit/Task5_llava_lora_ours \
    --version $PROMPT_VERSION \
    --data_path "${BASE_DIR}"/instructions/Flickr30k/train_brief_4w.json \
    --image_folder "${DATA_DIR}" \
    --vision_tower "${BASE_DIR}"/models/clip-vit-large-patch14-336 \
    --text_tower "${BASE_DIR}"/models/clip-vit-large-patch14-336 \
    --cur_task 5 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --output_dir "${BASE_DIR}"/output/ucit/Task6_llava_lora_ours \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --fp16 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

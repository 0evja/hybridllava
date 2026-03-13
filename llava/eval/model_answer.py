import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        
        # 处理不同的 Image Token 插入逻辑
        if getattr(self.model_config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "Currently, batch_size must be 1 for Hybrid expert routing consistency."
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # 初始化
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    # 加载模型 (这里会调用你之前修改过的含有 Hide.peft 和 non_lora 加载逻辑的 builder.py)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        args.model_base, 
        model_name, 
        text_tower=args.text_tower
    )

    # 【关键修复】加载完成后强制激活当前任务 + 确保 inference mode
    if hasattr(model, 'base_model'):
        actual_model = model.base_model.model if hasattr(model.base_model, 'model') else model.base_model
    else:
        actual_model = model
    if hasattr(actual_model, 'set_cur_task'):
        # expert_num = 已训练任务数 = cur_task + 1（Task 0 训练完 → 1 个专家可用）
        expert_num = getattr(args, 'expert_num', args.cur_task + 1)
        actual_model.set_cur_task(args.cur_task, expert_num=expert_num)
        print(f"✅ [Eval] Successfully activated cur_task = {args.cur_task}, expert_num = {expert_num}")
        
        # 额外保险：强制进入 inference 模式
        actual_model.eval()
        if hasattr(model, 'base_model'):
            model.base_model.eval()
    else:
        print("⚠️  Warning: model has no set_cur_task method")
        
    # 加载测试集
    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # 自动切换会话模式 (针对 plain 模型)
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'Auto switching conv-mode to {args.conv_mode}')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    # 推理循环
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        # 确保 Image Tensor 精度与模型一致 (bf16=False -> float16)
        image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)

        with torch.inference_mode():

            generate_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
            
            output_ids = generate_model.generate(
                input_ids=input_ids,
                images=image_tensor,          # 必须传 images，触发 Hybrid 的 multimodal prepare
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        
        # print(f"Generated Output: {outputs[:120]}...")   # 只打印前120字符，避免日志爆炸

        # 保存结果
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {"task_id": args.cur_task}
        }) + "\n")

    ans_file.close()
    print(f"Answers saved to {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--cur-task", type=int, default=0, help="Task ID to activate specific experts")
    parser.add_argument("--expert-num", type=int, default=None, help="Number of trained experts (default: cur_task + 1)")
    parser.add_argument("--temperature", type=float, default=0.0) # ImageNet-R 建议设为 0 以保证结果可复现
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--text-tower", type=str, default=None)
    args = parser.parse_args()
    # import pdb; pdb.set_trace()
    eval_model(args)
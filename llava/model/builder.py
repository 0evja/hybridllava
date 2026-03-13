import os
import sys
import warnings
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

# 导入自定义架构
from llava.model.language_model.llava_hybrid import LlavaHybridForCausalLM, LlavaHybridConfig
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", text_tower=None, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'llava' in model_name.lower():
        # --- 情况 A: 加载 LoRA 模型 ---
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print(f'Loading LLaVA from base model: {model_base}...')

            # LoRA 场景：先加载到 CPU，等所有权重（non-lora + LoRA）加载完再移到 CUDA
            lora_kwargs = {k: v for k, v in kwargs.items() if k != 'device_map'}
            lora_kwargs['torch_dtype'] = torch.float16

            if getattr(lora_cfg_pretrained, "model_type", None) == "llava_hybrid":
                print("Detected LlavaHybrid architecture, initializing...")
                model = LlavaHybridForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **lora_kwargs)
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **lora_kwargs)

            if text_tower is not None:
                print(f"Initializing CLIP tokenizer from: {text_tower}")
                clip_tokenizer = AutoTokenizer.from_pretrained(
                    text_tower,
                    model_max_length=77,
                    padding_side="right",
                    use_fast=True,
                )
                if hasattr(model, 'set_clip_tokenizer'):
                    model.set_clip_tokenizer(clip_tokenizer)
            
            if hasattr(model, 'set_tokenizer'):
                model.set_tokenizer(tokenizer)

            # --- 核心修复：加载 non-lora trainables (彻底剥离多余前缀) ---
            print('Loading additional LLaVA weights (non-lora trainables)...')
            non_lora_trainables_path = os.path.join(model_path, 'non_lora_trainables.bin')
            if os.path.exists(non_lora_trainables_path):
                non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu')
                
                print('Stripping recursive prefixes (base_model / model)...')
                cleaned_weights = {}
                for k, v in non_lora_trainables.items():
                    new_k = k
                    # 只剥离第一个匹配的前缀，避免过度剥离
                    for prefix in ['base_model.model.', 'base_model.', 'model.model.', 'model.']:
                        if new_k.startswith(prefix):
                            new_k = new_k[len(prefix):]
                            break
                    cleaned_weights[new_k] = v
                
                # strict=False 允许加载部分权重，并返回缺失/多余的 Key
                load_result = model.load_state_dict(cleaned_weights, strict=False, assign=True)

                # 诊断：显示加载了哪些关键权重
                loaded_keys = [k for k in cleaned_weights.keys() if any(x in k for x in ['task_prompts', 'anchor', 'boundary', 'mm_projector'])]
                print(f"Loaded non-lora keys (key subset): {loaded_keys[:10]}")

                missing_hide_keys = [k for k in load_result.missing_keys if any(x in k for x in ['task', 'anchor', 'boundary', 'mm_projector'])]
                if missing_hide_keys:
                    print(f"⚠️  WARNING: Missing keys during loading: {missing_hide_keys}")
                else:
                    print("✅ HiDe + mm_projector keys loaded successfully.")

                unexpected_keys = load_result.unexpected_keys
                if unexpected_keys:
                    print(f"⚠️  Unexpected keys (not in model): {unexpected_keys[:10]}")

            # 加载并合并 LoRA 权重
            from HiDe.peft import PeftModel
            print('Loading LoRA weights via HiDe.peft...')
            model = PeftModel.from_pretrained(model, model_path)
            # print('Merging LoRA weights...')
            # model = model.merge_and_unload()
            print('LoRA loaded in PEFT mode (no merge).')
            model.to(device=device)

        # --- 情况 B: 仅加载 Projector 模式 ---
        elif model_base is not None:
            print('Loading LLaVA from base model (Projector-only mode)...')
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            
            if getattr(cfg_pretrained, "model_type", None) == "llava_hybrid":
                model = LlavaHybridForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            model.load_state_dict({k: v.to(torch.float16) for k, v in mm_projector_weights.items()}, strict=False)
        
        # --- 情况 C: 直接加载完整模型 ---
        else:
            print(f'Loading full LLaVA model from {model_path}...')
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            if getattr(cfg_pretrained, "model_type", None) == "llava_hybrid":
                model = LlavaHybridForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    # --- 后处理：Vision & Text Towers ---
    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        if getattr(model.config, "mm_use_im_patch_token", True):
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

        if hasattr(model, 'get_text_tower'):
            text_tower_obj = model.get_text_tower()
            if text_tower_obj is not None:
                if not text_tower_obj.is_loaded:
                    text_tower_obj.load_model()
                text_tower_obj.to(device=device, dtype=torch.float16)

    model.eval()
    context_len = getattr(model.config, "max_sequence_length", 2048)

    return tokenizer, model, image_processor, context_len
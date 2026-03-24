#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import inspect

from .multimodal_encoder.builder import build_vision_tower, build_text_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from HiDe.peft.tuners import HiDeMOELoraModel
from collections import deque


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        
        if hasattr(config, "mm_text_tower"):
            self.text_tower = build_text_tower(config, delay_load=True)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_text_tower(self):
        text_tower = getattr(self, 'text_tower', None)
        if type(text_tower) is list:
            text_tower = text_tower[0]
        return text_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'),strict = False)

    def initialize_text_modules(self, model_args, fsdp=None):
        text_tower = model_args.text_tower

        if self.get_text_tower() is None:
            text_tower = build_text_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.text_tower = [text_tower]
            else:
                self.text_tower = text_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                text_tower = self.text_tower[0]
            else:
                text_tower = self.text_tower
            text_tower.load_model()


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_text_tower(self):
        return self.get_model().get_text_tower()

    def encode_images(self, images):
        clip_image_features, image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return clip_image_features.to(self.device), image_features.to(self.device)
    
    def debug_input_structure(cur_new_input_embeds, cur_new_labels, batch_idx):
        print(f"\n===== Input structure (batch {batch_idx}) =====")
        cur_pos = 0
        for i, tensor in enumerate(cur_new_input_embeds):
            part_len = tensor.shape[0]
            # 尝试识别类型
            if hasattr(tensor, 'dtype'):
                if hasattr(tensor, 'device'):
                    dev = tensor.device
                else:
                    dev = 'N/A'
            else:
                dev = 'N/A'
            part_type = "Unknown"
            if i == 0:
                part_type = "Text"  # 默认第一个是文本
            # 用特殊判断 prompt / image
            if hasattr(self, 'task_prompts') and tensor is target_prompts:
                part_type = "Prompt"
            elif 'image' in str(tensor.device).lower() or tensor.shape[1] == 576:  # 你的 image embed dim
                part_type = "Image"
            print(f"  Part {i:2d} | Type: {part_type:6} | shape: {tuple(tensor.shape)} | seq pos: [{cur_pos}-{cur_pos+part_len-1}]")
            cur_pos += part_len

        # labels
        print("  Labels shape:", cur_new_labels.shape)
        # 可视化 labels 中 IGNORE_INDEX 位置
        ignore_idx_count = (cur_new_labels == IGNORE_INDEX).sum().item()
        print(f"  Labels IGNORE_INDEX count: {ignore_idx_count}/{cur_new_labels.numel()} ({ignore_idx_count/cur_new_labels.numel():.2%})")
        print("=============================================\n")

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()

        # ====================== 【核心修复1】路由计算提前执行 ======================
        # 只有在有图像 + 不是纯自回归生成步（input_ids.shape[1] > 1）时才计算路由
        # 这样既保证训练和首次推理正确，又保证后续生成步复用上一次的 active_expert_idx
        if vision_tower is not None and images is not None and input_ids.shape[1] > 1:

            # --------------------- 图像特征提取 ---------------------
            if type(images) is list or images.ndim == 5:
                concat_images = torch.cat([image for image in images], dim=0)
                clip_image_features, projected_image_features = self.encode_images(concat_images)
                split_sizes = [image.shape[0] for image in images]
                image_features = torch.split(projected_image_features, split_sizes, dim=0)
                image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
                image_guide_features = clip_image_features
            else:
                image_guide_features, image_features = self.encode_images(images)

            assert image_features.shape[1] == 576, 'vision tower not a withprojection version.'

            # --------------------- 文本引导特征提取 ---------------------
            text_tower = self.get_text_tower()

            input_pad = np.where(
                input_ids.cpu().detach().numpy() != -200,
                input_ids.cpu().detach().numpy(),
                self.tokenizer.pad_token_id
            )
            decoded_inputs = self.tokenizer.batch_decode(input_pad, skip_special_tokens=True)
            decoded_hidden_inputs = ['\n'.join(decode_input.split('\n')[1:]) for decode_input in decoded_inputs]
            decoded_clip_inputs = [decode_input.split(' ASSISTANT')[0] for decode_input in decoded_hidden_inputs]

            clip_text_inputs = self.clip_tokenizer(
                decoded_clip_inputs,
                padding="longest",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            text_guide_features = text_tower(clip_text_inputs)

            # --------------------- 路由逻辑（训练 / 推理） ---------------------
            if self.training:
                current_image_features = image_guide_features
                current_text_features = text_guide_features
                task_id = self.cur_task
                self.active_expert_idx = task_id

                image_sum = self.image_anchors[task_id].data * self.image_boundary[task_id].data + current_image_features.detach().sum(dim=0)
                text_sum = self.text_anchors[task_id].data * self.text_boundary[task_id].data + current_text_features.detach().sum(dim=0)

                self.image_boundary[task_id].data += current_image_features.shape[0]
                self.text_boundary[task_id].data += current_text_features.shape[0]

                self.image_anchors[task_id].data.copy_(image_sum / self.image_boundary[task_id].data)
                self.text_anchors[task_id].data.copy_(text_sum / self.text_boundary[task_id].data)

            else:
                image_sim = []
                text_sim = []
                for image_anchor in self.image_anchors:
                    image_sims = F.cosine_similarity(image_guide_features.unsqueeze(1), image_anchor, dim=2)
                    image_sim.append(image_sims.max().item())
                for text_anchor in self.text_anchors:
                    text_sims = F.cosine_similarity(text_guide_features.unsqueeze(1), text_anchor, dim=2)
                    text_sim.append(text_sims.max().item())

                image_sim = np.array(image_sim[:self.expert_num])
                text_sim = np.array(text_sim[:self.expert_num])

                sim = (image_sim + text_sim) / 2
                sim_tensor = torch.tensor(sim, dtype=torch.float32, device=self.device)

                sim_softmax = F.softmax(sim_tensor / 0.1, dim=-1)
                self.active_expert_idx = torch.argmax(sim_softmax).item()

                compute_expert_weight = sim_softmax.tolist()

                proj_names = [
                    'q_proj', 'k_proj', 'v_proj', 'o_proj',  # self_attn 
                    'gate_proj', 'up_proj', 'down_proj'      # mlp 
                ]

                # 设置最后一层 LoRA 的 expert_weight（加权）
                for proj_name in proj_names:
                    if proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                        proj_layer = getattr(self.model.layers[-1].self_attn, proj_name)
                    else:
                        proj_layer = getattr(self.model.layers[-1].mlp, proj_name)

                    proj_layer.expert_weight = compute_expert_weight
                    # print(proj_layer.expert_weight)

        # ====================== 原始早返回逻辑（保持不变） ======================
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                ), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # ====================== 下面是原来的 embed + prompt 插入 + padding 逻辑 ======================
        # （完全保持你原来的代码，只做了极少量变量名对齐和健壮性小调整）

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full(
                        (cur_image_features.shape[0],), IGNORE_INDEX,
                        device=cur_labels.device, dtype=cur_labels.dtype
                    ))

            # ====================== Prompt 插入（你的架构核心） ======================
            if hasattr(self, 'task_prompts'):
                selected_task_id = self.active_expert_idx
                model_dtype = self.get_model().embed_tokens.weight.dtype
                target_prompts = self.task_prompts[selected_task_id].to(
                    dtype=model_dtype, device=cur_labels.device
                )
                
                # if batch_idx == 0:
                #     if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                #         print("\n[Prompt Debug]")
                #         print("selected_task_id:", selected_task_id)
                #         print("prompt grad enabled:", target_prompts.requires_grad)
                #         print("prompt dtype:", target_prompts.dtype)
                #         print("prompt shape:", target_prompts.shape)

                # 插入到 image token 之后：[sys][image][PROMPT][text]
                # 单图情况下，循环后 cur_new_input_embeds = [text_before_img, img, text_after_img]
                # image 在 index 1，所以 prompt 插入 index 2（image 之后）
                insert_pos = min(2, len(cur_new_input_embeds))
                cur_new_input_embeds.insert(insert_pos, target_prompts)
                cur_new_labels.insert(insert_pos, torch.full(
                    (target_prompts.shape[0],), IGNORE_INDEX,
                    device=cur_labels.device, dtype=cur_labels.dtype
                ))
            # def debug_input_structure(cur_new_input_embeds, cur_new_labels, batch_idx):
            #     """
            #     可视化每个 batch 的输入序列结构：
            #     - 显示每个部分类型（Text / Prompt / Image / Unknown）
            #     - 显示每个部分长度
            #     - 可视化 labels 中 IGNORE_INDEX 的比例
            #     """
            #     print(f"\n===== Input structure (batch {batch_idx}) =====")
                
            #     seq_visual = ""
            #     total_len = 0
                
            #     for i, tensor in enumerate(cur_new_input_embeds):
            #         part_len = tensor.shape[0]
            #         part_type = "Unknown"
                    
            #         # 判断类型
            #         if i == 0:
            #             part_type = "Text"
            #         elif hasattr(self, 'task_prompts') and tensor is getattr(self, 'task_prompts', [None])[0]:
            #             part_type = "Prompt"
            #         elif tensor.shape[1] == 576:
            #             part_type = "Image"
                    
            #         # 可视化拼接形式
            #         seq_visual += f"[{part_type}({part_len})]"
            #         total_len += part_len
                
            #     print(f"Sequence structure: {seq_visual} | total length: {total_len}")
                
            #     # labels 处理
            #     if isinstance(cur_new_labels, list):
            #         labels_cat = torch.cat(cur_new_labels)
            #     else:
            #         labels_cat = cur_new_labels
                
            #     ignore_idx_count = (labels_cat == IGNORE_INDEX).sum().item()
            #     print(f"Labels shape: {labels_cat.shape}")
            #     print(f"Labels IGNORE_INDEX count: {ignore_idx_count}/{labels_cat.numel()} ({ignore_idx_count/labels_cat.numel():.2%})")
            #     print("=============================================\n")
            # debug_input_structure(cur_new_input_embeds, cur_new_labels, batch_idx)
            # ====================== 必须在这里 cat ======================
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # 截断 + padding 逻辑（保持你原来的完整实现）
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        device = new_input_embeds[0].device

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=device)
        attention_mask_padded = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        position_ids_padded = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

        for i, (cur_new_embed, cur_new_labels_item) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]

            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels_item
                    attention_mask_padded[i, -cur_len:] = True
                    position_ids_padded[i, -cur_len:] = torch.arange(0, cur_len, dtype=torch.long, device=device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels_item
                    attention_mask_padded[i, :cur_len] = True
                    position_ids_padded[i, :cur_len] = torch.arange(0, cur_len, dtype=torch.long, device=device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask_padded.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        else:
            position_ids = position_ids_padded

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    # def prepare_inputs_labels_for_multimodal(
    #     self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    # ):
    #     vision_tower = self.get_vision_tower()
    #     # 如果没有图像，或者正在进行自回归生成（input_ids 长度为 1 且有 KV Cache）
    #     if vision_tower is None or images is None or input_ids.shape[1] == 1:
    #         if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
    #             # 计算当前最新的 KV Cache 总长度 + 当前 token
    #             target_shape = past_key_values[-1][-1].shape[-2] + 1
    #             # 修正 Attention Mask，确保当前 token 能看到之前所有的 KV Cache
    #             attention_mask = torch.cat((attention_mask, torch.ones(
    #                 (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
    #                 dtype=attention_mask.dtype,
    #                 device=attention_mask.device
    #             )), dim=1)
    #             # 修正 Position IDs：当前 token 的位置等于 Mask 的总和减 1
    #             position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                
    #         # 【核心修正】：返回值
    #         # 第1位必须返回 input_ids，第5位必须返回 None
    #         return input_ids, position_ids, attention_mask, past_key_values, None, labels

    #     if type(images) is list or images.ndim == 5:
    #         concat_images = torch.cat([image for image in images], dim=0)
    #         image_features = self.encode_images(concat_images)
    #         split_sizes = [image.shape[0] for image in images]
    #         image_features = torch.split(image_features, split_sizes, dim=0)
    #         image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
    #     else:
    #         image_guide_features, image_features = self.encode_images(images)

    #     assert image_features.shape[1] == 576, 'vision tower not a withprojection version.'
    #     text_tower = self.get_text_tower()

    #     # with torch.no_grad():
    #     #     # image_guide_features: bs, 4096
    #     #     image_guide_features = image_features[:,0]
        
    #     input_pad = np.where(input_ids.cpu().detach().numpy()!=-200,input_ids.cpu().detach().numpy(),self.tokenizer.pad_token_id)
    #     decoded_inputs = self.tokenizer.batch_decode(input_pad, skip_special_tokens=True)
    #     decoded_hidden_inputs = ['\n'.join(decode_input.split('\n')[1:]) for decode_input in decoded_inputs]
    #     decoded_clip_inputs = [decode_input.split(' ASSISTANT')[0] for decode_input in decoded_hidden_inputs]

    #     clip_text_inputs = self.clip_tokenizer(
    #             decoded_clip_inputs,
    #             padding="longest",
    #             max_length=77,
    #             truncation=True,
    #             return_tensors="pt",
    #         )

    #     # text_guide_features: bs, 768
    #     text_guide_features = text_tower(clip_text_inputs)

    #     if self.training:

    #         current_image_features = image_guide_features  # [batch_size, feature_dim]
    #         current_text_features = text_guide_features  # [batch_size, feature_dim]
    #         task_id = self.cur_task
    #         self.active_expert_idx = task_id

    #         image_sum = self.image_anchors[task_id] * self.image_boundary[task_id] + current_image_features.sum(dim=0)
    #         text_sum = self.text_anchors[task_id] * self.text_boundary[task_id] + current_text_features.sum(dim=0)

    #         self.image_boundary[task_id].data += current_image_features.shape[0]
    #         self.text_boundary[task_id].data += current_text_features.shape[0]

    #         self.image_anchors[task_id] = image_sum / self.image_boundary[task_id]
    #         self.text_anchors[task_id] = text_sum / self.text_boundary[task_id]
    #     else:
    #         image_sim = []
    #         text_sim = []
    #         for image_anchor in self.image_anchors:
    #             image_sims = F.cosine_similarity(image_guide_features.unsqueeze(1), image_anchor, dim=2)
    #             image_sim.append(image_sims.max().item())
    #         for text_anchor in self.text_anchors:
    #             text_sims = F.cosine_similarity(text_guide_features.unsqueeze(1), text_anchor, dim=2)
    #             text_sim.append(text_sims.max().item())

    #         image_sim = np.array(image_sim[:self.expert_num]) 
    #         text_sim = np.array(text_sim[:self.expert_num])  

    #         sim = (image_sim + text_sim) / 2

    #         sim_tensor = torch.tensor(sim, dtype=torch.float32)

    #         sim_softmax = F.softmax(sim_tensor / 0.1, dim = -1)
    #         self.active_expert_idx = torch.argmax(sim_softmax).item()

    #         # compute_expert_weight = torch.sigmoid(shifted_conf).tolist()
    #         compute_expert_weight = sim_softmax.tolist()
    #         # print(compute_expert_weight)

    #         proj_names = [
    #             'q_proj', 'k_proj', 'v_proj', 'o_proj',  # self_attn 
    #             'gate_proj', 'up_proj', 'down_proj'      # mlp 
    #         ]
    #         for proj_name in proj_names:
    #             if proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
    #                 proj_layer = getattr(self.model.layers[-1].self_attn, proj_name)
    #             else:
    #                 proj_layer = getattr(self.model.layers[-1].mlp, proj_name)

    #             proj_layer.expert_weight = compute_expert_weight
    #             # print(proj_layer.expert_weight)


    #     # TODO: image start / end is not implemented here to support pretraining.
    #     if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #         raise NotImplementedError

    #     # Let's just add dummy tensors if they do not exist,
    #     # it is a headache to deal with None all the time.
    #     # But it is not ideal, and if you have a better idea,
    #     # please open an issue / submit a PR, thanks.
    #     _labels = labels
    #     _position_ids = position_ids
    #     _attention_mask = attention_mask
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    #     else:
    #         attention_mask = attention_mask.bool()
    #     if position_ids is None:
    #         position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    #     if labels is None:
    #         labels = torch.full_like(input_ids, IGNORE_INDEX)

    #     # remove the padding using attention_mask -- TODO: double check
    #     input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    #     labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    #     new_input_embeds = []
    #     new_labels = []
    #     cur_image_idx = 0
    #     for batch_idx, cur_input_ids in enumerate(input_ids):
    #         num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
    #         if num_images == 0:
    #             cur_image_features = image_features[cur_image_idx]
    #             cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
    #             cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
    #             new_input_embeds.append(cur_input_embeds)
    #             new_labels.append(labels[batch_idx])
    #             cur_image_idx += 1
    #             continue

    #         image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
    #         cur_input_ids_noim = []
    #         cur_labels = labels[batch_idx]
    #         cur_labels_noim = []
    #         for i in range(len(image_token_indices) - 1):
    #             cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
    #             cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
    #         split_sizes = [x.shape[0] for x in cur_labels_noim]
    #         cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
    #         cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
    #         cur_new_input_embeds = []
    #         cur_new_labels = []

    #         for i in range(num_images + 1):
    #             cur_new_input_embeds.append(cur_input_embeds_no_im[i])
    #             cur_new_labels.append(cur_labels_noim[i])
    #             if i < num_images:
    #                 cur_image_features = image_features[cur_image_idx]
    #                 cur_image_idx += 1
    #                 cur_new_input_embeds.append(cur_image_features)
    #                 cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            
    #         if hasattr(self, 'task_prompts'):
    #             selected_task_id = self.active_expert_idx
                
    #             # 1. 取出 Prompt 并确保 dtype/device 与当前 embedding 一致
    #             target_prompts = self.task_prompts[selected_task_id]          # [prefix_len, hidden_size]
    #             model_dtype = self.get_model().embed_tokens.weight.dtype     # 关键：与模型当前精度一致
    #             device = cur_labels.device
                
    #             target_prompts = target_prompts.to(dtype=model_dtype, device=device)

    #             # 2. 决定插入位置（重要改进！）
    #             # 当前结构通常是: [system] + <image> + user question + ...
    #             # 我们插入在 system 之后、image/question 之前（语义上最合理）
    #             insert_pos = 1 if len(cur_new_input_embeds) > 0 else 0

    #             cur_new_input_embeds.insert(insert_pos, target_prompts)
    #             cur_new_labels.insert(insert_pos, torch.full((target_prompts.shape[0],), IGNORE_INDEX, device=cur_labels.device))
                

    #         cur_new_input_embeds = torch.cat(cur_new_input_embeds)
    #         cur_new_labels = torch.cat(cur_new_labels)

    #         new_input_embeds.append(cur_new_input_embeds)
    #         new_labels.append(cur_new_labels)

    #     tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    #     if tokenizer_model_max_length is not None:
    #         new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
    #         new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    #     # 2. 准备 Padded 容器
    #     # 此时的 max_len 是包含了 Prompt 长度后的 batch 内最长序列长度
    #     max_len = max(x.shape[0] for x in new_input_embeds)
    #     batch_size = len(new_input_embeds)
    #     device = new_input_embeds[0].device

    #     new_input_embeds_padded = []
    #     new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=device)
        
    #     # 关键修改：手动初始化 mask 和 position_ids
    #     # 这里使用 bool 并在最后转换，或者直接匹配原 mask 的 dtype
    #     attention_mask_padded = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    #     position_ids_padded = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    #     # 3. 循环填充并对齐 Mask/Position
    #     for i, (cur_new_embed, cur_new_labels_item) in enumerate(zip(new_input_embeds, new_labels)):
    #         cur_len = cur_new_embed.shape[0]
            
    #         # --- 分支 A: 左填充 (通常用于推理推理) ---
    #         if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
    #             new_input_embeds_padded.append(torch.cat((
    #                 torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=device),
    #                 cur_new_embed
    #             ), dim=0))
    #             if cur_len > 0:
    #                 new_labels_padded[i, -cur_len:] = cur_new_labels_item
    #                 attention_mask_padded[i, -cur_len:] = True
    #                 # 生成完整的位置 ID：0, 1, 2, ..., (cur_len-1)
    #                 position_ids_padded[i, -cur_len:] = torch.arange(0, cur_len, dtype=torch.long, device=device)
            
    #         # --- 分支 B: 右填充 (通常用于训练) ---
    #         else:
    #             new_input_embeds_padded.append(torch.cat((
    #                 cur_new_embed,
    #                 torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=device)
    #             ), dim=0))
    #             if cur_len > 0:
    #                 new_labels_padded[i, :cur_len] = cur_new_labels_item
    #                 attention_mask_padded[i, :cur_len] = True
    #                 # 生成完整的位置 ID
    #                 position_ids_padded[i, :cur_len] = torch.arange(0, cur_len, dtype=torch.long, device=device)

    #     # 4. 转换并打包返回
    #     new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    #     # 如果原始输入没有提供 labels/mask/ids，则返回 None，由基类后续生成
    #     # 但既然我们改了长度，最好在这里全部返回新生成的
    #     if _labels is None:
    #         new_labels = None
    #     else:
    #         new_labels = new_labels_padded

    #     if _attention_mask is None:
    #         attention_mask = None
    #     else:
    #         # 确保 dtype 与原始 mask 一致（LLaVA 通常需要 float 或 bool）
    #         attention_mask = attention_mask_padded.to(dtype=_attention_mask.dtype)

    #     if _position_ids is None:
    #         position_ids = None
    #     else:
    #         position_ids = position_ids_padded

    #     return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


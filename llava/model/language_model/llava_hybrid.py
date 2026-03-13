import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from transformers import AutoConfig, AutoModelForCausalLM
from .llava_llama import LlavaLlamaForCausalLM, LlavaConfig
from HiDe.peft.tuners.clitmoelora import HiDeMOELoraLinear

class LlavaHybridConfig(LlavaConfig):
    model_type = "llava_hybrid"
    num_tasks = 6 
    prefix_len = 10
    router_dim = 768

class LlavaHybridForCausalLM(LlavaLlamaForCausalLM):
    config_class = LlavaHybridConfig

    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size

        self.cur_task = 0
        self.expert_num = config.num_tasks
        self.num_tasks = config.num_tasks

        embed_dtype = self.get_model().embed_tokens.weight.dtype
        embed_device = self.get_model().embed_tokens.weight.device

        self.task_prompts = nn.Parameter(
            torch.randn(
                config.num_tasks,
                config.prefix_len,
                config.hidden_size,
                dtype=embed_dtype,
                device=embed_device
            ) * 0.02
        )
        # self.register_parameter(
        #     "task_prompts", 
        #     torch.nn.Parameter(torch.randn(config.num_tasks, config.prefix_len, config.hidden_size))
        # )   

        n = config.num_tasks
        self.image_anchors = nn.ParameterList(
            [nn.Parameter(0.1 * torch.randn(1, config.router_dim)) for _ in range(n)]
        )
        self.text_anchors = nn.ParameterList(
            [nn.Parameter(0.1 * torch.randn(1, config.router_dim)) for _ in range(n)]
        )
        self.image_boundary = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(n)]
        )
        self.text_boundary = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(n)]
        )
        # 3. 持久化状态变量
        self.active_expert_idx = 0
        self.expert_weight = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]


    def set_boundary_for_save(self):
        for name, param in self.image_boundary.named_parameters():
            param.requires_grad = True
        
        for name, param in self.text_boundary.named_parameters():
            param.requires_grad = True

        for name, param in self.image_anchors.named_parameters():
            param.requires_grad = True
        
        for name, param in self.text_anchors.named_parameters():
            param.requires_grad = True

    def set_clip_tokenizer(self, tokenizer):
        self.clip_tokenizer = tokenizer

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_cur_task(self, cur_task, expert_num=None):
        """同步任务状态到所有底层 LoRA 专家分支"""
        self.cur_task = cur_task
        self.active_expert_idx = cur_task
        self.expert_num = expert_num

        # 遍历所有子模块，确保底层 LoRA 层切换到正确专家
        for module in self.modules():
            if isinstance(module, HiDeMOELoraLinear) or hasattr(module, 'set_cur_task'):
                if module is not self:
                    module.set_cur_task(cur_task, expert_num)

        # 保证参数在持续学习中是可训练的
        self.task_prompts.requires_grad = True

        for name, param in self.image_anchors.named_parameters():
            param.requires_grad = True
        
        for name, param in self.text_anchors.named_parameters():
            param.requires_grad = True


# 注册到系统
AutoConfig.register("llava_hybrid", LlavaHybridConfig)
AutoModelForCausalLM.register(LlavaHybridConfig, LlavaHybridForCausalLM)
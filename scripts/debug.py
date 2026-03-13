import torch
import os

# --- 请修改为你 Task 0 训练输出的路径 ---
BIN_PATH = "/home/hechen/zms/MLLM_Factory/HiDe-LLaVA/output/ucit/Task1_llava_lora_ours/non_lora_trainables.bin"

if not os.path.exists(BIN_PATH):
    print(f"❌ 找不到文件: {BIN_PATH}")
else:
    # 加载权重字典
    state_dict = torch.load(BIN_PATH, map_location='cpu')
    
    # 查找含有 'task_prompts' 的键名（处理可能的前缀）
    prompt_key = [k for k in state_dict.keys() if 'task_prompts' in k]
    
    if not prompt_key:
        print("❌ 在 bin 文件中未找到 'task_prompts' 键。")
        print(f"当前文件包含的键有: {list(state_dict.keys())}")
    else:
        key = prompt_key[0]
        prompts = state_dict[key] # 形状通常是 [6, 10, hidden_size]
        
        print(f"✅ 成功找到键: {key}")
        print(f"📊 Prompt 矩阵形状: {prompts.shape}")
        
        # 提取 Task 0 的前 5 个数值进行比对
        # 假设维度是 [Task, Prefix_Len, Hidden]
        task0_sample = prompts[0, 0, :5] 
        
        print("\n--- [重点] Task 0 第一个 Token 的前 5 个数值 ---")
        print(task0_sample.tolist())
        print("----------------------------------------------")
        
        # 计算统计量
        print(f"Task 0 标准差 (std): {prompts[0].std().item():.4f}")
        print(f"Task 1 标准差 (std): {prompts[1].std().item():.4f}")
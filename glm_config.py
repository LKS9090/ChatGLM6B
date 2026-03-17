# -*- coding:utf-8 -*-
import torch


class ProjectConfig(object):
    def __init__(self):
        # 定义是否使用 GPU
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # 定义 ChatGLM-6B 模型的路径
        self.pre_model = r'C:\Users\L9090\PycharmProjects\LLM\chatGLM\chatglm-6b'
        # 定义训练数据的路径
        self.train_path = r'C:\Users\L9090\PycharmProjects\LLM\chatGLM\data\mixed_train_dataset.jsonl'
        # 定义验证集的路径
        self.dev_path = r'C:\Users\L9090\PycharmProjects\LLM\chatGLM\data\mixed_dev_dataset.jsonl'
        # 是否使用LoRA方法微调
        self.use_lora = True
        # 是否使用 P-Tuing 方法微调
        self.use_ptuning = False
        # 秩==8
        self.lora_rank = 8
        # 一个批次多少样本
        self.batch_size = 4
        # 训练几轮
        self.epochs = 10  # 增加 epochs 数量，让早停有发挥作用的空间
        # 学习率
        self.learning_rate = 3e-5
        # 权重权重系数
        self.weight_decay = 0
        # 学习率预热比例
        self.warmup_ratio = 0.06
        # 梯度裁剪的最大范数（防止梯度爆炸）
        self.max_grad_norm = 1.0
        # 梯度积累步数（每 accumulation_steps 个 batch 更新一次参数）
        self.gradient_accumulation_steps = 4
        # context 文本的输入长度限制
        self.max_source_seq_len = 100
        # target 文本长度限制
        self.max_target_seq_len = 100
        # 每隔多少步打印日志
        self.logging_steps = 10
        # 每隔多少步保存
        self.save_freq = 200
        # 如果你使用了 P-Tuing，要定义伪 tokens 的长度
        self.pre_seq_len = 200
        self.prefix_projection = False # 默认为 False，即 p-tuning，如果为 True，即 p-tuning-v2
        # 保存模型的路径
        self.save_dir = r'C:\Users\L9090\PycharmProjects\LLM\chatGLM\checkpoints'
        
        # ==================== 早停机制配置 ====================
        self.use_early_stopping = True       # 是否启用早停
        self.patience = 3                    # 容忍度：连续多少个 epoch 不改善就停止
        self.min_delta = 1e-4                # 最小改善阈值（小于这个值认为没有改善）


if __name__ == '__main__':
    pc = ProjectConfig()
    print(pc.save_dir)
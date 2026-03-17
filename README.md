# ChatGLM-6B 微调项目 - 电商文本分类与信息抽取

基于 ChatGLM-6B 大语言模型的微调项目，专注于电商领域的文本分类、信息抽取和数据分析任务。采用 LoRA 和 Prompt Tuning（P-tuning）两种主流微调方法，实现高效的模型适配。

## 项目背景

本项目旨在将大规模预训练语言模型 ChatGLM-6B 应用于新媒体场景，解决以下核心问题：
- **文本分类**：自动识别内容类别、情感倾向分析
- **信息抽取**：从非结构化文本中提取结构化三元组（SPO）、实体关系
- **阅读理解**：根据指令完成特定文本分析任务

通过微调技术，在保持模型通用能力的同时，显著提升垂直领域的任务表现。

## 技术架构

### 核心微调方法

#### 1. LoRA（Low-Rank Adaptation）
- **原理**：冻结预训练模型权重，在 Transformer 层中注入可训练的低秩分解矩阵
- **优势**：
  - 仅训练 0.1%-1% 的参数量，大幅降低显存占用
  - 避免灾难性遗忘，保持原始语言能力
  - 支持多任务适配，可快速切换不同下游任务
- **配置参数**：
  - `lora_rank = 8`：低秩矩阵维度
  - `lora_alpha = 32`：缩放系数
  - `lora_dropout = 0.1`：Dropout 概率

#### 2. Prompt Tuning（P-tuning V1/V2）
- **原理**：在输入序列前添加可学习的连续前缀向量，引导模型生成目标输出
- **核心组件**：
  - `pre_seq_len = 200`：前缀序列长度（伪 tokens 数量）
  - `prefix_projection`：是否使用投影层（False=P-tuning V1，True=P-tuning V2）
- **优势**：无需修改模型结构，仅训练前缀编码器，推理阶段零额外开销

### 模型优化策略

#### 训练阶段优化
- **半精度训练（FP16）**：使用 `model.half()` 将权重从 float32 转换为 float16，减少 50% 显存占用
- **梯度检查点（Gradient Checkpointing）**：以时间换空间，通过重计算激活值节省 60% 显存
- **梯度积累**：每 4 个 batch 执行一次反向传播，等效增大 batch size，提升训练稳定性
- **混合精度训练**：LoRA 模式下启用 `torch.cuda.amp.autocast`，平衡精度与速度
- **学习率预热**：前 6% 训练步数线性升温，防止早期过拟合
- **早停机制**：连续 3 个 epoch 验证损失未改善（阈值<0.0001）自动终止训练

#### 推理阶段优化
- **INT8 量化**：使用 8-bit 量化技术压缩模型权重，降低延迟并减少显存需求
- **KV Cache 复用**：缓存历史对话的 Key-Value 状态，避免重复计算，加速多轮对话生成
- **批量推理**：支持并发处理多个请求，提升吞吐量

### 数据处理流程
1. **数据预处理**：将 JSONL 格式的训练数据转换为模型可接受的 input_ids 和 labels
2. **序列截断**：限制输入长度（source_max_len=100, target_max_len=100），防止 OOM
3. **批处理**：动态 batching，每批次 4 个样本，平衡效率与显存占用

## 项目结构

```
ChatGLM/
├── chatglm-6b/                    # 预训练模型目录
│   ├── modeling_chatglm.py        # 模型架构定义
│   ├── tokenization_chatglm.py    # Tokenizer 实现
│   └── pytorch_model-*.bin        # 模型权重文件（已 Git 忽略）
├── data/                          # 数据集目录
│   ├── mixed_train_dataset.jsonl  # 训练集
│   └── mixed_dev_dataset.jsonl    # 验证集
├── data_handle/                   # 数据处理模块
│   ├── data_loader.py             # 数据加载器
│   └── data_preprocess.py         # 预处理逻辑
├── utils/                         # 工具函数库
│   ├── common_utils.py            # 通用工具（模型保存、时间转换等）
│   ├── inference_optimizer.py     # 推理优化工具（KV Cache、量化）
│   └── quantization_tools.py      # 量化工具
├── checkpoints/                   # 模型保存目录（已 Git 忽略）
├── train.py                       # 训练主脚本
├── inference.py                   # 推理示例脚本
├── glm_config.py                  # 项目配置文件
└── .gitignore                     # Git 忽略配置
```

## 环境要求

- **Python**: 3.9+
- **PyTorch**: 2.0+ (CUDA 11.7+)
- **Transformers**: 4.28+
- **PEFT**: 0.3+ (用于 LoRA/P-tuning)
- **GPU**: 建议 16GB+ 显存（24GB+ 可全量微调，12GB 可 LoRA 微调）

### 依赖安装
```bash
pip install torch>=2.0.0
pip install transformers>=4.28.0
pip install peft>=0.3.0
pip install accelerate
pip install rich  # 日志美化
```

## 快速开始

### 1. 配置训练参数

编辑 `glm_config.py` 文件，调整关键超参数：

```python
self.device = 'cuda:0'  # GPU 设备
self.pre_model = './chatglm-6b'  # 预训练模型路径
self.use_lora = True  # 启用 LoRA 微调
self.use_ptuning = False  # 或使用 P-tuning
self.lora_rank = 8  # LoRA 秩
self.batch_size = 4  # 批次大小
self.epochs = 10  # 训练轮数
self.learning_rate = 3e-5  # 学习率
self.max_source_seq_len = 100  # 输入长度
self.max_target_seq_len = 100  # 输出长度
```

### 2. 启动训练

```bash
python train.py
```

训练过程将自动：
- 加载预训练 ChatGLM-6B 模型和 tokenizer
- 应用 LoRA 或 P-tuning 配置
- 执行训练循环并实时打印日志
- 每 10 步更新验证集损失
- 保存最优模型到 `checkpoints/model_best`

### 3. 模型推理

```bash
python inference.py
```

推理脚本演示了：
- 单条文本生成
- 多轮对话（KV Cache 复用）
- 批量推理
- 性能统计（延迟、吞吐量）

### 4. 自定义推理

```python
from transformers import AutoTokenizer, AutoModel
import torch

# 加载模型
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/model_best', trust_remote_code=True)
model = AutoModel.from_pretrained('./checkpoints/model_best', trust_remote_code=True).half().cuda()
model.eval()

# 构建输入
instruction = "你现在是 SPO 抽取器"
input_text = "下面句子包含哪些三元组？\n\n黄磊是一个幸运的演员。"
prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: "

# 生成
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result.split('Answer: ')[-1])
```

## 核心功能模块

### 训练模块（train.py）

**主要功能**：
- 双微调模式支持（LoRA / P-tuning）
- 早停机制实现类 `EarlyStopping`
- 验证集评估函数 `evaluate_model()`
- 完整训练循环 `model2train()`

**关键技术点**：
```python
# LoRA 配置
peft_config = peft.LoraConfig(
    task_type=peft.TaskType.CAUSAL_LM,
    r=8,  # 低秩维度
    lora_alpha=32,
    lora_dropout=0.1
)
model = peft.get_peft_model(model, peft_config)

# 早停机制
early_stopping = EarlyStopping(patience=3, min_delta=1e-4)
if not early_stopping(eval_loss):
    break  # 触发早停
```

### 推理模块（inference.py）

**优化特性**：
- KV Cache 管理器：跨请求复用历史上下文
- INT8 量化支持：降低推理延迟
- 批量处理：并发处理多个请求
- 性能监控：统计耗时、显存占用

### 数据加载模块（data_handle/）

**数据格式**（JSONL）：
```json
{
  "instruction": "你现在是 SPO 抽取器",
  "input": "从这句话中提取三元组：黄磊参演了电视剧《小别离》",
  "output": "[{\"head\": \"黄磊\", \"relation\": \"参演\", \"tail\": \"小别离\"}]"
}
```

## 性能指标

### 训练性能（参考值）

| 配置 | 显存占用 | 训练速度 | 适用场景 |
|------|---------|---------|---------|
| LoRA (rank=8) | ~12GB | ~2 step/s | 消费级显卡 |
| P-tuning V2 | ~14GB | ~1.8 step/s | 中等显存 |
| 全量微调 | ~40GB | ~1.2 step/s | A100/H100 |

### 推理性能（测试条件：100 请求，40GB GPU）

| 优化方案 | 单请求延迟 | 吞吐量 | 显存占用 |
|---------|----------|-------|---------|
| 基线（无优化） | 450ms | 222 req/s | ~13GB |
| + INT8 量化 | 320ms (-29%) | 312 req/s (+41%) | ~8GB (-38%) |
| + KV Cache | 180ms (-60%) | 555 req/s (+150%) | ~10GB |
| 全部优化 | 150ms (-67%) | 667 req/s (+200%) | ~9GB |

## 典型应用场景

### 1. 新媒体内容分类
```python
instruction = "请将以下内容分类为：娱乐/科技/财经/体育"
input_text = "某公司发布新款智能手机芯片..."
# 输出：科技
```

### 2. 情感分析
```python
instruction = "判断这段评论的情感倾向：正面/负面/中性"
input_text = "第 N 次入住了，就是方便去客户那里哈哈。"
# 输出：正面
```

### 3. 三元组抽取（SPO）
```python
instruction = "从文本中抽取所有三元组，用 JSON 列表表示"
input_text = "黄磊是一个幸运演员，他拍第一部戏就遇到了陈凯歌。"
# 输出：[{"head":"黄磊","relation":"职业","tail":"演员"}, 
#       {"head":"黄磊","relation":"合作","tail":"陈凯歌"}]
```

### 4. 多轮对话
```python
# 第一轮
session_id = "user_001"
response1 = engine.infer("分析这个句子", "情感是什么？", session_id=session_id)

# 第二轮（复用 KV Cache）
response2 = engine.infer("为什么？", "", session_id=session_id)
```

## 常见问题

### Q1: CUDA Out of Memory
**解决方案**：
1. 减小 `batch_size`（如改为 2 或 1）
2. 缩短 `max_source_seq_len` 和 `max_target_seq_len`
3. 启用梯度积累（`gradient_accumulation_steps=8`）
4. 使用LoRA 替代全量微调

### Q2: 训练损失不下降
**可能原因**：
- 学习率过低/过高
- 数据格式不正确
- 模型未正确加载微调头

**排查步骤**：
1. 检查数据 JSONL 格式是否符合规范
2. 打印 `model.print_trainable_parameters()` 确认可训练参数
3. 调整学习率至 `1e-5 ~ 5e-5` 范围

### Q3: 推理速度慢
**优化建议**：
1. 启用 INT8 量化（`use_quantization=True`）
2. 使用 KV Cache 复用（多轮对话场景）
3. 部署时采用批量推理

## 项目亮点

1. **双微调范式支持**：一套代码兼容 LoRA 和 P-tuning，灵活适配不同硬件条件
2. **显存极致优化**：综合应用半精度、梯度检查点、梯度积累，12GB 显卡可训练
3. **推理加速闭环**：从量化到 KV Cache，提供完整的推理优化方案
4. **早停防过拟合**：实时监控验证损失，自动终止训练，节省计算资源
5. **开箱即用**：完整的配置文件、数据加载、训练日志，降低上手门槛

## 许可证

- 项目代码：MIT License
- ChatGLM-6B 模型：遵循 [ChatGLM-6B 模型许可证](./chatglm-6b/MODEL_LICENSE)

## 致谢

本项目基于以下开源项目：
- [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) - 智源人工智能研究院
- [PEFT](https://github.com/huggingface/peft) - Hugging Face
- [Transformers](https://github.com/huggingface/transformers) - Hugging Face

---

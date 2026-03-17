# ======================== 板块 1：导入依赖库 & 初始化配置 ========================
# 导入 os 库，用于文件/路径操作（如保存模型）
import os
# 导入 time 库，用于计算训练耗时、速度
import time
# 导入 partial，用于固定函数参数（本代码暂未直接使用，为通用工具）
from functools import partial
# peft 第三方库：含有 P-Tuning、LoRA 等微调方法
import peft
# autocast 是 PyTorch 中一种混合精度的技术，可在保持数值精度的情况下提高训练速度和减少显存占用。
# 该方法混合精度训练，如果在 CPU 环境中不起任何作用
from torch.cuda.amp import autocast as autocast
# 从 transformers 导入：
# AutoTokenizer（加载预训练 tokenizer）、AutoConfig（加载模型配置）、AutoModel（加载预训练模型）、get_scheduler（学习率调度器）
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_scheduler
# 导入自定义通用工具函数（如保存模型、时间转换等）
from utils.common_utils import *
# 导入自定义数据加载函数（get_data 用于加载训练/验证 dataloader）
from data_handle.data_loader import *
# 导入项目配置（如模型路径、微调方式、学习率、训练轮数等）
from glm_config import *
# 如果损失出现为 None，备用导入 AdamW 优化器


# 实例化项目配置类，加载所有训练超参数（如 pre_model 路径、use_lora、epochs 等）
pc = ProjectConfig()

"""
这段代码是 ChatGLM-6B 模型的微调训练核心脚本，专门适配新媒体信息抽取 / 文本分类任务，核心能力包括：
1. 支持两种主流微调方式：P-tuning（V1/V2）和 LoRA（低秩适配）；
2. 集成模型优化策略：半精度训练、梯度检查点、混合精度（autocast）、权重衰减分组、学习率预热调度；
3. 完整的训练闭环：训练循环（梯度反向传播）、实时日志打印、验证集评估、最优模型保存；
4. 显存优化：关闭缓存、梯度检查点、半精度转换等，适配低配 GPU 训练。
5. 早停机制：当验证集损失连续 patience 个 epoch 不改善时，自动终止训练，防止过拟合。
"""

# ======================== 板块 2：定义早停机制类 ========================
class EarlyStopping:
    """早停机制实现类 - 防止过拟合"""
    
    def __init__(self, patience=3, min_delta=1e-4, verbose=False):
        """
        Args:
            patience: 容忍度 - 连续多少个 epoch 不改善就停止
            min_delta: 最小改善阈值 - 小于这个值认为没有改善
            verbose: 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0  # 记录连续未改善的 epoch 数
        self.best_loss = None  # 最佳验证损失
        self.early_stop = False  # 是否触发早停
        
    def __call__(self, eval_loss):
        """
        检查是否满足早停条件
        
        Args:
            eval_loss: 当前验证集损失
            
        Returns:
            bool: 是否应该继续训练
        """
        # 如果是第一个 epoch，初始化 best_loss
        if self.best_loss is None:
            self.best_loss = eval_loss
            if self.verbose:
                print(f"[EarlyStopping] 初始最佳损失：{self.best_loss:.5f}")
            return True
        
        # 检查是否有改善
        if eval_loss < self.best_loss - self.min_delta:
            # 有改善，更新 best_loss，重置计数器
            improvement = self.best_loss - eval_loss
            self.best_loss = eval_loss
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] 损失改善：{improvement:.5f}, 当前最佳：{self.best_loss:.5f}")
            return True
        else:
            # 没有改善，计数器 +1
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] 损失未改善 ({self.counter}/{self.patience}), "
                      f"当前：{eval_loss:.5f}, 最佳：{self.best_loss:.5f}")
            
            # 检查是否达到容忍度
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"\n{'='*80}")
                    print(f"[EarlyStopping] 触发早停！连续 {self.patience} 个 epoch 未改善")
                    print(f"最佳验证损失：{self.best_loss:.5f}")
                    print(f"{'='*80}\n")
                self.early_stop = True
                return False
            
            return True


# ======================== 板块 2：定义模型评估函数（验证集） ========================
def evaluate_model(model, dev_dataloader):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        data_loader: 测试集的dataloader
    """
    # 将模型切换为评估模式（禁用dropout、batchnorm等训练特有的层）
    model.eval()
    # 初始化列表，存储每个batch的损失值
    loss_list = []
    # 禁用梯度计算（评估阶段不需要反向传播，节省显存）
    with torch.no_grad():
        # 遍历验证集的每个batch
        for batch in dev_dataloader:
            # 如果使用LoRA微调，启用混合精度训练
            if pc.use_lora:
                with autocast():
                    # 前向传播计算损失：
                    # input_ids/labels转换为long类型并移到指定设备（GPU/CPU）
                    loss = model(
                        input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                        labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                    ).loss
            # 不使用LoRA时，直接计算损失
            else:
                loss = model(
                    input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                    labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                ).loss
            # 将损失值从GPU移到CPU，解除梯度关联，转换为float并加入列表
            loss_list.append(float(loss.cpu().detach()))
    # 评估完成后，切换回训练模式（启用dropout等）
    model.train()
    # 返回验证集平均损失（所有batch损失的均值）
    return sum(loss_list) / len(loss_list)

# ======================== 板块 3：定义核心训练函数 ========================
def model2train():
    # --------------------- 子板块 3.1：加载 tokenizer 和模型配置 ---------------------
    # 加载 ChatGLM 预训练 tokenizer，trust_remote_code=True 适配自定义模型代码
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True)

    # 加载 ChatGLM 预训练模型的配置文件（如网络结构、超参数等）
    config = AutoConfig.from_pretrained(pc.pre_model, trust_remote_code=True)
    # 打印配置信息（调试用）
    print(f'config-->{config}')

    # 如果使用ptuning微调手段
    if pc.use_ptuning:
        # 设置P-tuning的前缀序列长度（prefix sequence length）
        config.pre_seq_len = pc.pre_seq_len
        # 可以指定是P-tuning-V1或者V2（prefix_projection=True对应V2）
        config.prefix_projection = pc.prefix_projection

    # --------------------- 子板块 3.2：加载并配置 ChatGLM 预训练模型 ---------------------
    # 加载 ChatGLM-6B 预训练模型
    model = AutoModel.from_pretrained(pc.pre_model,
                                      config=config,  # 传入自定义配置（如P-tuning参数）
                                      trust_remote_code=True)  # 信任远程自定义代码

    # #model.half()将模型数据类型从默认的float32精度转换为更低的float16精度，减少内存
    print(f'model-->{model}')
    for name, prameters in model.named_parameters():
        print(f'prameters类型--》{prameters.dtype}')
        print("*"*80)
    # 将模型权重从float32转换为float16（半精度），大幅减少显存占用
    model = model.half().to(pc.device)
    # print("隔开")
    # for name, prameters in model.named_parameters():
    #     print(f'prameters类型--》{prameters.dtype}')
    #     print("*"*80)
    # #在趋动云运行的时候（kill），改一下下面代码：
    # model = model.half()
    # 打印模型结构（调试用）
    # print(model)
    # 梯度检查点 = 用 “时间换显存” 它是专门给训练 / 反向传播用的优化
    # 保存部分激活值，未保存的反向传播时重新计算
    # 启用梯度检查点（Gradient Checkpointing），牺牲少量速度换显存
    model.gradient_checkpointing_enable()
    # 启用输入梯度计算（适配P-tuning/LoRA的梯度传播）
    model.enable_input_require_grads()
    # 不进行缓存（禁用key/value缓存），减少显存占用（训练阶段不需要生成缓存）
    # KV Cache 是推理阶段的优化：生成文本时，缓存前面 token 的 key/value 值，避免重复计算，加速逐 token 生成
    model.config.use_cache = False
    # print(f'model.transformer.prefix_encoder-->{model.transformer.prefix_encoder}')
    # 如果使用P-tuning，将前缀编码器（prefix encoder）转换为float32，避免精度不足
    if pc.use_ptuning:
        model.transformer.prefix_encoder.float()

    # 如果使用lora方法微调
    if pc.use_lora:
        # 将模型的输出头（lm_head）包装为CastOutputToFloat，确保输出为float32（避免精度问题）
        model.lm_head = CastOutputToFloat(model.lm_head)
        # 定义LoRA配置
        peft_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM, # 任务类型：自回归语言模型（ChatGLM是因果语言模型）
            inference_mode=False, # 训练模式（推理时设为True，禁用dropout）
            r=pc.lora_rank, # 低秩矩阵的维度（LoRA的核心参数，越小显存占用越少）
            lora_alpha=32, # 缩放系数（平衡LoRA更新的幅度）
            lora_dropout=0.1, # Dropout概率，防止过拟合
        )
        print(f'peft_config--》{peft_config}')
        # 将基础模型封装为LoRA模型（仅训练LoRA的低秩矩阵，冻结原模型权重）
        model = peft.get_peft_model(model, peft_config)

    print('*'*80)
    print(f'model2-->{model}')
    # 将模型移到指定设备（GPU/CPU，从配置中读取）
    model = model.half().to(pc.device)
    # 打印模型中可训练的参数信息（如数量、占比），验证P-tuning/LoRA是否只训练部分参数
    model.print_trainable_parameters()


    # --------------------- 子板块 3.3：配置优化器（带权重衰减分组） ---------------------
    # 定义不需要权重衰减的参数类型：bias（偏置）、LayerNorm.weight（层归一化权重）
    no_decay = ["bias", "LayerNorm.weight"]
    # 分组设置优化器参数（不同组用不同的权重衰减）
    optimizer_grouped_parameters = [
        {
            # 第一组：需要权重衰减的参数（排除bias和LayerNorm.weight）
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": pc.weight_decay, # 权重衰减系数（从配置读取）
        },
        {
            # 第二组：不需要权重衰减的参数（包含bias或LayerNorm.weight）
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, # 禁用权重衰减
        },
    ]

    # 使用AdamW优化器（带权重衰减的Adam），传入分组参数和学习率（从配置读取）
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=pc.learning_rate)
    # 如果你在训练的时候，出现loss为None，这个时候，换一下AdamW的来源
    # optimizer = AdamW(optimizer_grouped_parameters, lr=pc.learning_rate)


    # --------------------- 子板块 3.4：加载训练/验证数据 ---------------------
    # 调用自定义数据加载函数，获取训练集和验证集的dataloader（批量加载数据）
    train_dataloader, dev_dataloader = get_data()


    # --------------------- 子板块 3.5：配置学习率调度器 ---------------------
    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader) # 每个epoch的步数（batch数）
    #指定总的训练步数，它会被学习率调度器用来确定学习率的变化规律，确保学习率在整个训练过程中得以合理地调节
    max_train_steps = pc.epochs * num_update_steps_per_epoch # 总步数 = 轮数 * 每轮步数
    warm_steps = int(pc.warmup_ratio * max_train_steps) # 预热阶段的训练步数（按比例计算）
    # 定义学习率调度器：linear策略（预热阶段lr线性上升，之后线性下降）
    lr_scheduler = get_scheduler(
        name='linear', # 调度器类型：线性
        optimizer=optimizer, # 关联的优化器
        num_warmup_steps=warm_steps, # 预热步数（前warm_steps步lr从0升到设定值）
        num_training_steps=max_train_steps, # 总训练步数
    )

    # # --------------------- 子板块 3.6：训练循环 ---------------------
    # 定义训练的一些参数变量
    loss_list = [] # 存储训练过程中的损失值（用于计算平均损失）
    tic_train = time.time() # 记录训练开始时间（用于计算速度）
    global_step, best_eval_loss = 0, float('inf') # 全局步数初始化；最佳验证损失初始化为无穷大
    
    # 初始化早停机制（如果配置启用）
    early_stopping = None
    if pc.use_early_stopping:
        early_stopping = EarlyStopping(
            patience=pc.patience,
            min_delta=pc.min_delta,
            verbose=True
        )
        print(f"\n{'='*80}")
        print(f"[早停机制] 已启用")
        print(f"  - 容忍度 (patience): {pc.patience}")
        print(f"  - 最小改善阈值 (min_delta): {pc.min_delta}")
        print(f"{'='*80}\n")
    
    # 遍历每个训练轮次（从 1 到设定的 epochs）
    for epoch in range(1, pc.epochs + 1):
        print("开始训练")
        # 遍历训练集的每个 batch
        for batch in train_dataloader:
            # 如果使用LoRA，启用混合精度训练（减少显存，加速训练）
            if pc.use_lora:
                # torch.cuda.amp.autocast 是 PyTorch 中一种混合精度的技术（仅在 GPU 上训练时可使用）
                with autocast():
                    # torch.cuda.amp.autocast 是 PyTorch 中一种混合精度的技术（仅在 GPU 上训练时可使用）
                    # 模型前向传播，计算损失
                    loss = model.forward(
                        input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device), # input_ids 移到指定设备，类型为 long
                        labels=batch['labels'].to(dtype=torch.long, device=pc.device) # labels 移到指定设备，类型为 long
                    ).loss / pc.gradient_accumulation_steps
            # 不使用LoRA 时，直接前向传播计算损失
            else:
                loss = model(
                    input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                    labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                ).loss / pc.gradient_accumulation_steps

            # 缩放后的损失反向传播（梯度积累模式下，loss 已除以 accumulation_steps）
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=pc.max_grad_norm)
            
            # 判断是否进行优化器更新（梯度积累）
            if (global_step + 1) % pc.gradient_accumulation_steps == 0:
                # 梯度更新，更新模型参数
                optimizer.step()
                # 梯度清零
                optimizer.zero_grad()
                # 学习率调度器步进，更新学习率
                lr_scheduler.step()

            # 将当前 batch 的损失值存入列表（用于计算平均损失）
            loss_list.append(float(loss.cpu().detach()))

            # 全局步数 +1
            global_step += 1
            # 每 logging_steps 步打印一次训练日志
            if global_step % pc.logging_steps == 0:
                # 计算耗时（当前批次的训练时间）
                time_diff = time.time() - tic_train
                # 计算平均损失
                loss_avg = sum(loss_list) / len(loss_list)
                # 打印训练日志：全局步数、进度、epoch、损失、训练速度、预计剩余时间
                print("global step %d ( %02.2f%% ) , epoch: %d, loss: %.5f, speed: %.2f step/s, ETA: %s"
                      % (
                          global_step, # 当前全局步数
                          global_step / max_train_steps * 100, # 训练进度（百分比）
                          epoch, # 当前 epoch
                          loss_avg, # 平均损失
                          pc.logging_steps / time_diff, # 训练速度（步/秒）
                          second2time(int(max_train_steps - global_step) / (pc.logging_steps / time_diff)) # 预计剩余时间
                      ))
                # 重置计时起点
                tic_train = time.time()

            # 每 save_freq 步进行一次验证和模型保存
            # if global_step % pc.save_freq == 0:
            #     cur_save_dir = os.path.join(pc.save_dir, "model_%d" % global_step)
            #     save_model(model, cur_save_dir)
            #     tokenizer.save_pretrained(cur_save_dir)
            #     print(f'Model has saved at {cur_save_dir}.')

                # 在验证集上评估当前模型
                eval_loss = evaluate_model(model, dev_dataloader)

                # 打印验证损失
                print("Evaluation Loss: %.5f" % (eval_loss))
                
                # 检查早停条件（如果启用了早停机制）
                if pc.use_early_stopping and early_stopping:
                    should_continue = early_stopping(eval_loss)
                    
                    # 如果触发早停，终止训练
                    if not should_continue:
                        print(f"\n[早停] Epoch {epoch}: 训练提前终止")
                        
                        # 保存最终模型
                        cur_save_dir = os.path.join(pc.save_dir, "model_early_stop")
                        save_model(model, cur_save_dir)
                        tokenizer.save_pretrained(cur_save_dir)
                        print(f'早停模型已保存至：{cur_save_dir}')
                        
                        # 跳出训练循环
                        break
                
                # 如果当前验证损失低于最佳损失，更新最佳损失并保存模型
                if eval_loss < best_eval_loss:
                    print(
                        f"Min eval loss has been updated: {best_eval_loss:.5f} --> {eval_loss:.5f}"
                    )
                    best_eval_loss = eval_loss # 更新最佳验证损失
                    # 定义最佳模型保存路径
                    cur_save_dir = os.path.join(pc.save_dir, "model_best")
                    # 保存模型（自定义函数）
                    save_model(model, cur_save_dir)
                    # 保存 tokenizer 到同一目录
                    tokenizer.save_pretrained(cur_save_dir)
                    # 打印保存信息
                    print(f'Best model has saved at {cur_save_dir}.')
                # 重置计时起点
                tic_train = time.time()
        
        # 检查是否需要提前结束整个训练（早停触发）
        if pc.use_early_stopping and early_stopping and early_stopping.early_stop:
            break

# ======================== 板块 4：主函数入口 ========================
if __name__ == '__main__':
    # 执行训练函数
    model2train()
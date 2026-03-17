# ======================== 板块1：导入依赖库 & 初始化配置 ========================
# 导入PyTorch核心库，用于构建神经网络、张量操作
import torch
# 导入PyTorch的神经网络模块，用于定义网络层/类
import torch.nn as nn
# 导入项目配置文件（包含use_lora等微调参数）
from glm_config import *
# 导入copy库，用于深拷贝模型（LoRA合并权重时避免修改原模型）
import copy
# 实例化项目配置类，加载use_lora等超参数
pc = ProjectConfig()


"""
这段代码是 ChatGLM-6B 微调项目的通用工具脚本，核心实现三类辅助功能，服务于模型训练全流程：
1. CastOutputToFloat 类：解决 LoRA 微调时模型输出精度溢出问题，强制将输出转为 float32；
2. second2time 函数：将秒数转换为 HH:MM:SS 格式的字符串，用于训练日志中显示剩余时间（ETA）；
3. save_model 函数：根据是否使用 LoRA 差异化保存模型（LoRA 需合并权重后保存，避免只存适配器参数）；
4. 主函数：测试 second2time 函数的转换效果。

"""
# ======================== 板块2：定义精度转换类（解决LoRA精度问题） ========================
# 继承nn.Sequential（序列容器），用于包装模型输出层，强制转换输出精度为float32
class CastOutputToFloat(nn.Sequential):
    # 重写forward方法（前向传播逻辑）
    def forward(self, x):
        # super().forward(x)：执行父类nn.Sequential的前向传播（即原输出层的计算）
        # .to(torch.float32)：将输出张量强制转换为float32精度，避免LoRA微调时float16精度溢出
        return super().forward(x).to(torch.float32)

# ======================== 板块3：定义时间转换工具函数（训练日志用） ========================
def second2time(seconds: int):
    """
    将秒转换成时分秒。

    Args:
        seconds (int): _description_
    """
    # divmod(a, b)：返回(a//b, a%b)，即商和余数
    # 第一步：将总秒数转换为 分钟(m) + 剩余秒数(s)
    m, s = divmod(seconds, 60)
    # 打印分钟数（调试用）
    print(f'm--》{m}')
    # 打印剩余秒数（调试用）
    print(f's--》{s}')
    # 第二步：将分钟数转换为 小时(h) + 剩余分钟数(m)
    h, m = divmod(m, 60)
    # 打印小时数（调试用）
    print(f'h--》{h}')
    # 打印剩余分钟数（调试用）
    print(f'm--》{m}')
    # 格式化输出：HH:MM:SS（不足两位补0），比如3661秒→01:01:01
    return "%02d:%02d:%02d" % (h, m, s)

# ======================== 板块4：定义模型保存工具函数（适配LoRA/非LoRA） ========================
def save_model(
        model,
        cur_save_dir: str
    ):
    """
    存储当前模型

    Args:
        cur_save_path (str): 存储路径。
    """
    # 如果使用LoRA微调（仅训练低秩适配器，原模型权重冻结）
    if pc.use_lora:                       # merge lora params with origin model
        # 深拷贝原模型（避免合并权重时修改训练中的模型）
        merged_model = copy.deepcopy(model)
        # 如果直接保存，只保存的是adapter也就是lora模型的参数
        # 合并LoRA适配器参数和原模型权重，卸载LoRA封装（转为普通ChatGLM模型）
        merged_model = merged_model.merge_and_unload()
        # 保存合并后的完整模型到指定路径
        merged_model.save_pretrained(cur_save_dir)
    # 不使用LoRA时（如P-tuning/全量微调）
    else:
        # 直接保存模型（无需合并，原模型权重已更新）
        model.save_pretrained(cur_save_dir)

# ======================== 板块5：主函数（测试时间转换函数） ========================
if __name__ == '__main__':
    # 测试second2time函数：转换3661秒（1小时1分1秒）
    result = second2time(3661)
    # 打印转换结果（预期输出：01:01:01）
    print(result)
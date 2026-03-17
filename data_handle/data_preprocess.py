# ======================== 板块1：导入必要的库和模块 ========================
# 导入json库，用于解析JSON格式的样本数据
import json
# 返回的字符串包含有关异常的详细信息
# 导入traceback库，用于捕获并打印异常的详细堆栈信息，方便调试
import traceback
# 导入numpy库，用于将处理后的列表转换为数组（适配模型输入格式）
import numpy as np
# tqdm进度条
# 导入tqdm库，用于显示循环的进度条，提升可视化体验
from tqdm import tqdm
# 从datasets库导入load_dataset函数，用于加载文本数据集（HuggingFace datasets格式）
from datasets import load_dataset
# 从transformers库导入AutoTokenizer，用于加载预训练tokenizer（适配ChatGLM）
from transformers import AutoTokenizer
# 从functools导入partial，用于固定函数参数（本代码暂未使用，但为常用工具）
from functools import partial

# 从自定义的glm_config模块导入所有配置（如模型路径、数据集路径等）
from glm_config import *

"""
这段代码是面向ChatGLM 模型的新媒体信息抽取 / 文本分类任务的数据集预处理脚本，核心实现两大功能：
1.convert_example_chatglm 函数：将原始 JSON 格式的训练样本（包含 context 输入文本和 target 标注文本）转换为 ChatGLM 模型训练所需的输入格式，包括 token 编码、长度截断、特殊 token（gMASK、<sop>/<eop>）拼接、padding 填充、labels 掩码（非目标部分置为 - 100）等核心逻辑；
2.get_max_length 函数：统计数据集里输入（context）和输出（target）文本的 token 长度分布（最大值、平均值、中位数），为设置合理的max_source_seq_len（输入最大长度）和max_target_seq_len（输出最大长度）提供数据依据；
3.主函数入口：加载配置、数据集和 tokenizer，支持测试长度统计函数或样本转换函数。
"""

# ======================== 板块2：定义ChatGLM样本转换核心函数 ========================
def convert_example_chatglm(
        examples: dict,
        tokenizer,
        max_source_seq_len: int,
        max_target_seq_len: int,
    ):
    """
    将样本数据转换为Prompt-tuning模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '{"context": "年基准利率4.35%。从实际看...", "target": "2017年银行贷款基准利率"}',
                                                            ...
                                                ]
                                            }
        max_source_seq_len (int): prompt最大长度
        max_target_seq_len (int): 答案最大长度

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[1525, 10, ...], [758, 2345, ...]],
                            'labels': [[822, 10, ...], [125, 58...]]
                        }
    """
    # 初始化输出字典，存储处理后的input_ids和labels
    tokenized_output = {
        'input_ids': [],  # 模型输入的token id列表
        'labels': []      # 模型训练的标签（非目标部分置为-100）
    }
    # 设定最大句子长度 = 输入最大长度 + 输出最大长度（整体不超过该值）
    max_seq_length = max_source_seq_len + max_target_seq_len

    # 遍历每个样本（examples['text']是所有样本的列表）
    for example in examples['text']:
        try:
            # print(f'example-->{example}')
            # print(f'example2type-->{type(example)}')
            # 将字符串格式的JSON样本解析为字典（提取context和target）
            example = json.loads(example)
            # print(f'example==>{type(example)}')
            # print(f'example==>{example}')
            # 提取输入文本（context）
            context = example["context"]
            # 提取标注文本（target）
            target = example["target"]

            # 对prompt输入进行编码
            # 将context文本转换为token id，不添加特殊token（后续手动拼接）
            prompts_ids = tokenizer.encode(
                text=context,
                add_special_tokens=False
            )
            # print(f'prompts_ids--》{prompts_ids}\n{len(prompts_ids)}')
            # 对target目标文本输入进行编码
            # 将target文本转换为token id，不添加特殊token（后续手动拼接）
            target_ids = tokenizer.encode(
                text=target,
                add_special_tokens=False
            )
            # print(f'target_ids--》{target_ids}\n{len(target_ids)}')
            # print("*" * 80)

            # print('37010-->', tokenizer.convert_ids_to_tokens([37010, 12, 5, 76331, 83362]))

            # 对prompt输入如果超过设置的最大长度截断
            # source需要留1个位置给[gMASK] token，因此截断到max_source_seq_len - 1
            if len(prompts_ids) >= max_source_seq_len:
                prompts_ids = prompts_ids[:max_source_seq_len - 1]
            # 对target如果超过设置的最大长度截断
            # target需要留1个位置给<sop>（开头）和1个位置给<eop>（结尾），因此截断到max_target_seq_len - 2
            if len(target_ids) >= max_target_seq_len-1:
                target_ids = target_ids[:max_target_seq_len - 2]
            # print(f'new_prompts_ids--》{prompts_ids}\n{len(prompts_ids)}')
            # print(f'new_target_ids--》{target_ids}\n{len(target_ids)}')
            # print('*'*80)
            # 将输入和target合并作为ChatGLM-6B模型的输入（结合glm架构图去理解）
            # 拼接特殊token：source_ids + [gMASK] + <sop> + target_ids + <eop>
            input_ids = tokenizer.build_inputs_with_special_tokens(prompts_ids, target_ids)
            # print(f'input_ids-->{input_ids}')
            # print(f'input_ids-->{len(input_ids)}')
            # 查找bos_token_id的位置（<sop>是ChatGLM的bos_token，对应target的起始位置）
            context_length = input_ids.index(tokenizer.bos_token_id)
            # print(f'context_length-->{context_length}')
            # [gMASK]在source的最后一位（bos_token前一位）
            mask_position = context_length - 1
            # 构建labels：source部分（到gMASK为止）置为-100（模型不计算这部分损失），target部分保留真实id
            labels = [-100] * context_length + input_ids[mask_position + 1:]
            # 计算需要padding的长度（补齐到max_seq_length）
            pad_len = max_seq_length - len(input_ids)
            # print(f'pad_len-->{pad_len}')

            # 对input_ids进行padding填充（不足max_seq_length的部分补pad_token_id）
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            # print(f'input_ids-->{input_ids}\n{len(input_ids)}')
            # 对labels进行padding填充（不足部分补-100，不计算损失）
            labels = labels + [-100] * pad_len
            # print(f'labels-->{labels}\n{len(labels)}')

            # 将处理后的input_ids和labels添加到输出字典
            tokenized_output['input_ids'].append(input_ids)
            tokenized_output['labels'].append(labels)

        # 捕获所有异常，打印异常样本和详细堆栈信息，避免程序中断
        except:
            print(f'"{example}" -> {traceback.format_exc()}')
            continue

    # 将输出字典中的列表转换为numpy数组（适配模型训练的输入格式）
    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    # 返回处理后的数据集（input_ids和labels均为二维numpy数组）
    return tokenized_output


# ======================== 板块3：定义数据集token长度统计函数 ========================
def get_max_length(
        tokenizer,
        dataset_file: str
):
    """
    测试数据集最大的输入/输出tokens是多少。

    Args:
        tokenizer: ChatGLM的tokenizer实例
        dataset_file (str): 数据集文件路径
    """
    # 初始化列表，存储每个样本的source（context）token长度
    source_seq_len_list = []
    # 初始化列表，存储每个样本的target token长度
    target_seq_len_list = []
    # 关键修改：打开文件时显式指定编码为utf-8（解决Windows默认GBK编码问题）
    # 如果文件有BOM头（比如记事本保存的UTF-8文件），改用 encoding='utf-8-sig'
    with open(dataset_file, 'r', encoding='utf-8') as f:
        # 遍历文件所有行，tqdm显示进度条
        for line in tqdm(f.readlines()):
            try:  # 新增异常捕获，避免单条数据错误中断整体统计
                # 将每行字符串解析为JSON字典
                line = json.loads(line)
                # 对context进行编码，得到token id列表（不添加特殊token，和主函数逻辑一致）
                source_len = tokenizer.encode(line['context'], add_special_tokens=False)
                # 记录该样本context的token长度
                source_seq_len_list.append(len(source_len))

                # 对target进行编码，得到token id列表（不添加特殊token）
                target_len = tokenizer.encode(line['target'], add_special_tokens=False)
                # 记录该样本target的token长度
                target_seq_len_list.append(len(target_len))

            except Exception as e:
                print(f"处理行数据出错：{line[:100]} -> {str(e)}")
                continue

    # 增加空列表判断，避免统计时报错
    if source_seq_len_list:
        print(
            f"【Source Sequence】 Max: {max(source_seq_len_list)}, Avg: {int(sum(source_seq_len_list) / len(source_seq_len_list))}, Middle: {sorted(source_seq_len_list)[int(len(source_seq_len_list) / 2)]}.")
    else:
        print("【Source Sequence】 无有效数据")

    if target_seq_len_list:
        print(
            f"【Target Sequence】 Max: {max(target_seq_len_list)}, Avg: {int(sum(target_seq_len_list) / len(target_seq_len_list))}, Middle: {sorted(target_seq_len_list)[int(len(target_seq_len_list) / 2)]}.")
    else:
        print("【Target Sequence】 无有效数据")


# ======================== 板块4：主函数（测试/运行入口） ========================
if __name__ == '__main__':
    # 实例化项目配置类（从glm_config导入，包含模型路径、数据集路径等）
    pc = ProjectConfig()
    # 加载文本数据集（HuggingFace datasets格式），train路径从配置中读取
    train_dataset = load_dataset('text', data_files={'train': pc.train_path})
    # print(type(train_dataset))
    # print(train_dataset)
    # print('*'*80)
    # print(train_dataset['train'])
    # print('*'*80)
    # print(train_dataset['train']["text"])
    # 加载预训练tokenizer（从配置的预训练模型路径读取，trust_remote_code=True适配ChatGLM自定义代码）
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True)
    # 以下为测试convert_example_chatglm函数的代码（注释状态）
    # tokenized_output = convert_example_chatglm(examples=train_dataset['train'],
    #                                    tokenizer=tokenizer,
    #                                    max_source_seq_len=50,
    #                                    max_target_seq_len=50)
    # print(len(tokenized_output["input_ids"][2]))
    # print(len(tokenized_output["labels"][2]))

    # 调用长度统计函数，输出数据集token长度分布（为设置max_source/target_seq_len提供依据）
    get_max_length(tokenizer, pc.train_path)
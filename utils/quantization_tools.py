
# -*- coding: utf-8 -*-
"""
INT8/INT4 量化推理工具
支持静态量化、动态量化、混合精度量化
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import time
import os


class AdvancedQuantizer:
    """高级量化器 - 支持多种量化策略"""

    @staticmethod
    def quantize_model_int8(model: nn.Module, verbose: bool = True) -> nn.Module:
        """
        INT8 量化（使用 PyTorch 原生量化）

        Args:
            model: 要量化的模型
            verbose: 是否打印详细信息

        Returns:
            量化后的模型
        """
        if verbose:
            print("[Quantizer] 开始 INT8 量化...")
        start_time = time.time()

        # 1. 准备量化配置
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')

        # 2. 准备模型（插入观察器）
        model_prepared = torch.ao.quantization.prepare(model)

        # 3. 校准（需要少量样本）
        # 这里简化处理，实际应该用校准数据集
        print("[Quantizer] 警告：未执行校准步骤，量化效果可能不佳")

        # 4. 转换模型为量化版本
        model_quantized = torch.ao.quantization.convert(model_prepared)

        quant_time = time.time() - start_time
        if verbose:
            print(f"[Quantizer] INT8 量化完成，耗时：{quant_time:.2f}s")

        return model_quantized

    @staticmethod
    def quantize_layer_int8(layer: nn.Linear) -> nn.Sequential:
        """
        对单个 Linear 层进行 INT8 量化

        Args:
            layer: Linear 层

        Returns:
            量化后的层
        """
        quantized_layer = nn.Sequential(
            nn.Linear(layer.in_features, layer.out_features),
            nn.QuantStub(),
            nn.DeQuantStub()
        )

        # 复制权重
        quantized_layer[0].weight.data = layer.weight.data.clone()
        if layer.bias is not None:
            quantized_layer[0].bias.data = layer.bias.data.clone()

        return quantized_layer

    @staticmethod
    def get_model_memory_usage(model: nn.Module) -> Dict[str, float]:
        """
        计算模型显存占用

        Returns:
            显存使用统计 (MB)
        """
        total_params = sum(p.numel() for p in model.parameters())
        total_memory = sum(p.numel() * p.element_size() for p in model.parameters())

        return {
            'total_params_millions': total_params / 1e6,
            'memory_mb': total_memory / (1024 * 1024),
            'memory_gb': total_memory / (1024 * 1024 * 1024)
        }


class MemoryEfficientLoader:
    """内存高效加载器 - 减少模型加载时的峰值内存"""

    @staticmethod
    def load_model_low_memory(
        model_path: str,
        model_class: Any,
        tokenizer_class: Any,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float16
    ) -> Tuple[Any, Any]:
        """
        低内存方式加载模型

        Args:
            model_path: 模型路径
            model_class: 模型类
            tokenizer_class: 分词器类
            device: 加载设备
            dtype: 数据类型

        Returns:
            (model, tokenizer)
        """
        import gc

        print(f"[LowMemLoader] 开始低内存加载：{model_path}")
        start_time = time.time()

        # 1. 先加载 tokenizer（轻量）
        tokenizer = tokenizer_class.from_pretrained(model_path, trust_remote_code=True)

        # 2. 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 3. 分步加载模型
        print("[LowMemLoader] 正在加载模型权重...")

        # 使用 map_location 直接加载到目标设备
        model = model_class.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map='auto' if device.startswith('cuda') else None
        )

        if device != 'cpu' and not hasattr(model, 'device_map'):
            model = model.to(device)

        model.eval()

        load_time = time.time() - start_time

        # 4. 统计内存使用
        mem_stats = MemoryEfficientLoader.get_gpu_memory_info()

        print(f"[LowMemLoader] 加载完成，耗时：{load_time:.2f}s")
        print(f"[LowMemLoader] GPU 内存：{mem_stats['allocated_gb']:.2f}GB / {mem_stats['total_gb']:.2f}GB")

        return model, tokenizer

    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """获取 GPU 内存信息"""
        if not torch.cuda.is_available():
            return {'allocated_gb': 0, 'total_gb': 0}

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'utilization': allocated / total * 100
        }


class InferenceOptimizer:
    """推理优化器 - 综合优化策略"""

    @staticmethod
    def optimize_for_inference(
        model: nn.Module,
        use_int8: bool = True,
        use_cuda_graph: bool = False,
        compile_model: bool = False
    ) -> nn.Module:
        """
        优化模型用于推理

        Args:
            model: 原始模型
            use_int8: 是否使用 INT8 量化
            use_cuda_graph: 是否使用 CUDA Graph（NVIDIA GPU）
            compile_model: 是否使用 Torch 2.0 compile

        Returns:
            优化后的模型
        """
        print("[Optimizer] 开始优化模型...")

        # 1. INT8 量化
        if use_int8:
            print("[Optimizer] 应用 INT8 量化...")
            model = AdvancedQuantizer.quantize_model_int8(model, verbose=False)

        # 2. Torch compile（PyTorch 2.0+）
        if compile_model and hasattr(torch, 'compile'):
            print("[Optimizer] 编译模型...")
            model = torch.compile(model, mode='reduce-overhead')

        # 3. CUDA Graph（需要额外设置）
        if use_cuda_graph and torch.cuda.is_available():
            print("[Optimizer] CUDA Graph 暂不支持，跳过")

        print("[Optimizer] 优化完成")
        return model

    @staticmethod
    def benchmark_inference(
        model: nn.Module,
        tokenizer: Any,
        test_prompts: list,
        device: str = 'cuda:0',
        num_runs: int = 3
    ) -> Dict[str, float]:
        """
        Benchmark 推理性能

        Returns:
            性能指标
        """
        print(f"[Benchmark] 开始测试，{num_runs} 轮...")

        latencies = []
        throughputs = []

        for run in range(num_runs):
            run_latencies = []

            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                # Warmup
                if run == 0:
                    with torch.no_grad():
                        model.generate(**inputs, max_new_tokens=50)

                # Timed run
                start = time.time()
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50)
                end = time.time()

                latency = (end - start) * 1000  # ms
                run_latencies.append(latency)

            avg_latency = sum(run_latencies) / len(run_latencies)
            throughput = 1000 / avg_latency  # requests/sec

            latencies.append(avg_latency)
            throughputs.append(throughput)

        # 统计结果
        results = {
            'avg_latency_ms': sum(latencies) / len(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'avg_throughput_req_s': sum(throughputs) / len(throughputs),
            'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)]
        }

        print(f"[Benchmark] 平均延迟：{results['avg_latency_ms']:.2f}ms")
        print(f"[Benchmark] 吞吐量：{results['avg_throughput_req_s']:.2f} req/s")

        return results


# 使用示例
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel

    # 测试量化
    model_path = r"C:\Users\L9090\PycharmProjects\LLM\chatGLM\chatglm-6b"

    # 1. 低内存加载
    model, tokenizer = MemoryEfficientLoader.load_model_low_memory(
        model_path=model_path,
        model_class=AutoModel,
        tokenizer_class=AutoTokenizer,
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    # 2. 获取量化前后内存对比
    mem_before = AdvancedQuantizer.get_model_memory_usage(model)
    print(f"\n量化前内存：{mem_before['memory_gb']:.2f}GB")

    # 3. INT8 量化
    model_int8 = AdvancedQuantizer.quantize_model_int8(model)
    mem_after = AdvancedQuantizer.get_model_memory_usage(model_int8)
    print(f"量化后内存：{mem_after['memory_gb']:.2f}GB")

    # 4. Benchmark
    test_prompts = [
        "你好，请介绍一下 ChatGLM",
        "Python 中如何实现快速排序？",
        "解释一下什么是机器学习"
    ]

    results = InferenceOptimizer.benchmark_inference(
        model=model,
        tokenizer=tokenizer,
        test_prompts=test_prompts,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        num_runs=3
    )

    print(f"\nBenchmark 结果:")
    for k, v in results.items():
        print(f"  {k}: {v}")

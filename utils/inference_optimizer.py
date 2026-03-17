
# -*- coding: utf-8 -*-
"""
推理优化核心模块
实现动态 batching + KV cache 复用 + INT8 量化
"""

import torch
import time
from typing import List, Dict, Tuple, Optional, Any
from collections import OrderedDict
import threading
from queue import PriorityQueue, Queue
import numpy as np
import asyncio



class KVCacheManager:
    """KV Cache 管理器 - 复用历史对话的 KV 状态"""

    def __init__(self, max_cache_size: int = 100, cache_ttl: float = 300.0):
        """
        Args:
            max_cache_size: 最大缓存数量
            cache_ttl: 缓存生存时间 (秒)
        """
        self.max_cache_size = max_cache_size
        self.cache_ttl = cache_ttl
        self.cache = OrderedDict()  # session_id -> (past_key_values, timestamp)
        self.lock = threading.Lock()

    def get(self, session_id: str) -> Optional[Tuple]:
        """获取指定 session 的 KV cache"""
        with self.lock:
            if session_id not in self.cache:
                return None

            past_key_values, timestamp = self.cache[session_id]

            # 检查是否过期
            if time.time() - timestamp > self.cache_ttl:
                del self.cache[session_id]
                return None

            # 移到末尾 (最近使用)
            self.cache.move_to_end(session_id)
            return past_key_values

    def put(self, session_id: str, past_key_values: Tuple) -> None:
        """保存 KV cache"""
        with self.lock:
            # 如果缓存已满，删除最旧的
            if len(self.cache) >= self.max_cache_size:
                self.cache.popitem(last=False)

            self.cache[session_id] = (past_key_values, time.time())

    def remove(self, session_id: str) -> None:
        """删除指定 session 的缓存"""
        with self.lock:
            if session_id in self.cache:
                del self.cache[session_id]

    def clear(self):
        """清空所有缓存"""
        with self.lock:
            self.cache.clear()

    def cleanup_expired(self) -> int:
        """清理过期缓存，返回清理数量"""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                k for k, (_, ts) in self.cache.items()
                if current_time - ts > self.cache_ttl
            ]

            for key in expired_keys:
                del self.cache[key]

            return len(expired_keys)


class ModelCacheManager:
    """模型缓存管理器 - 减少冷启动时间"""

    def __init__(self, max_model_count: int = 3):
        """
        Args:
            max_model_count: 最大缓存模型数量
        """
        self.max_model_count = max_model_count
        self.model_cache = OrderedDict()  # model_path -> (model, tokenizer, load_time)
        self.lock = threading.Lock()

    def load_model(self, model_path: str, model_loader_fn) -> Tuple[Any, Any, float]:
        """
        加载模型（从缓存或重新加载）

        Args:
            model_path: 模型路径
            model_loader_fn: 模型加载函数

        Returns:
            (model, tokenizer, load_time)
        """
        with self.lock:
            start_time = time.time()

            # 尝试从缓存获取
            if model_path in self.model_cache:
                model, tokenizer, _ = self.model_cache[model_path]
                self.model_cache.move_to_end(model_path)
                load_time = time.time() - start_time
                print(f"[ModelCache] 从缓存加载模型：{model_path}, 耗时：{load_time:.3f}s")
                return model, tokenizer, load_time

            # 缓存未命中，需要重新加载
            print(f"[ModelCache] 缓存未命中，加载模型：{model_path}")
            model, tokenizer = model_loader_fn(model_path)

            # 如果缓存已满，删除最旧的
            if len(self.model_cache) >= self.max_model_count:
                self.model_cache.popitem(last=False)

            load_time = time.time() - start_time
            self.model_cache[model_path] = (model, tokenizer, load_time)

            print(f"[ModelCache] 新加载模型：{model_path}, 耗时：{load_time:.3f}s")
            return model, tokenizer, load_time

    def warmup(self, model_paths: List[str], model_loader_fn) -> None:
        """预加载模型到缓存"""
        print(f"[ModelCache] 开始预热 {len(model_paths)} 个模型...")
        for path in model_paths:
            self.load_model(path, model_loader_fn)
        print(f"[ModelCache] 预热完成，缓存中模型数：{len(self.model_cache)}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            return {
                'cached_models': len(self.model_cache),
                'max_capacity': self.max_model_count,
                'model_paths': list(self.model_cache.keys())
            }

    def clear(self):
        """清空模型缓存"""
        with self.lock:
            self.model_cache.clear()


class QuantizedModelWrapper:
    """量化模型包装器 - INT8 量化推理"""

    def __init__(self, model, tokenizer, quantization_bit: int = 8):
        """
        Args:
            model: 原始模型
            tokenizer: 分词器
            quantization_bit: 量化位数 (默认 8)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.quantization_bit = quantization_bit
        self.quantized = False

        if quantization_bit > 0:
            self._quantize()

    def _quantize(self):
        """执行量化操作"""
        if self.quantized:
            print("[Quantization] 模型已量化")
            return

        print(f"[Quantization] 开始 INT{self.quantization_bit}量化...")
        start_time = time.time()

        # 使用 ChatGLM 内置的量化方法
        if hasattr(self.model, 'quantize'):
            self.model = self.model.quantize(self.quantization_bit)
            self.quantized = True
            quant_time = time.time() - start_time
            print(f"[Quantization] 量化完成，耗时：{quant_time:.2f}s")
        else:
            print("[Quantization] 警告：模型不支持 quantize 方法，跳过量化")

    def generate(self, **kwargs):
        """生成接口"""
        with torch.no_grad():
            return self.model.generate(**kwargs)

    def chat(self, query: str, history: List = None, **kwargs):
        """对话接口"""
        if hasattr(self.model, 'chat'):
            return self.model.chat(self.tokenizer, query, history, **kwargs)
        else:
            # 回退到标准生成
            inputs = self.tokenizer(query, return_tensors="pt")
            outputs = self.generate(**inputs, **kwargs)
            response = self.tokenizer.decode(outputs[0])
            return response, history


class OptimizedInferenceEngine:
    """优化推理引擎 - 整合所有优化技术"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda:0',
        use_quantization: bool = True,
        quantization_bit: int = 8,
        enable_kv_cache: bool = True,
        max_batch_size: int = 8,
        cache_model: bool = True
    ):
        """
        Args:
            model_path: 模型路径
            device: 推理设备
            use_quantization: 是否使用量化
            quantization_bit: 量化位数
            enable_kv_cache: 启用 KV cache
            max_batch_size: 最大 batch 大小
            cache_model: 是否缓存模型
        """
        self.device = device
        self.enable_kv_cache = enable_kv_cache
        
        # 初始化各个管理器
        self.model_cache = ModelCacheManager() if cache_model else None
        self.kv_cache_manager = KVCacheManager() if enable_kv_cache else None
        
        # 加载模型
        self.model, self.tokenizer, load_time = self._load_model(
            model_path, 
            use_quantization,
            quantization_bit
        )
        
        self.load_time = load_time
        self.inference_count = 0
        self.total_time = 0.0
        self.batch_count = 0
        self.throughput_history = []
        
        print(f"[OptimizedEngine] 初始化完成")
        print(f"  - 模型加载时间：{load_time:.3f}s")
        print(f"  - KV Cache: {'启用' if enable_kv_cache else '禁用'}")
        print(f"  - 量化：{'INT' + str(quantization_bit) if use_quantization else '禁用'}")
        print(f"  - 最大 Batch 大小：{max_batch_size}")
    
    def _load_model(self, model_path: str, use_quantization: bool, quantization_bit: int):
        """加载模型（支持缓存）"""
        from transformers import AutoTokenizer, AutoModel
        
        def loader_fn(path):
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            model = AutoModel.from_pretrained(path, trust_remote_code=True).half().to(self.device)
            model.eval()
            
            if use_quantization and hasattr(model, 'quantize'):
                model = model.quantize(quantization_bit)
            
            return model, tokenizer
        
        if self.model_cache:
            return self.model_cache.load_model(model_path, loader_fn)
        else:
            start = time.time()
            model, tokenizer = loader_fn(model_path)
            return model, tokenizer, time.time() - start
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[str]:
        """
        处理一个批次的推理请求
        
        Args:
            batch: 请求列表
            
        Returns:
            回答列表
        """
        batch_start = time.time()
        
        # 构造 prompts
        prompts = []
        for req in batch:
            prompt = f"Instruction: {req['instruction']}\n"
            if req.get('sentence'):
                prompt += f"Input: {req['sentence']}\n"
            prompt += "Answer: "
            prompts.append(prompt)
        
        # Batch tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Batch generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=req.get('max_new_tokens', 300),
                num_return_sequences=1
            )
        
        # Decode
        responses = []
        for i, output in enumerate(outputs):
            response = self.tokenizer.decode(output, skip_special_tokens=True)
            response = response.split('Answer: ')[-1] if 'Answer: ' in response else response
            responses.append(response)
        
        # 统计
        batch_time = time.time() - batch_start
        self.batch_count += 1
        throughput = len(batch) / batch_time
        self.throughput_history.append(throughput)
        
        print(f"[BatchProcess] Batch 大小：{len(batch)}, 耗时：{batch_time:.3f}s, 吞吐量：{throughput:.2f} req/s")
        
        return responses
    
    def infer(
        self,
        instruction: str,
        sentence: str = "",
        session_id: Optional[str] = None,
        max_new_tokens: int = 300,
        use_kv_cache: bool = True
    ) -> str:
        """
        单次推理（支持 KV cache 复用）
        
        Args:
            instruction: 指令
            sentence: 输入句子
            session_id: 会话 ID（用于 KV cache）
            max_new_tokens: 最大生成长度
            use_kv_cache: 是否使用 KV cache
            
        Returns:
            生成的回答
        """
        start_time = time.time()
        
        # 普通推理模式（带 KV cache）
        input_text = f"Instruction: {instruction}\n"
        if sentence:
            input_text += f"Input: {sentence}\n"
        input_text += "Answer: "
        
        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # 获取 KV cache（如果有）
        past_key_values = None
        if use_kv_cache and session_id and self.kv_cache_manager:
            past_key_values = self.kv_cache_manager.get(session_id)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                past_key_values=past_key_values,
                use_cache=use_kv_cache
            )
        
        # 保存 KV cache
        if use_kv_cache and session_id and self.kv_cache_manager:
            new_past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
            if new_past_key_values:
                self.kv_cache_manager.put(session_id, new_past_key_values)
        
        # 解码
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split('Answer: ')[-1] if 'Answer: ' in response else response
        
        # 统计
        inference_time = time.time() - start_time
        self.inference_count += 1
        self.total_time += inference_time
        
        print(f"[Inference] 耗时：{inference_time:.3f}s, 平均：{self.total_time/self.inference_count:.3f}s")
        
        return response
    
    def batch_infer(
        self,
        requests: List[Dict[str, str]],
        max_new_tokens: int = 300
    ) -> List[str]:
        """
        批量推理
        
        Args:
            requests: 请求列表 [{"instruction": "...", "sentence": "..."}, ...]
            max_new_tokens: 最大生成长度
            
        Returns:
            回答列表
        """
        if not requests:
            return []
        
        start_time = time.time()
        
        # 构造 prompts
        prompts = []
        for req in requests:
            prompt = f"Instruction: {req['instruction']}\n"
            if req.get('sentence'):
                prompt += f"Input: {req['sentence']}\n"
            prompt += "Answer: "
            prompts.append(prompt)
        
        # Batch tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Batch generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_return_sequences=1
            )
        
        # Decode
        responses = []
        for i, output in enumerate(outputs):
            response = self.tokenizer.decode(output, skip_special_tokens=True)
            response = response.split('Answer: ')[-1] if 'Answer: ' in response else response
            responses.append(response)
        
        # 统计
        batch_time = time.time() - start_time
        avg_time = batch_time / len(requests)
        throughput = len(requests) / batch_time
        
        print(f"[BatchInference] Batch 大小：{len(requests)}, 总耗时：{batch_time:.3f}s, 平均：{avg_time:.3f}s/请求，吞吐量：{throughput:.2f} req/s")
        
        return responses
    
    def benchmark_throughput(self, num_requests: int = 100) -> Dict[str, float]:
        """
        基准测试吞吐量
        
        Args:
            num_requests: 测试请求数量
            
        Returns:
            吞吐量统计数据
        """
        test_requests = [
            {"instruction": "翻译", "sentence": "Hello, how are you?"},
            {"instruction": "总结", "sentence": "今天天气晴朗，适合外出游玩。"},
            {"instruction": "问答", "sentence": "Python 的创始人是谁？"}
        ] * (num_requests // 3 + 1)
        test_requests = test_requests[:num_requests]
        
        # 顺序推理
        print("\n=== 顺序推理测试 ===")
        start = time.time()
        for req in test_requests:
            self.infer(req['instruction'], req['sentence'], max_new_tokens=50)
        sequential_time = time.time() - start
        sequential_throughput = num_requests / sequential_time
        
        print(f"顺序推理：{sequential_time:.3f}s, 吞吐量：{sequential_throughput:.2f} req/s")
        
        # 批量推理
        print("\n=== 批量推理测试 ===")
        start = time.time()
        self.batch_infer(test_requests, max_new_tokens=50)
        batch_time = time.time() - start
        batch_throughput = num_requests / batch_time
        
        print(f"批量推理：{batch_time:.3f}s, 吞吐量：{batch_throughput:.2f} req/s")
        
        speedup = batch_throughput / sequential_throughput
        
        return {
            'sequential_throughput': sequential_throughput,
            'batch_throughput': batch_throughput,
            'speedup': speedup
        }
    
    def benchmark_memory(self) -> Dict[str, float]:
        """
        基准测试显存占用
        
        Returns:
            显存统计数据
        """
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
        
        torch.cuda.empty_cache()
        
        # FP16 模型显存
        from transformers import AutoTokenizer, AutoModel
        tokenizer_fp16 = AutoTokenizer.from_pretrained(
            next(iter(self.model_cache.model_cache.keys())) if self.model_cache else "tmp",
            trust_remote_code=True
        )
        mem_fp16 = torch.cuda.memory_allocated() / (1024**3)
        
        # INT8 模型显存
        if hasattr(self.model, 'quantized') and self.model.quantized:
            mem_int8 = torch.cuda.memory_allocated() / (1024**3)
            savings = (1 - mem_int8 / mem_fp16) * 100 if mem_fp16 > 0 else 0
            
            print(f"\n=== 显存占用测试 ===")
            print(f"FP16 显存：{mem_fp16:.2f}GB")
            print(f"INT8 显存：{mem_int8:.2f}GB")
            print(f"显存节省：{savings:.1f}%")
            
            return {
                'fp16_memory_gb': mem_fp16,
                'int8_memory_gb': mem_int8,
                'memory_reduction_percent': savings
            }
        
        return {'fp16_memory_gb': mem_fp16}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = {
            'total_inferences': self.inference_count,
            'total_time': self.total_time,
            'avg_inference_time': self.total_time / max(1, self.inference_count),
            'model_load_time': self.load_time,
            'kv_cache_enabled': self.enable_kv_cache,
            'batch_count': self.batch_count,
            'kv_cache_count': len(self.kv_cache_manager.cache) if self.kv_cache_manager else 0,
        }
        
        if self.throughput_history:
            stats['avg_throughput'] = sum(self.throughput_history) / len(self.throughput_history)
            stats['max_throughput'] = max(self.throughput_history)
        
        if self.model_cache:
            stats['model_cache'] = self.model_cache.get_cache_stats()
        
        if torch.cuda.is_available():
            stats['memory_allocated_gb'] = torch.cuda.memory_allocated(self.device) / (1024**3)
            stats['memory_reserved_gb'] = torch.cuda.memory_reserved(self.device) / (1024**3)
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        if self.kv_cache_manager:
            cleaned = self.kv_cache_manager.cleanup_expired()
            print(f"[Cleanup] 清理了 {cleaned} 个过期 KV cache")
        
        if self.model_cache:
            stats = self.model_cache.get_cache_stats()
            print(f"[Cleanup] 模型缓存：{stats['cached_models']}/{stats['max_capacity']}")

# 使用示例
if __name__ == "__main__":
    # 初始化优化推理引擎
    engine = OptimizedInferenceEngine(
        model_path=r"C:\Users\L9090\PycharmProjects\LLM\chatGLM\chatglm-6b",
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        use_quantization=True,
        quantization_bit=8,
        enable_kv_cache=True,
        cache_model=True
    )

    # 测试单次推理（带 KV cache）
    response1 = engine.infer(
        instruction="现在你是一个非常厉害的 SPO 抽取器。",
        sentence="黄磊是一个特别幸运的演员。",
        session_id="user_001",
        max_new_tokens=100
    )
    print(f"回答 1: {response1}")

    # 测试多轮对话（复用 KV cache）
    response2 = engine.infer(
        instruction="继续分析",
        sentence="这句话的情感倾向是什么？",
        session_id="user_001",
        max_new_tokens=100
    )
    print(f"回答 2: {response2}")

    # 测试批量推理
    requests = [
        {"instruction": "翻译以下句子", "sentence": "Hello, how are you?"},
        {"instruction": "提取实体", "sentence": "马云创立了阿里巴巴"},
        {"instruction": "情感分析", "sentence": "这个产品很好用"}
    ]

    responses = engine.batch_infer(requests, max_new_tokens=50)
    for i, resp in enumerate(responses):
        print(f"批量请求{i}: {resp}")

    # 打印性能统计
    stats = engine.get_performance_stats()
    print(f"\n性能统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 清理
    engine.cleanup()


def create_optimized_engine(
    model_path: str,
    device: str = 'cuda:0',
    use_quantization: bool = True,
    quantization_bit: int = 8,
    enable_kv_cache: bool = True,
    cache_model: bool = True
) -> OptimizedInferenceEngine:
    """创建优化推理引擎的工厂函数"""
    return OptimizedInferenceEngine(
        model_path=model_path,
        device=device,
        use_quantization=use_quantization,
        quantization_bit=quantization_bit,
        enable_kv_cache=enable_kv_cache,
        cache_model=cache_model
    )

# ======================== 板块 1：导入依赖库 ========================
import time
import torch
from transformers import AutoTokenizer, AutoModel
from utils.inference_optimizer import OptimizedInferenceEngine

# ======================== 板块 2：定义核心推理函数 ========================
def inference(
        model,
        tokenizer,
        instuction: str,
        sentence: str,
        session_id: str = None,
        kv_cache_manager = None,
        max_new_tokens: int = 300
    ):
    """
    优化的 inference 函数（支持 KV cache 复用）
    """
    print(f'instruction-->{instuction}')
    print(f'sentence-->{sentence}')
    
    model.eval()
    
    with torch.no_grad():
        input_text = f"Instruction: {instuction}\n"
        if sentence:
            input_text += f"Input: {sentence}\n"
        input_text += "Answer: "
        
        print(f'input_text-->{input_text}')
        
        batch = tokenizer(input_text, return_tensors="pt")
        print(f'batch--->{batch["input_ids"].shape}')
        
        # 获取 KV cache
        past_key_values = None
        if session_id and kv_cache_manager:
            past_key_values = kv_cache_manager.get(session_id)
            if past_key_values:
                print(f"[KV Cache] 复用 session {session_id} 的缓存")
        
        # Generate
        out = model.generate(
            input_ids=batch["input_ids"].to(model.device),
            max_new_tokens=max_new_tokens,
            past_key_values=past_key_values,
            use_cache=(past_key_values is not None)
        )
        
        # 保存 KV cache
        if session_id and kv_cache_manager:
            if hasattr(out, 'past_key_values') and out.past_key_values:
                kv_cache_manager.put(session_id, out.past_key_values)
                print(f"[KV Cache] 已更新 session {session_id} 的缓存")
        
        out_text = tokenizer.decode(out[0])
        print(f'out_text-->{out_text}')
        
        answer = out_text.split('Answer: ')[-1]
        return answer


# ======================== 板块 3：主函数 ========================
if __name__ == '__main__':
    from rich import print
    from utils.inference_optimizer import (
        OptimizedInferenceEngine,
        KVCacheManager,
        ModelCacheManager
    )
    
    device = 'mps:0' if torch.backends.mps.is_available() else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    max_new_tokens = 300
    model_path = r"C:\Users\L9090\PycharmProjects\LLM\chatGLM\chatglm-6b"
    
    # 方案 A: 使用优化推理引擎（推荐）
    print("\n=== 使用优化推理引擎 ===")
    engine = OptimizedInferenceEngine(
        model_path=model_path,
        device=device,
        use_quantization=True,
        quantization_bit=8,
        enable_kv_cache=True,
        cache_model=True
    )
    
    samples = [
        {
            'instruction': "现在你是一个非常厉害的 SPO 抽取器。",
            "input": "下面这句中包含了哪些三元组，用 json 列表的形式回答。\n\n73 获奖记录人物评价：黄磊是一个特别幸运的演员，拍第一部戏就碰到了导演陈凯歌。",
        },
        {
            'instruction': "你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。",
            "input": "下面子中的主语是什么类别，输出成列表形式。\n\n第 N 次入住了，就是方便去客户那里哈哈。"
        }
    ]
    
    # 测试优化引擎
    start = time.time()
    
    # 单条推理（带 KV cache）
    response1 = engine.infer(
        instruction=samples[0]['instruction'],
        sentence=samples[0]['input'],
        session_id="test_session_1",
        max_new_tokens=max_new_tokens
    )
    print(f'\n回答 1: {response1}')
    
    # 多轮对话（复用 KV cache）
    response2 = engine.infer(
        instruction="继续分析这个句子",
        sentence="情感倾向是什么？",
        session_id="test_session_1",
        max_new_tokens=max_new_tokens
    )
    print(f'\n回答 2: {response2}')
    
    # 批量推理
    batch_requests = [
        {"instruction": "翻译", "sentence": "Hello world"},
        {"instruction": "总结", "sentence": "今天天气很好"}
    ]
    batch_responses = engine.batch_infer(batch_requests, max_new_tokens=100)
    print(f'\n批量推理结果：{batch_responses}')
    
    # 性能统计
    stats = engine.get_performance_stats()
    print(f"\n=== 性能统计 ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    
    print(f'\n总耗时：{round(time.time() - start, 2)}s')
    
    # 清理
    engine.cleanup()
    
    # 方案 B: 传统方式（对比）
    print("\n=== 传统推理方式（对比）===")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to(device)
    model.eval()
    
    kv_cache_manager = KVCacheManager()
    
    start = time.time()
    for i, sample in enumerate(samples):
        print(f'\nsample {i}-->{sample}')
        print("*"*80)
        
        res = inference(
            model,
            tokenizer,
            sample['instruction'],
            sample['input'],
            session_id=f"session_{i}",
            kv_cache_manager=kv_cache_manager,
            max_new_tokens=max_new_tokens
        )
        
        print(f'res {i}: {res}')
        break
    
    print(f'传统方式耗时：{round(time.time() - start, 2)}s')

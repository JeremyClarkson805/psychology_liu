import sys
import os

# 将 src 目录添加到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm.client import LLMClient
from config.llm_config import LLMConfig

def test_kimi_temperatures():
    config = LLMConfig()
    client = LLMClient(config)
    
    model_name = "kimi-k2.5"
    prompt = "你好"
    temperatures = [0, 0.5, 1, 1.5, 2]
    
    print(f"🚀 开始测试模型: {model_name}")
    print(f"📝 输入内容: {prompt}")
    
    for temp in temperatures:
        print(f"\n{'='*40}")
        print(f"🌡️ 当前设置温度: {temp}")
        print(f"{'='*40}")
        
        try:
            # 模拟 client.chat 的内部逻辑，但手动指定温度
            messages = [{"role": "user", "content": prompt}]
            extra_body = config.get_extra_body(model_name)
            
            # 直接调用 _do_stream 来强制使用我们指定的温度
            # 注意：如果 Kimi API 确实不支持非 1 的温度，这里会抛出异常
            response_text = client._do_stream(
                model_name=model_name,
                messages=messages,
                temperature=temp,
                extra_body=extra_body,
                verbose=True
            )
            
            print(f"\n✅ 温度 {temp} 响应成功")
        except Exception as e:
            print(f"\n❌ 温度 {temp} 请求失败: {e}")

if __name__ == "__main__":
    test_kimi_temperatures()

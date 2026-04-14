import sys
import os

# 将 src 目录添加到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm.client import LLMClient
from config.llm_config import LLMConfig
from persona_runner import get_system_prompt, PERSONAS

def run_thinking_test(client, model_name, extra_body=None):
    # 使用 persona_runner.py 中的第一个人格
    persona = PERSONAS[0]
    persona_desc = persona["description"]
    
    # 获取真实的 system_prompt
    system_prompt = get_system_prompt(persona_desc)
    
    # 构造题目信息 (模拟问卷4第12题)
    q_title = "问卷四 执行功能行为评定量表"
    q = {
        "question_num": 12,
        "question_text": "我容易反应过度，情绪激动",
        "options": ["没有", "有时", "经常"]
    }
    
    # 使用 persona_runner.py 中的 base_prompt 格式
    prompt = (
        f"当前问卷：【{q_title}】\n"
        f"题号: {q['question_num']} | 题目: {q['question_text']} | 可选选项: {q['options']}\n"
        "请选择最符合你现状的选项内容（直接写出选项的具体文字），并返回严格的 JSON 格式。"
    )
    
    print(f"\n{'='*60}")
    print(f"🧪 测试模型: {model_name} | extra_body: {extra_body}")
    print(f"{'='*60}")
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # 调用 client 内部的 openai 客户端以支持 reasoning_content 捕获
        response = client.client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            extra_body=extra_body
        )
        
        last_type = None
        
        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta:
                # 尝试获取思维链内容 (DeepSeek 官方字段)
                reasoning = getattr(delta, 'reasoning_content', None)
                content = getattr(delta, 'content', None)
                
                if reasoning:
                    if last_type != 'thinking':
                        print(f"\n\033[34m[THINKING PROCESS]\033[0m\n", end="", flush=True)
                        last_type = 'thinking'
                    print(reasoning, end="", flush=True)
                
                if content:
                    if last_type != 'content':
                        print(f"\n\n\033[32m[FINAL RESPONSE (JSON)]\033[0m\n", end="", flush=True)
                        last_type = 'content'
                    print(content, end="", flush=True)
                    
        print(f"\n\n✅ {model_name} 测试完成")
    except Exception as e:
        print(f"\n❌ {model_name} 测试失败: {e}")

def main():
    config = LLMConfig()
    client = LLMClient(config)
    
    # 测试 1: 常规模型 v3.2 + 手动开启 thinking 参数
    run_thinking_test(client, "deepseek-v3.2", extra_body={"thinking": {"type": "enabled"}})
    
    # 测试 2: v-api 专用思考模型 deepseek-v3.2-thinking
    run_thinking_test(client, "deepseek-v3.2", extra_body={"thinking": {"type": "disabled"}})

if __name__ == "__main__":
    main()

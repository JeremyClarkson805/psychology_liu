import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.llm_config import LLMConfig
from src.llm.client import LLMClient

def main():
    config = LLMConfig()
    
    print("Config created successfully:")
    print(config.model_dump())
    
    client = LLMClient(config=config)
    print("\nClient created successfully.")
    print("Models configured:", client.config.model_names)
    
    print("\n[Testing Connection] Sending a test message: '你好'...")
    responses = client.chat(prompt="你好", system_prompt="You are a helpful assistant.")
    for model, resp in responses.items():
        print(f"\n--- Response from {model} ---")
        print(resp)

if __name__ == "__main__":
    main()

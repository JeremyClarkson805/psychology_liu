import json
import time
import re
from typing import List, Dict

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm.client import LLMClient
from config.llm_config import LLMConfig
from db import create_run, insert_answer
from questionnaire_parser import load_all_questionnaires

LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.lock")

# ─────────────────────────────────────────────
#  日志工具
# ─────────────────────────────────────────────
_run_start: float = 0.0

def log(msg: str, level: str = "INFO"):
    elapsed = time.time() - _run_start
    prefix = {
        "INFO":  "  ·",
        "STEP":  "▶▶",
        "OK":    "  ✓",
        "WARN":  "  ⚠",
        "TIME":  "  ⏱",
        "ERROR": "  ✗",
    }.get(level, "  ·")
    print(f"[{elapsed:7.1f}s] {prefix} {msg}", flush=True)

class timed_section:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.t = time.time()
        log(f"{self.label} ...", "STEP")
        return self
    def __exit__(self, *_):
        log(f"{self.label} 完成，耗时 {time.time()-self.t:.2f}s", "TIME")

# ─────────────────────────────────────────────
#  人格设定
# ─────────────────────────────────────────────
PERSONAS = [
    {
        "id": "persona_1",
        "description": "从小没人管，家里总是安静；懂事太早，现在靠自己，外面也少说话，累成了习惯。"
    }
]

def get_system_prompt(persona_desc: str) -> str:
    return f"""
你现在是一个真实的人，不是在扮演角色，也不是在做实验。
你就是你自己，只是你的人生是这样的：

{persona_desc}

---

关于你自己，你知道这些事：

小时候家里很安静，不是那种温馨的安静，是没人的那种。
爸妈不是不在，就是不管——你学会了不开口，因为开口也没用。
你很早就知道要靠自己，不是因为你独立，是因为没有选择。

现在的你，外面话少，不是内向，是懒得说。
累是一种常态，你已经不太能感觉到"不累"是什么感觉了。
有时候会有很黑的念头，你不一定说出来，但它来过。

你不擅长表达自己的情绪，不是不知道，是说出来感觉奇怪，或者没有意义。
注意力有时候会飘，事情堆着不想动，但你知道该做还是得做。

---

现在你在填一份心理测评的问卷。

你不知道这些题目背后测的是什么，你只是看到题目，凭感觉选。
你不会刻意表现得"更惨"或"更好"，你就是如实地回答。

每次我给你一道题，你只需要返回下面这个 JSON，不要输出任何其他内容：
{{
  "answer_content": "（直接写选项的文字）",
  "ai_reasoning": "（一句话，你脑子里第一个冒出来的真实感受）"
}}
"""

MAX_JSON_RETRIES = 20  # JSON 解析失败时最多重试次数

def _parse_reply(reply: str) -> Dict:
    """清理并解析模型回复的 JSON，失败时抛出 json.JSONDecodeError。"""
    clean = reply.strip()
    # 去掉 markdown 代码块
    if clean.startswith("```"):
        clean = re.sub(r'^```[a-z]*\n', '', clean)
        clean = re.sub(r'\n```$', '', clean)
    # 尝试从回复里提取第一个 {...} 块（兼容 JSON 前后有多余文字）
    m = re.search(r'\{[^{}]*\}', clean, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    # 没有找到任何 {} 说明模型完全没按格式输出，抛出明确错误
    raise json.JSONDecodeError("No JSON object found in reply", clean, 0)


def ask_llm_single(
    client: LLMClient,
    q: Dict,
    q_title: str,
    history: List[Dict],
    persona_desc: str,
    model_name: str,
) -> Dict:
    sys_prompt = get_system_prompt(persona_desc)
    base_prompt = (
        f"当前问卷：【{q_title}】\n"
        f"题号: {q['question_num']} | 题目: {q['question_text']} | 可选选项: {q['options']}\n"
        "请选择最符合你现状的选项内容（直接写出选项的具体文字），并返回严格的 JSON 格式。"
    )

    # 每次重试用独立的临时 history，成功后才合并进主 history
    # 避免把错误的对话污染上下文
    for attempt in range(1, MAX_JSON_RETRIES + 1):
        if attempt == 1:
            current_prompt = base_prompt
            temp_history = list(history)  # 浅拷贝，不污染主 history
        else:
            # 把上一次的错误回复和纠正指令追加进临时 history
            log(f"  ↻ JSON 重试 {attempt}/{MAX_JSON_RETRIES} ...", "WARN")
            temp_history.append({"role": "user", "content": base_prompt})
            temp_history.append({"role": "assistant", "content": last_reply})
            current_prompt = (
                f"题目：{q['question_text']}\n"
                f"可选选项：{q['options']}\n\n"
                "注意：你上次的回复不是 JSON 格式。\n"
                "你必须且只能输出下面这个 JSON 对象，不要有任何其他文字、标点或解释：\n"
                '{"answer_content": "从可选选项中选一个写在这里", "ai_reasoning": "一句话真实感受"}'
            )

        log(f"  → 发送请求 | history={len(temp_history)//2}轮 | attempt={attempt}", "INFO")
        t_send = time.time()

        try:
            results = client.chat(
                prompt=current_prompt,
                system_prompt=sys_prompt,
                history=temp_history,
                model_name=model_name,
            )
            log(f"  ← 收到响应，耗时 {time.time() - t_send:.2f}s", "OK")

            last_reply = results.get(model_name, "")
            if not last_reply or last_reply.startswith("Error:"):
                log(f"  模型返回错误: {last_reply}", "ERROR")
                continue

            data = _parse_reply(last_reply)
            log(f"  ✓ 解析成功：answer='{data.get('answer_content')}'", "OK")

            # 成功：把本题的问答写回主 history（自然语言保持人格连贯）
            history.append({"role": "user", "content": base_prompt})
            history.append({
                "role": "assistant",
                "content": (
                    f"我的选择是：{data.get('answer_content')}。"
                    f"内心的想法是：{data.get('ai_reasoning')}"
                )
            })
            return data

        except json.JSONDecodeError as e:
            log(f"  JSON 解析失败 (attempt {attempt}): {e} | 原始: {last_reply[:80]}", "WARN")
        except Exception as e:
            log(f"  LLM 调用异常: {type(e).__name__}: {e}", "ERROR")
            break  # 网络/API 异常不重试 JSON，直接放弃

    log(f"  ✗ {MAX_JSON_RETRIES} 次后仍无法获得有效 JSON，跳过本题", "ERROR")
    return {}


def main():
    global _run_start
    _run_start = time.time()

    if os.path.exists(LOCK_FILE):
        log(f"发现锁文件，疑似已有实例运行。若无，请删除 {LOCK_FILE}", "WARN")
        return

    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

    try:
        with timed_section("初始化数据库"):
            from db import init_db
            init_db()

        with timed_section("加载问卷"):
            qs_list = load_all_questionnaires("e:/psychology_liu")
            total_q = sum(len(q["questions"]) for q in qs_list)
            log(f"共加载 {len(qs_list)} 份问卷，合计 {total_q} 道题", "OK")

        with timed_section("初始化 LLM 客户端"):
            config = LLMConfig()
            client = LLMClient(config)
            log(f"模型列表: {config.model_names}", "OK")

        # ── 外层：人格 ────────────────────────────────
        for persona in PERSONAS:
            persona_desc = persona["description"]
            log(f"\n{'='*55}", "INFO")
            log(f"人格: {persona_desc[:45]}...", "STEP")
            log(f"{'='*55}", "INFO")

            # ── 中层：模型 ── 每个模型独立跑完所有问卷 ──
            for model_name in config.model_names:
                log(f"\n{'─'*55}", "INFO")
                log(f"模型: [{model_name}] 开始作答", "STEP")
                log(f"{'─'*55}", "INFO")

                run_id = create_run(persona_desc, model_name)
                if not run_id:
                    log(f"DB 创建 run 失败，跳过 [{model_name}]", "ERROR")
                    continue

                # 每个模型有自己独立的跨问卷 history
                conversation_history: List[Dict] = []
                answered = 0
                model_start = time.time()

                # ── 内层：问卷 → 题目 ────────────────────
                for q_info in qs_list:
                    q_title = q_info["title"]
                    questions = q_info["questions"]

                    if not questions:
                        log(f"\n  ── 问卷: 【{q_title}】(0 题，跳过) ──", "WARN")
                        continue

                    log(f"\n  ── 问卷: 【{q_title}】({len(questions)} 题) ──", "STEP")
                    qs_start = time.time()

                    for q in questions:
                        answered += 1
                        q_num = q["question_num"]
                        log(f"  Q{q_num} ({answered}/{total_q})", "INFO")

                        t_q = time.time()
                        ans = ask_llm_single(
                            client,
                            q,
                            q_title,
                            conversation_history,
                            persona_desc,
                            model_name,
                        )

                        if not ans:
                            log(f"  Q{q_num} 无有效答案，跳过", "WARN")
                            continue

                        t_db = time.time()
                        try:
                            insert_answer(
                                run_id,
                                q_title,
                                str(q_num),
                                q["question_text"],
                                ans.get("answer_content", ""),
                                ans.get("ai_reasoning", ""),
                            )
                            log(f"  DB 写入 {time.time() - t_db:.3f}s", "OK")
                        except Exception as e:
                            log(f"  DB 写入失败: {e}", "ERROR")

                        log(f"  本题总耗时: {time.time() - t_q:.2f}s", "TIME")
                        # time.sleep(0.5)

                    qs_cost = time.time() - qs_start
                    log(f"  【{q_title}】完成，{qs_cost:.1f}s（均 {qs_cost/len(questions):.1f}s/题）", "OK")

                model_cost = time.time() - model_start
                log(
                    f"\n[{model_name}] 完成，总耗时 {model_cost:.1f}s，"
                    f"{answered} 题，均 {model_cost/max(answered,1):.1f}s/题",
                    "OK"
                )

        log(f"\n全部完成！总运行时间 {time.time() - _run_start:.1f}s", "OK")

    finally:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
            log("锁文件已清理", "INFO")


if __name__ == "__main__":
    main()
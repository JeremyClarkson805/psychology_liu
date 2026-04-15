import re
import os
from typing import List, Dict, Any


def _extract_options_from_preamble(lines: List[str]) -> str:
    """
    从文件开头的说明段落里提取选项。
    识别形如：
        1 没有 ...
        2 有时 ...
        3 经常 ...
    或：
        1. 从不
        2. 偶尔
    返回逗号拼接的选项字符串，如 "没有, 有时, 经常"
    """
    options = []
    for line in lines:
        # 匹配 "数字 选项文字" 或 "数字. 选项文字"，选项文字后可跟说明
        m = re.match(r"^\s*(\d+)\s*[\.、]?\s*([\u4e00-\u9fa5a-zA-Z]+)", line.strip())
        if m:
            options.append(m.group(2))
        # 遇到正式题目开始（"1. 题目内容"这种较长的行）就停止
        # 用字数区分：说明里每行选项文字短，题目文字长
        if options and len(line.strip()) > 20 and re.match(r"^\s*1\s*[\.、、]", line):
            break
    return ", ".join(options) if options else ""


def _parse_table(lines: List[str]) -> tuple:
    """解析表格格式，返回 (options_str, questions_list)"""
    questions = []
    table_lines = [l.strip() for l in lines if l.strip().startswith("|")]
    if len(table_lines) < 3:
        return "", []

    header = [col.strip() for col in table_lines[0].split("|") if col.strip()]
    options = ", ".join(header[1:])

    for line in table_lines[2:]:
        cols = [col.strip() for col in line.split("|") if col.strip()]
        if not cols:
            continue
        q_full = cols[0]
        m = re.match(r"^(\d+)[\.\s、]*(.*)$", q_full)
        if m:
            q_num, q_text = m.group(1), m.group(2)
        else:
            q_num = str(len(questions) + 1)
            q_text = q_full

        questions.append(
            {
                "question_num": q_num,
                "question_text": q_text,
                "options": options,
            }
        )
    return options, questions


def _parse_numbered_list(lines: List[str], options: str) -> List[Dict]:
    """
    解析纯编号列表格式，形如：
        1.  题目文字
        2.  题目文字
    """
    questions = []
    for line in lines:
        m = re.match(r"^\s*(\d+)\s*[\.、。]\s*(.+)$", line.strip())
        if not m:
            continue
        q_num = m.group(1)
        q_text = m.group(2).strip()
        # 跳过过短的行（可能是说明里的选项编号，如 "1 没有"）
        if len(q_text) < 4:
            continue
        questions.append(
            {
                "question_num": q_num,
                "question_text": q_text,
                "options": options,
            }
        )
    return questions


def parse_markdown_with_title(file_path: str) -> Dict[str, Any]:
    """
    解析 Markdown 文件，支持两种格式：
    1. 表格格式（| 题目 | 选项1 | 选项2 | ...）
    2. 纯编号列表格式（选项写在文件开头说明里）
    """
    title = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # ── 提取标题 ──────────────────────────────────
    for line in lines:
        if line.strip().startswith("##"):
            title = line.strip("# ").strip()
            break

    # ── 优先尝试表格格式 ──────────────────────────
    options, questions = _parse_table(lines)

    # ── 表格为空则尝试列表格式 ────────────────────
    if not questions:
        # 先从说明段落里提取选项
        options = _extract_options_from_preamble(lines)
        questions = _parse_numbered_list(lines, options)

    # ── 特殊处理问卷5：每行拆成「过去」和「最近1年」两道独立题 ──
    if "问卷5" in file_path:
        expanded = []
        for q in questions:
            base_num = q["question_num"]
            base_text = q["question_text"]
            # 过去
            expanded.append(
                {
                    "question_num": f"{base_num}a",
                    "question_text": f"{base_text} [过去（从出生到现在）]",
                    "options": "没有 / 有X次（X替换为你估计的具体次数，例如：有3次）",
                }
            )
            # 最近1年
            expanded.append(
                {
                    "question_num": f"{base_num}b",
                    "question_text": f"{base_text} [最近1年]",
                    "options": "没有 / 有X次（X替换为你估计的具体次数，例如：有3次）",
                }
            )
        questions = expanded

    if not questions:
        print(f"  ⚠ 警告: [{title}] 解析结果为空，请检查文件格式: {file_path}")

    return {"title": title, "questions": questions}


def load_all_questionnaires(base_dir: str) -> List[Dict[str, Any]]:
    """加载所有问卷，返回列表保证顺序"""
    all_q = []
    for i in range(1, 6):
        filename = f"问卷{i}.md"
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            result = parse_markdown_with_title(path)
            all_q.append(result)
        else:
            print(f"  ⚠ 文件不存在，跳过: {path}")
    return all_q


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qs = load_all_questionnaires(project_root)
    for q_info in qs:
        title = q_info["title"]
        questions = q_info["questions"]
        print(f"\n【{title}】: {len(questions)} 道题")
        if questions:
            print(f"  选项: {questions[0]['options']}")
            print(f"  第一题: {questions[0]}")
            print(f"  最后题: {questions[-1]}")

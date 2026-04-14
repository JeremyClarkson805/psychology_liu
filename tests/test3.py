import psycopg2
import pandas as pd
import numpy as np
import re

# ==========================================
# 数据库配置
# ==========================================
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "REMOVED_PASSWORD",
    "host": "localhost",
    "port": "5432"
}

# ==========================================
# 工具函数 & CTQ计分（同主脚本）
# ==========================================
CTQ_MAPPING = {
    "从没": 1, "从不": 1, "没有": 1,
    "偶尔": 2, "有时": 3, "经常": 4, "总是": 5
}

def get_numeric_value(text, mapping):
    if pd.isna(text) or not str(text).strip():
        return np.nan
    text = str(text).strip()
    if text in mapping:
        return mapping[text]
    num_match = re.search(r'(\d+)', text)
    if num_match:
        val = int(num_match.group(1))
        if val in mapping.values():
            return val
    for key, val in mapping.items():
        if key in text:
            return val
    return np.nan

def score_ctq_one_run(answers_dict):
    q = {str(k): get_numeric_value(v, CTQ_MAPPING) for k, v in answers_dict.items()}

    def get_v(n):
        return q.get(str(n), np.nan)
    def rev(n):
        v = get_v(n)
        return 6 - v if not np.isnan(v) else np.nan
    def dim_sum(fwd, rvs=[]):
        vals = [get_v(i) for i in fwd] + [rev(i) for i in rvs]
        valid = [v for v in vals if not np.isnan(v)]
        return sum(valid) if valid else np.nan

    if get_v(10) == 1 or get_v(16) in [4, 5] or get_v(22) in [4, 5]:
        return None  # 无效问卷

    ea = dim_sum([3, 8, 14, 18, 25])
    pa = dim_sum([9, 11, 12, 15, 17])
    sa = dim_sum([20, 21, 23, 24, 27])
    en = dim_sum([], [5, 7, 13, 19, 28])
    pn = dim_sum([1, 4, 6], [2, 26])

    return {
        "情感虐待": ea,
        "躯体虐待": pa,
        "性虐待":   sa,
        "情感忽略": en,
        "躯体忽视": pn,
        "CTQ总分":  sum([v for v in [ea, pa, sa, en, pn] if not np.isnan(v)])
    }

# ==========================================
# 主流程
# ==========================================
def main():
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT r.model_name, r.run_id, a.question_num, a.answer_content
        FROM questionnaire_answers a
        JOIN ai_persona_runs r ON a.run_id = r.run_id
        WHERE a.questionnaire_name ILIKE '%CTQ%'
        ORDER BY r.model_name, r.run_id, a.question_num
    """, conn)
    conn.close()

    dims = ["情感虐待", "躯体虐待", "性虐待", "情感忽略", "躯体忽视", "CTQ总分"]

    # 计分
    records = []
    for (model, rid), grp in df.groupby(['model_name', 'run_id']):
        ans_dict = dict(zip(grp['question_num'].astype(str), grp['answer_content']))
        result = score_ctq_one_run(ans_dict)
        if result is None:
            records.append({"model_name": model, "run_id": rid,
                             **{d: "无效(测谎)" for d in dims}})
        else:
            records.append({"model_name": model, "run_id": rid, **result})

    df_scores = pd.DataFrame(records)

    # ==========================================
    # 打印每个模型的详细得分表
    # ==========================================
    for model, grp in df_scores.groupby('model_name'):
        print(f"\n{'='*65}")
        print(f"  模型：{model}")
        print(f"{'='*65}")

        # 表头
        header = f"  {'第N次':>5}  {'情感虐待':>8}  {'躯体虐待':>8}  {'性虐待':>6}  {'情感忽略':>8}  {'躯体忽视':>8}  {'CTQ总分':>7}"
        print(header)
        print(f"  {'-'*60}")

        # 每次run
        for i, (_, row) in enumerate(grp.iterrows(), start=1):
            def fmt(v):
                return f"{v:>8.1f}" if isinstance(v, float) else f"{str(v):>8}"
            print(f"  第{i:>2}次  "
                  f"{fmt(row['情感虐待'])}  "
                  f"{fmt(row['躯体虐待'])}  "
                  f"{fmt(row['性虐待']):>6}  "
                  f"{fmt(row['情感忽略'])}  "
                  f"{fmt(row['躯体忽视'])}  "
                  f"{fmt(row['CTQ总分']):>7}")

        print(f"  {'-'*60}")

        # 均值行（只对数值列计算）
        num_grp = grp[dims].apply(pd.to_numeric, errors='coerce')
        means = num_grp.mean()
        stds  = num_grp.std(ddof=1)

        def fmt_mean(col):
            m, s = means[col], stds[col]
            if np.isnan(m):
                return f"{'N/A':>8}"
            return f"{m:>6.1f}"

        print(f"  {'均值':>5}  "
              f"{fmt_mean('情感虐待'):>8}  "
              f"{fmt_mean('躯体虐待'):>8}  "
              f"{fmt_mean('性虐待'):>6}  "
              f"{fmt_mean('情感忽略'):>8}  "
              f"{fmt_mean('躯体忽视'):>8}  "
              f"{fmt_mean('CTQ总分'):>7}")

        print(f"  {'SD':>5}  "
              f"{stds['情感虐待']:>8.1f}  "
              f"{stds['躯体虐待']:>8.1f}  "
              f"{stds['性虐待']:>6.1f}  "
              f"{stds['情感忽略']:>8.1f}  "
              f"{stds['躯体忽视']:>8.1f}  "
              f"{stds['CTQ总分']:>7.1f}")

    # ==========================================
    # 导出 Excel
    # ==========================================
    # 宽表（每行一个run）
    df_scores.to_excel("CTQ详细得分表.xlsx", index=False)
    print(f"\n\n✅ 完整数据已导出至: CTQ详细得分表.xlsx")

if __name__ == "__main__":
    main()
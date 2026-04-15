import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re

matplotlib.rcParams['font.family'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from db import get_connection

CTQ_MAPPING = {
    "从没": 1, "从不": 1, "没有": 1,
    "偶尔": 2,
    "有时": 3,
    "经常": 4,
    "总是": 5
}

# CTQ 各维度切割分（标准临床切割分）
CTQ_CUTOFFS = {
    "情感虐待": 9,
    "躯体虐待": 8,
    "性虐待":   6,
    "情感忽略": 10,
    "躯体忽视": 8,
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
        return None

    return {
        "情感虐待": dim_sum([3, 8, 14, 18, 25]),
        "躯体虐待": dim_sum([9, 11, 12, 15, 17]),
        "性虐待":   dim_sum([20, 21, 23, 24, 27]),
        "情感忽略": dim_sum([], [5, 7, 13, 19, 28]),
        "躯体忽视": dim_sum([1, 4, 6], [2, 26]),
    }

def get_ctq_scores():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT r.model_name, r.run_id, a.question_num, a.answer_content
        FROM questionnaire_answers a
        JOIN ai_persona_runs r ON a.run_id = r.run_id
        WHERE a.questionnaire_name ILIKE '%CTQ%'
        ORDER BY r.model_name, r.run_id, a.question_num
    """, conn)
    conn.close()

    records = []
    for (model, rid), grp in df.groupby(['model_name', 'run_id']):
        ans_dict = dict(zip(grp['question_num'].astype(str), grp['answer_content']))
        result = score_ctq_one_run(ans_dict)
        if result is None:
            print(f"  ⚠️  {model} / run_id={rid} 未通过测谎，已排除")
            continue
        row = {"model_name": model, "run_id": rid}
        row.update(result)
        records.append(row)

    return pd.DataFrame(records)

def plot_single_model(model_name, grp):
    dims    = ["情感虐待", "躯体虐待", "性虐待", "情感忽略", "躯体忽视"]
    cutoffs = [CTQ_CUTOFFS[d] for d in dims]
    colors  = ["#E88080", "#80A8E8", "#80D8A0", "#F0C070", "#C0A0E8"]

    means = [grp[d].mean() for d in dims]
    stds  = [grp[d].std(ddof=1) for d in dims]
    x     = np.arange(len(dims))
    bar_w = 0.55

    fig, ax = plt.subplots(figsize=(9, 7))

    bars = ax.bar(x, means, width=bar_w, color=colors,
                  edgecolor='white', linewidth=0.8,
                  yerr=stds, capsize=6,
                  error_kw=dict(ecolor='#444444', elinewidth=1.4, capthick=1.4))

    # 数值标注
    for bar, mean_val, std_val in zip(bars, means, stds):
        if not np.isnan(mean_val):
            y_pos = mean_val + (std_val if not np.isnan(std_val) else 0) + 0.4
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{mean_val:.1f}",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 各维度切割分虚线
    for xi, (cutoff, color) in enumerate(zip(cutoffs, colors)):
        ax.hlines(cutoff,
                  xi - bar_w / 2, xi + bar_w / 2,
                  colors='red', linestyles='--', linewidth=1.5, alpha=0.8)

    # 图例
    ax.hlines([], 0, 0, colors='red', linestyles='--', linewidth=1.5,
              label='各维度临床切割分')
    ax.legend(fontsize=10, loc='upper right')

    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=12)
    ax.set_ylabel("维度总分", fontsize=12)
    ax.set_ylim(0, 30)
    ax.set_title(f"CTQ-28量表 {model_name} 5个维度总均分及标准差",
                 fontsize=13, fontweight='bold', pad=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 文件名去掉特殊字符，避免保存失败
    safe_name = re.sub(r'[\\/:*?"<>|]', '_', model_name)
    out_path  = f"CTQ_{safe_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ 已保存: {out_path}")

def main():
    print("📡 正在从数据库获取 CTQ 数据并计分...")
    df_scores = get_ctq_scores()

    if df_scores.empty:
        print("❌ 没有有效的 CTQ 数据，请检查数据库。")
        return

    print(f"\n✅ 有效记录数: {len(df_scores)} 条")
    print(df_scores.groupby('model_name').size().rename('有效run数').to_string())

    print("\n🎨 正在逐模型生成图表...")
    for model_name, grp in df_scores.groupby('model_name'):
        plot_single_model(model_name, grp)

    print("\n🎉 全部完成！共生成 6 张图。")

if __name__ == "__main__":
    main()
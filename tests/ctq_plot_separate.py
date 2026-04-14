import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re

# ==========================================
# 0. 中文字体设置
# ==========================================
matplotlib.rcParams['font.family'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import os
import sys
# 将 src 目录添加到路径中以便导入 db
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from db import get_connection

# ==========================================
# 3. 获取数据并计分
# ==========================================
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


# ==========================================
# 4. 单模型画图并保存
# ==========================================
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


# ==========================================
# 5. 主流程
# ==========================================
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
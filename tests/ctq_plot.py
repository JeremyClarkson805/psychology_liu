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
# 4. 从数据库获取数据并计分
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
# 5. 聚合：每个模型的各维度均值 + 标准差
# ==========================================
def aggregate(df_scores):
    dims = ["情感虐待", "躯体虐待", "性虐待", "情感忽略", "躯体忽视"]
    result = []
    for model, grp in df_scores.groupby('model_name'):
        row = {"model_name": model}
        for d in dims:
            row[f"{d}_mean"] = grp[d].mean()
            row[f"{d}_std"]  = grp[d].std(ddof=1)
            row[f"{d}_n"]    = grp[d].count()
        result.append(row)
    return pd.DataFrame(result)


# ==========================================
# 6. 画图：每个模型一张子图
# ==========================================
def plot_ctq(df_agg):
    dims     = ["情感虐待", "躯体虐待", "性虐待", "情感忽略", "躯体忽视"]
    cutoffs  = [CTQ_CUTOFFS[d] for d in dims]
    colors   = ["#E88080", "#80A8E8", "#80D8A0", "#F0C070", "#C0A0E8"]

    models   = df_agg['model_name'].tolist()
    n_models = len(models)

    # 每行3列
    ncols = 3
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(18, 7 * nrows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    x = np.arange(len(dims))
    bar_w = 0.55

    for idx, row in df_agg.iterrows():
        ax = axes[idx]
        means = [row[f"{d}_mean"] for d in dims]
        stds  = [row[f"{d}_std"]  for d in dims]

        bars = ax.bar(x, means, width=bar_w, color=colors,
                      edgecolor='white', linewidth=0.8,
                      yerr=stds, capsize=5,
                      error_kw=dict(ecolor='#444444', elinewidth=1.2, capthick=1.2))

        # 数值标注
        for bar, mean_val in zip(bars, means):
            if not np.isnan(mean_val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (max(stds) if not np.isnan(max(stds)) else 0) + 0.3,
                        f"{mean_val:.1f}",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 各维度切割分虚线（不同颜色同色）
        for xi, (cutoff, color) in enumerate(zip(cutoffs, colors)):
            ax.hlines(cutoff,
                      xi - bar_w / 2, xi + bar_w / 2,
                      colors='red', linestyles='--', linewidth=1.4, alpha=0.7)

        # 统一的切割分图例说明（只画一条代表性红虚线进图例）
        ax.hlines([], 0, 0, colors='red', linestyles='--', linewidth=1.4,
                  label='各维度临床切割分')
        ax.legend(fontsize=8, loc='upper right')

        ax.set_xticks(x)
        ax.set_xticklabels(dims, fontsize=11)
        ax.set_ylabel("维度总分", fontsize=11)
        ax.set_ylim(0, 30)
        ax.set_title(f"CTQ-28  {row['model_name']}", fontsize=12, fontweight='bold', pad=8)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 隐藏多余子图
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("CTQ-28 量表 — 6个模型各维度均值（均值 ± SD，红虚线为各维度临床切割分）",
                 fontsize=14, fontweight='bold', y=1.05)

    out_path = "CTQ_6模型柱状图.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图片已保存至: {out_path}")
    plt.show()


# ==========================================
# 7. 主流程
# ==========================================
def main():
    print("📡 正在从数据库获取 CTQ 数据并计分...")
    df_scores = get_ctq_scores()

    if df_scores.empty:
        print("❌ 没有有效的 CTQ 数据，请检查数据库。")
        return

    print(f"\n✅ 有效记录数: {len(df_scores)} 条")
    print(df_scores.groupby('model_name').size().rename('有效run数').to_string())

    df_agg = aggregate(df_scores)

    print("\n📊 各模型各维度均值：")
    dims = ["情感虐待", "躯体虐待", "性虐待", "情感忽略", "躯体忽视"]
    for _, row in df_agg.iterrows():
        print(f"\n  {row['model_name']}")
        for d in dims:
            print(f"    {d}: 均值={row[f'{d}_mean']:.2f}, SD={row[f'{d}_std']:.2f}")

    print("\n🎨 正在生成图表...")
    plot_ctq(df_agg)


if __name__ == "__main__":
    main()
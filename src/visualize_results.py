import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import get_connection
from analyze_data import parse_ctq, parse_erq, parse_brief, parse_nssi

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def fetch_results_from_db():
    """从数据库获取所有已评分数据"""
    conn = get_connection()
    try:
        # 获取所有答题明细
        query_ans = "SELECT run_id, questionnaire_name, question_num, answer_content FROM questionnaire_answers"
        df_ans = pd.read_sql(query_ans, conn)

        # 获取模型执行信息
        query_runs = "SELECT run_id, model_name FROM ai_persona_runs"
        df_runs = pd.read_sql(query_runs, conn)

        return df_ans, df_runs
    finally:
        conn.close()

def process_scores(df_ans, df_runs):
    """根据原始答案计算各维度得分"""
    run_ids = df_ans['run_id'].unique()
    scored_list = []

    for rid in run_ids:
        df_run = df_ans[df_ans['run_id'] == rid]
        model_name = df_runs[df_runs['run_id'] == rid]['model_name'].values[0]

        scores = {"Model": model_name, "RunID": rid}

        # 1. CTQ
        ctq_df = df_run[df_run['questionnaire_name'].str.contains("CTQ", na=False)]
        if not ctq_df.empty:
            ctq_s, valid = parse_ctq(ctq_df)
            if valid:
                scores.update(ctq_s)

        # 2. ERQ
        erq_df = df_run[df_run['questionnaire_name'].str.contains("情绪调节", na=False)]
        if not erq_df.empty:
            scores.update(parse_erq(erq_df))

        # 3. BRIEF
        brief_df = df_run[df_run['questionnaire_name'].str.contains("执行功能", na=False)]
        if not brief_df.empty:
            scores.update(parse_brief(brief_df))

        # 4. NSSI
        nssi_df = df_run[df_run['questionnaire_name'].str.contains("NSSI", na=False)]
        if not nssi_df.empty:
            scores.update(parse_nssi(nssi_df))

        scored_list.append(scores)

    return pd.DataFrame(scored_list)

# 维度名称中文映射表
LABEL_MAPPING = {
    "CTQ_EA": "情感虐待",
    "CTQ_PA": "躯体虐待",
    "CTQ_SA": "性虐待",
    "CTQ_EN": "情感忽略",
    "CTQ_PN": "躯体忽视",
    "CTQ_Total": "CTQ总分",
    "ERQ_CR": "认知重评",
    "ERQ_ES": "表达抑制",
    "ERQ_Total": "ERQ总分",
    "BRIEF_Total": "执行功能总分",
    "NSSI_Freq": "自伤频率"
}

def plot_model_comparison(df):
    """为每个维度组绘制并保存独立的对比图 (美化版)"""
    if df.empty:
        print("没有可用于绘图的数据。")
        return

    # 设置 Seaborn 主题风格
    sns.set_theme(style="whitegrid", font='SimHei')
    plt.rcParams['axes.unicode_minus'] = False

    # 定义要展示的维度组及其对应的文件名后缀
    dimension_groups = {
        "CTQ_童年创伤": (["CTQ_EA", "CTQ_PA", "CTQ_SA", "CTQ_EN", "CTQ_PN"], "ctq"),
        "ERQ_情绪调节": (["ERQ_CR", "ERQ_ES"], "erq"),
        "BRIEF_执行功能与NSSI_自伤": (["BRIEF_Total", "NSSI_Freq"], "brief_nssi")
    }

    # 确保输出目录存在
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取平均分
    avg_df = df.groupby("Model").mean(numeric_only=True).reset_index()

    for group_name, (dims, file_suffix) in dimension_groups.items():
        # 准备绘图用的长表数据
        available_dims = [d for d in dims if d in avg_df.columns]
        if not available_dims:
            continue

        plot_data = avg_df.melt(id_vars="Model", value_vars=available_dims,
                               var_name="Dimension", value_name="Score")

        # 将维度名称替换为中文
        plot_data["Dimension"] = plot_data["Dimension"].map(lambda x: LABEL_MAPPING.get(x, x))

        # 创建画布，增加高度以适应图例
        plt.figure(figsize=(12, 7))

        # 恢复最初的配色方案 (默认调色板)
        ax = sns.barplot(
            data=plot_data,
            x="Dimension",
            y="Score",
            hue="Model",
            edgecolor=".2"
        )

        # 在柱状图上方添加数值标签
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9, fontweight='bold')

        # 美化标题和轴标签
        plt.title(f"模型心理维度对比: {group_name.replace('_', ' ')}", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("评估维度", fontsize=13, labelpad=10)
        plt.ylabel("平均得分 (分值越高程度越重)", fontsize=13, labelpad=10)

        # 优化图例位置
        plt.legend(title="AI 模型", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

        # 移除顶部和右侧边框，使画面更清爽
        sns.despine(offset=10, trim=True)

        # 保存独立图片
        file_path = os.path.join(output_dir, f"model_comparison_{file_suffix}.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"美化图表已生成: {file_path}")

def main():
    print("正在从数据库加载数据...")
    df_ans, df_runs = fetch_results_from_db()

    print("正在计算评分维度...")
    scored_df = process_scores(df_ans, df_runs)
    print(scored_df.shape)
    print(scored_df.isnull().sum())
    print(scored_df[scored_df.isnull().any(axis=1)][['Model', 'RunID']].to_string())

    print("正在生成可视化对比图...")
    plot_model_comparison(scored_df)

if __name__ == "__main__":
    main()

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
import re

# DB Configuration
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "REMOVED_PASSWORD",
    "host": "localhost",
    "port": "5432"
}

def get_db_data():
    """从数据库获取答题原始数据"""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        # 获取所有答题明细
        query_ans = """
            SELECT run_id, questionnaire_name, question_num, answer_content 
            FROM questionnaire_answers
        """
        df_ans = pd.read_sql(query_ans, conn)
        
        # 获取模型和人格信息
        query_runs = "SELECT run_id, model_name, persona_prompt FROM ai_persona_runs"
        df_runs = pd.read_sql(query_runs, conn)
        
        return df_ans, df_runs
    finally:
        conn.close()

def get_score_from_text(text, mapping):
    """从文本中鲁棒地提取分值"""
    if not text:
        return np.nan
    text = str(text).strip()

    # 1. 尝试精确匹配
    if text in mapping:
        return mapping[text]

    # 2. 尝试正则提取括号中的数字或纯数字
    num_match = re.search(r'(\d+)', text)
    if num_match:
        val = int(num_match.group(1))
        # 验证提取的数字是否在合理范围内 (根据 mapping 的值判断)
        if val in mapping.values():
            return val

    # 3. 尝试关键词模糊匹配
    for key, val in mapping.items():
        if key in text:
            return val

    return np.nan

def parse_ctq(df_run):
    """
    CTQ-SF 儿童创伤问卷 评分
    映射：从没=1, 偶尔=2, 有时=3, 经常=4, 总是=5
    """
    mapping = {"从没": 1, "偶尔": 2, "有时": 3, "经常": 4, "总是": 5}
    q_map = {row['question_num']: get_score_from_text(row['answer_content'], mapping) 
             for _, row in df_run.iterrows()}

    # 效度/诚实性检查
    is_valid = True
    if q_map.get('10') == 1: is_valid = False
    if q_map.get('16') in [4, 5]: is_valid = False
    if q_map.get('22') in [4, 5]: is_valid = False

    if not is_valid:
        return None, False

    # 维度计分 (使用 np.nanmean 处理可能缺失的题目)
    def get_v(q): return q_map.get(str(q), np.nan)
    def rev(q): 
        v = get_v(q)
        return 6 - v if not np.isnan(v) else np.nan

    def calc_dim(q_list, rev_list=[]):
        vals = [get_v(q) for q in q_list] + [rev(q) for q in rev_list]
        valid_vals = [v for v in vals if not np.isnan(v)]
        if not valid_vals: return np.nan
        # 如果有缺失题，按比例折算 (即取平均值再乘以总题数)
        return np.mean(valid_vals) * len(q_list + rev_list)

    # 1) 情感虐待：3, 8, 14, 18, 25
    ea = calc_dim([3, 8, 14, 18, 25])
    # 2) 躯体虐待：9, 11, 12, 15, 17
    pa = calc_dim([9, 11, 12, 15, 17])
    # 3) 性虐待：20, 21, 23, 24, 27
    sa = calc_dim([20, 21, 23, 24, 27])
    # 4) 情感忽略 (反向)：5, 7, 13, 19, 28
    en = calc_dim([], [5, 7, 13, 19, 28])
    # 5) 躯体忽视：1, 4, 6, 及 2(反向), 26(反向)
    pn = calc_dim([1, 4, 6], [2, 26])

    total = sum([v for v in [ea, pa, sa, en, pn] if not np.isnan(v)])
    scores = {"CTQ_Total": total, "CTQ_EA": ea, "CTQ_PA": pa, "CTQ_SA": sa, "CTQ_EN": en, "CTQ_PN": pn}
    return scores, True

def parse_erq(df_run):
    """
    ERQ 情绪调节问卷
    认知重评：1, 3, 5, 7, 8, 10
    表达抑制：2, 4, 6, 9
    """
    mapping = {"非常不同意": 1, "比较不同意": 2, "有点不同意": 3, "中立": 4, "有点同意": 5, "比较同意": 6, "非常同意": 7}
    q_map = {row['question_num']: get_score_from_text(row['answer_content'], mapping) 
             for _, row in df_run.iterrows()}

    def calc_dim(q_list):
        vals = [q_map.get(str(i), np.nan) for i in q_list]
        valid_vals = [v for v in vals if not np.isnan(v)]
        if not valid_vals: return np.nan
        return np.mean(valid_vals) * len(q_list)

    cr = calc_dim([1, 3, 5, 7, 8, 10])
    es = calc_dim([2, 4, 6, 9])

    total = (cr if not np.isnan(cr) else 0) + (es if not np.isnan(es) else 0)
    return {"ERQ_CR": cr, "ERQ_ES": es, "ERQ_Total": total if total > 0 else np.nan}


def parse_brief(df_run):
    """BRIEF-A 执行功能量表 (1-3分，总分越高受损越重)"""
    mapping = {"没有": 1, "有时": 2, "经常": 3}
    q_map = {row['question_num']: mapping.get(row['answer_content'], 1) 
             for _, row in df_run.iterrows()}
    
    total = sum(q_map.values())
    return {"BRIEF_Total": total}

def parse_nssi(df_run):
    """
    NSSI 自伤行为 (采用频率累加法作为连续变量)
    规则：提取 '有X次' 中的 X 进行求和。
    只取最近1年的记录 (question_num 以后缀 'b' 结尾)
    """
    def extract_freq(text):
        if not text or "没有" in text:
            return 0
        match = re.search(r'(\d+)', text)
        return int(match.group(1)) if match else 0

    # 过滤最近一年的题目 (1b, 2b ... 9b)
    df_recent = df_run[df_run['question_num'].str.endswith('b')]
    total_freq = df_recent['answer_content'].apply(extract_freq).sum()
    
    return {"NSSI_Freq": total_freq}

def run_sem_analysis(df):
    """
    路径分析 (使用 semopy 或 pingouin)
    进行数据清理，确保没有 NaN
    """
    # 数据清理：路径分析必须要求所有相关列都有值
    cols = ['CTQ_Total', 'ERQ_Total', 'BRIEF_Total', 'NSSI_Freq']
    clean_df = df.dropna(subset=cols)
    
    if len(clean_df) < 5:
        print(f"\n[!] 样本量过小 ({len(clean_df)})，跳过路径分析。")
        return None

    try:
        from semopy import Model
        print("\n=== SEM 路径分析 (semopy) ===")
        desc = """
        BRIEF_Total ~ CTQ_Total
        ERQ_Total ~ CTQ_Total
        NSSI_Freq ~ BRIEF_Total + ERQ_Total
        """
        model = Model(desc)
        model.fit(clean_df)
        estimates = model.inspect()
        print(estimates[estimates['op'] == '~'][['lval', 'op', 'rval', 'Estimate', 'p-value']])
        return estimates
    except Exception as e:
        print(f"\n[!] semopy 运行失败: {e}，尝试 pingouin...")
        try:
            import pingouin as pg
            print("\n=== 路径分析 (pingouin) ===")
            for target, predictors in [
                ('BRIEF_Total', ['CTQ_Total']),
                ('ERQ_Total', ['CTQ_Total']),
                ('NSSI_Freq', ['BRIEF_Total', 'ERQ_Total'])
            ]:
                lm = pg.linear_regression(clean_df[predictors], clean_df[target])
                print(f"\nPath to {target}:")
                print(lm[['names', 'coef', 'pval']])
        except Exception as e2:
            print(f"[!] 统计库调用失败: {e2}")

def calculate_similarity(df):
    """计算相似度指标 (MAE, ICC, Cosine)"""
    cols = ['CTQ_Total', 'ERQ_Total', 'BRIEF_Total', 'NSSI_Freq']
    # 仅针对完整数据计算
    full_df = df.dropna(subset=cols)
    
    if full_df.empty:
        print("\n[!] 没有完整的模型数据可用于相似度分析。")
        return

    print(f"\n=== 模型一致性分析 (有效样本: {len(full_df)}) ===")
    avg_scores = full_df[cols].mean()
    
    results = []
    for _, row in full_df.iterrows():
        vec = row[cols].values.astype(float)
        ref = avg_scores.values.astype(float)
        
        mae = np.mean(np.abs(vec - ref))
        cos_sim = np.dot(vec, ref) / (np.linalg.norm(vec) * np.linalg.norm(ref))
        
        results.append({
            "run_id": row['run_id'],
            "model": row['model_name'],
            "MAE": round(mae, 3),
            "Cosine": round(cos_sim, 4)
        })
    
    sim_df = pd.DataFrame(results)
    print(sim_df.to_string(index=False))

def main():
    # 1. 获取数据
    df_ans, df_runs = get_db_data()
    run_ids = df_ans['run_id'].unique()
    
    # 2. 逐个运行评分
    scored_data = []
    invalid_count = 0
    
    for rid in run_ids:
        df_run = df_ans[df_ans['run_id'] == rid]
        
        # CTQ & 效度检查
        ctq_df = df_run[df_run['questionnaire_name'].str.contains("CTQ")]
        ctq_scores, is_valid = parse_ctq(ctq_df)
        if not is_valid:
            invalid_count += 1
            continue
            
        row = {"run_id": rid}
        row.update(ctq_scores)
        
        # ERQ
        erq_df = df_run[df_run['questionnaire_name'].str.contains("情绪调节")]
        row.update(parse_erq(erq_df))
        
        # BRIEF
        brief_df = df_run[df_run['questionnaire_name'].str.contains("执行功能")]
        row.update(parse_brief(brief_df))
        
        # NSSI
        nssi_df = df_run[df_run['questionnaire_name'].str.contains("NSSI")]
        row.update(parse_nssi(nssi_df))
        
        scored_data.append(row)
    
    # 合并模型信息
    final_df = pd.DataFrame(scored_data)
    final_df = final_df.merge(df_runs, on='run_id')
    
    print(f"有效样本数: {len(final_df)} (剔除无效/欺骗回答: {invalid_count})")
    
    if len(final_df) < 2:
        print("样本量不足，无法进行统计分析。")
        return

    # 3. 统计分析
    run_sem_analysis(final_df)
    calculate_similarity(final_df)

if __name__ == "__main__":
    main()

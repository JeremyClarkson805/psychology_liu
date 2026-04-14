import psycopg2
import pandas as pd
import numpy as np
import re

# ==========================================
# 1. 数据库配置与数据获取
# ==========================================
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "REMOVED_PASSWORD",
    "host": "localhost",
    "port": "5432"
}

def get_db_data():
    """从数据库获取所有答题原始数据"""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        query_ans = """
            SELECT run_id, questionnaire_name, question_num, answer_content 
            FROM questionnaire_answers
        """
        df_ans = pd.read_sql(query_ans, conn)
        return df_ans
    finally:
        conn.close()

# ==========================================
# 2. 基础文本转数字工具
# ==========================================
def get_numeric_value(text, mapping):
    """通用文本转分值函数"""
    if pd.isna(text) or not str(text).strip():
        return np.nan
    text = str(text).strip()

    # 精确匹配
    if text in mapping:
        return mapping[text]

    # 包含匹配（优先长key，避免歧义）
    for key in sorted(mapping.keys(), key=len, reverse=True):
        if key in text:
            return mapping[key]

    # 纯数字兜底
    num_match = re.search(r'(\d+)', text)
    if num_match:
        val = int(num_match.group(1))
        if val in mapping.values():
            return val

    return np.nan


def df_to_dict(df):
    """将DataFrame的题号和答案转为字典格式，方便计分"""
    return dict(zip(df['question_num'].astype(str), df['answer_content']))


# ==========================================
# 3. 各量表计分逻辑（已修正）
# ==========================================

# ------------------------------------------
# 3.1 CTQ-SF 儿童创伤问卷
# ------------------------------------------
def score_ctq(answers_dict):
    """
    CTQ-SF 计分规则：
    - 5点计分（从不=1 ~ 总是=5）
    - 反向题公式：6 - 原始分
    - 各维度5题直接求和（范围5~25）
    - 效度检查：
        第10题=1(从不) → 问卷无效
        第16或22题=4或5  → 问卷无效
    - 每个维度得分 > 5 判定为存在该类创伤（可按需修改切割分）
    """
    mapping = {
        "从没": 1, "从不": 1, "没有": 1,
        "偶尔": 2, "有时": 3, "经常": 4, "总是": 5
    }
    q_map = {str(k): get_numeric_value(v, mapping) for k, v in answers_dict.items()}

    def get_v(q):
        return q_map.get(str(q), np.nan)

    def rev(q):
        v = get_v(q)
        return 6 - v if not np.isnan(v) else np.nan

    # ── 效度检查 ──────────────────────────────────────────────
    q10 = get_v('10')
    q16 = get_v('16')
    q22 = get_v('22')
    if (not np.isnan(q10) and q10 == 1) or \
       (not np.isnan(q16) and q16 in [4, 5]) or \
       (not np.isnan(q22) and q22 in [4, 5]):
        return {"儿童创伤_问卷状态": "无效(未通过测谎)"}

    # ── 各维度直接求和（修正：不再用均值×题数，改为直接 sum）──
    def calc_dim(pos_qs, rev_qs=[]):
        """
        pos_qs: 正向题题号列表
        rev_qs: 反向题题号列表
        返回：有效题得分之和；若全部缺失则返回 nan
        """
        vals = [get_v(q) for q in pos_qs] + [rev(q) for q in rev_qs]
        valid_vals = [v for v in vals if not np.isnan(v)]
        return sum(valid_vals) if valid_vals else np.nan

    # 情感虐待：3,8,14,18,25（全正向）
    ea = calc_dim([3, 8, 14, 18, 25])
    # 躯体虐待：9,11,12,15,17（全正向）
    pa = calc_dim([9, 11, 12, 15, 17])
    # 性虐待：20,21,23,24,27（全正向）
    sa = calc_dim([20, 21, 23, 24, 27])
    # 情感忽略：5,7,13,19,28（全反向）
    en = calc_dim([], [5, 7, 13, 19, 28])
    # 躯体忽视：1,4,6（正向） + 2,26（反向）
    pn = calc_dim([1, 4, 6], [2, 26])

    dims = [ea, pa, sa, en, pn]
    total = sum(v for v in dims if not np.isnan(v))

    # ── 切割分：各维度 > 5 判定有创伤 ──────────────────────────
    # 注意：量表原始切割分通常为 EA>=8, PA>=8, SA>=6, EN>=10, PN>=8
    # 此处保留原始逻辑 >5，如需更改请修改 CUTOFF 值
    CUTOFF = 5

    return {
        "儿童创伤_问卷状态":             "有效",
        "儿童创伤_总分":                 total,
        "儿童创伤_情感虐待_得分":         ea,
        "儿童创伤_情感虐待_判定(1有0无)":  int(not np.isnan(ea) and ea > CUTOFF),
        "儿童创伤_躯体虐待_得分":         pa,
        "儿童创伤_躯体虐待_判定(1有0无)":  int(not np.isnan(pa) and pa > CUTOFF),
        "儿童创伤_性虐待_得分":           sa,
        "儿童创伤_性虐待_判定(1有0无)":    int(not np.isnan(sa) and sa > CUTOFF),
        "儿童创伤_情感忽略_得分":         en,
        "儿童创伤_情感忽略_判定(1有0无)":  int(not np.isnan(en) and en > CUTOFF),
        "儿童创伤_躯体忽视_得分":         pn,
        "儿童创伤_躯体忽视_判定(1有0无)":  int(not np.isnan(pn) and pn > CUTOFF),
    }


# ------------------------------------------
# 3.2 ERQ 情绪调节问卷
# ------------------------------------------
def score_erq(answers_dict):
    """
    ERQ 计分规则：
    - 7点计分（非常不同意=1 ~ 非常同意=7）
    - 认知重评（CR）：题目 1,3,5,7,8,10（6题求和，范围6~42）
    - 表达抑制（ES）：题目 2,4,6,9（4题求和，范围4~28）
    - 中位数分组在 main() 中完成
    """
    mapping = {
        "非常不同意": 1, "很不同意": 1,
        "比较不同意": 2, "有点不同意": 3,
        "中立": 4, "不确定": 4, "一般": 4,
        "有点同意": 5, "比较同意": 6,
        "非常同意": 7, "很同意": 7
    }
    q_map = {str(k): get_numeric_value(v, mapping) for k, v in answers_dict.items()}

    def calc_dim(q_list):
        vals = [q_map.get(str(i), np.nan) for i in q_list]
        valid_vals = [v for v in vals if not np.isnan(v)]
        return sum(valid_vals) if valid_vals else np.nan

    cr = calc_dim([1, 3, 5, 7, 8, 10])   # 认知重评
    es = calc_dim([2, 4, 6, 9])           # 表达抑制

    return {
        "情绪调节_认知重评_得分": cr,
        "情绪调节_表达抑制_得分": es,
    }


# ------------------------------------------
# 3.3 BRIEF-A 执行功能量表
# ------------------------------------------

# BRIEF-A 常模 T 分换算表（成人版，自评，示例占位）
# ⚠️ 实际使用时请替换为对应年龄段/性别的官方常模
# 格式：{原始总分: T分}，支持线性插值
# 设为 None 则只输出原始分，不进行T分换算
BRIEF_NORM = None
# 示例（请替换为真实常模数据）：
# BRIEF_NORM = {75: 40, 85: 45, 95: 50, 105: 55, 112: 60, 118: 65, 125: 70}

def raw_to_t_score(raw_score, norm_dict):
    """
    将原始分转换为T分（含线性插值）。
    norm_dict：{原始分: T分} 字典
    若原始分超出常模范围，则返回边界T分。
    """
    if norm_dict is None or np.isnan(raw_score):
        return np.nan
    keys = sorted(norm_dict.keys())
    if raw_score <= keys[0]:
        return norm_dict[keys[0]]
    if raw_score >= keys[-1]:
        return norm_dict[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= raw_score <= hi:
            t_lo, t_hi = norm_dict[lo], norm_dict[hi]
            ratio = (raw_score - lo) / (hi - lo)
            return round(t_lo + ratio * (t_hi - t_lo), 1)
    return np.nan


def score_brief(answers_dict):
    """
    BRIEF-A 计分规则：
    - 3点计分（从不=1, 有时=2, 经常=3）
    - 75题求和，原始分越高表示执行功能受损越严重
    - 若配置了 BRIEF_NORM 常模，则换算T分并给出临床判定：
        T >= 65：临床显著受损
        T 60~64：边缘/轻度困难
        T <= 59：正常范围
    """
    mapping = {
        "从不": 1, "没有": 1, "不": 1,
        "有时": 2, "偶尔": 2,
        "经常": 3, "总是": 3, "一直": 3
    }
    q_map = {str(k): get_numeric_value(v, mapping) for k, v in answers_dict.items()}
    valid_scores = [v for v in q_map.values() if not np.isnan(v)]

    if not valid_scores:
        return {
            "执行功能_原始总分": np.nan,
            "执行功能_T分":      np.nan,
            "执行功能_临床判定": np.nan,
        }

    raw = sum(valid_scores)
    t_score = raw_to_t_score(raw, BRIEF_NORM)

    if not np.isnan(t_score):
        if t_score >= 65:
            clinical = "临床显著受损(T>=65)"
        elif t_score >= 60:
            clinical = "边缘/轻度困难(T=60~64)"
        else:
            clinical = "正常范围(T<=59)"
    else:
        clinical = "无常模，无法判定（仅供参考原始分）"

    return {
        "执行功能_原始总分": raw,
        "执行功能_T分":      t_score,
        "执行功能_临床判定": clinical,
    }


# ------------------------------------------
# 3.4 NSSI 非自杀性自伤行为量表
# ------------------------------------------
def score_nssi(answers_dict):
    """
    NSSI 计分规则（You, Leung & Fu, 2012 中文版）：
    - 4点计分：0次=1, 1~2次=2, 3~5次=3, 6次及以上=4
    - 12题求和，范围 12~48
    - 总分 > 12 判定为存在自伤行为
      （12分即全部选"0次"，>12表示至少有1次自伤）
    """
    mapping = {
        "0次": 1, "没有": 1, "从不": 1, "无": 1,
        "1～2次": 2, "1~2次": 2, "1-2次": 2, "1到2次": 2,
        "3～5次": 3, "3~5次": 3, "3-5次": 3, "3到5次": 3,
        "6次及以上": 4, "6次以上": 4, "6次或以上": 4,
        "6次": 4, "以上": 4,
    }
    q_map = {str(k): get_numeric_value(v, mapping) for k, v in answers_dict.items()}
    valid_scores = [v for v in q_map.values() if not np.isnan(v)]

    if not valid_scores:
        return {
            "自伤行为_总分":             np.nan,
            "自伤行为_危险判定(1有0无)":  np.nan,
        }

    total = sum(valid_scores)
    return {
        "自伤行为_总分":             total,
        "自伤行为_危险判定(1有0无)":  1 if total > 12 else 0,
    }


# ==========================================
# 4. 主流程
# ==========================================
def main():
    print("正在连接数据库并获取数据...")
    df_ans = get_db_data()
    run_ids = df_ans['run_id'].unique()
    print(f"共获取到 {len(run_ids)} 个答题人员(run_id)的作答记录。\n")

    all_scores = []

    for rid in run_ids:
        df_run = df_ans[df_ans['run_id'] == rid]
        person_score = {"受测者ID(run_id)": rid}

        print(f"========== [ 受测者ID: {rid} 的问卷得分 ] ==========")

        # ── CTQ 儿童创伤 ──────────────────────────────────────
        ctq_df = df_run[df_run['questionnaire_name'].str.contains("CTQ", case=False, na=False)]
        if not ctq_df.empty:
            res = score_ctq(df_to_dict(ctq_df))
            person_score.update(res)
            status = res.get('儿童创伤_问卷状态')
            if status == "有效":
                print(f"【儿童创伤】 状态: {status} | 总分: {res.get('儿童创伤_总分')}")
                print(f"           情感虐待: {res.get('儿童创伤_情感虐待_得分')} | "
                      f"躯体虐待: {res.get('儿童创伤_躯体虐待_得分')} | "
                      f"性虐待: {res.get('儿童创伤_性虐待_得分')} | "
                      f"情感忽略: {res.get('儿童创伤_情感忽略_得分')} | "
                      f"躯体忽视: {res.get('儿童创伤_躯体忽视_得分')}")
            else:
                print(f"【儿童创伤】 状态: {status}")
        else:
            print("【儿童创伤】 缺失")

        # ── ERQ 情绪调节 ──────────────────────────────────────
        erq_df = df_run[df_run['questionnaire_name'].str.contains("情绪调节|ERQ", case=False, na=False)]
        if not erq_df.empty:
            res = score_erq(df_to_dict(erq_df))
            person_score.update(res)
            print(f"【情绪调节】 认知重评: {res.get('情绪调节_认知重评_得分')}分 | "
                  f"表达抑制: {res.get('情绪调节_表达抑制_得分')}分")
        else:
            print("【情绪调节】 缺失")

        # ── BRIEF 执行功能 ────────────────────────────────────
        brief_df = df_run[df_run['questionnaire_name'].str.contains("执行功能|BRIEF", case=False, na=False)]
        if not brief_df.empty:
            res = score_brief(df_to_dict(brief_df))
            person_score.update(res)
            print(f"【执行功能】 原始总分: {res.get('执行功能_原始总分')} | "
                  f"T分: {res.get('执行功能_T分')} | "
                  f"判定: {res.get('执行功能_临床判定')}")
        else:
            print("【执行功能】 缺失")

        # ── NSSI 自伤行为 ─────────────────────────────────────
        nssi_df = df_run[df_run['questionnaire_name'].str.contains("NSSI|自伤", case=False, na=False)]
        if not nssi_df.empty:
            res = score_nssi(df_to_dict(nssi_df))
            person_score.update(res)
            risk = res.get('自伤行为_危险判定(1有0无)')
            risk_text = "有自伤行为" if risk == 1 else "无自伤行为"
            print(f"【自伤行为】 总分: {res.get('自伤行为_总分')} | 判定: {risk_text}")
        else:
            print("【自伤行为】 缺失")

        print("-" * 55)
        all_scores.append(person_score)

    # ── 后处理：ERQ 中位数高低分组 ────────────────────────────
    final_df = pd.DataFrame(all_scores)

    if not final_df.empty:
        for col, label in [
            ('情绪调节_认知重评_得分', '情绪调节_认知重评_高低分组'),
            ('情绪调节_表达抑制_得分', '情绪调节_表达抑制_高低分组'),
        ]:
            if col in final_df.columns:
                median_val = final_df[col].median()
                final_df[label] = final_df[col].apply(
                    lambda x: '高分组' if pd.notna(x) and x >= median_val
                    else ('低分组' if pd.notna(x) else np.nan)
                )

        export_path = "所有问卷最终得分表.xlsx"
        final_df.to_excel(export_path, index=False)
        print(f"\n所有 {len(final_df)} 条记录已导出至: {export_path}")

    return final_df


if __name__ == "__main__":
    main()
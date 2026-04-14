import psycopg2
import pandas as pd

DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "REMOVED_PASSWORD",
    "host": "localhost",
    "port": "5432"
}

conn = psycopg2.connect(**DB_CONFIG)

# 查1：所有模型名称和每个模型的run数量
df_models = pd.read_sql("""
    SELECT model_name, COUNT(*) as run_count 
    FROM ai_persona_runs 
    GROUP BY model_name 
    ORDER BY model_name
""", conn)
print("=== 模型列表 ===")
print(df_models.to_string())

# 查2：所有问卷名称（确认CTQ的名字）
df_q = pd.read_sql("SELECT DISTINCT questionnaire_name FROM questionnaire_answers", conn)
print("\n=== 问卷名称 ===")
print(df_q.to_string())

# 查3：CTQ原始答题数据（全部）
df_ctq = pd.read_sql("""
    SELECT r.model_name, r.run_id, a.question_num, a.answer_content
    FROM questionnaire_answers a
    JOIN ai_persona_runs r ON a.run_id = r.run_id
    WHERE a.questionnaire_name ILIKE '%CTQ%'
    ORDER BY r.model_name, r.run_id, a.question_num
""", conn)
print("\n=== CTQ原始数据 ===")
print(df_ctq.to_string())

conn.close()
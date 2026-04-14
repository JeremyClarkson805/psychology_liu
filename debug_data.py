from src.db import get_connection
import pandas as pd

def debug():
    conn = get_connection()
    try:
        # 查询 GPT 模型的所有回答，看看它们长什么样
        query = """
            SELECT model_name, questionnaire_name, question_num, answer_content 
            FROM questionnaire_answers 
            JOIN ai_persona_runs ON questionnaire_answers.run_id = ai_persona_runs.run_id 
            WHERE model_name LIKE '%gpt%'
            LIMIT 50
        """
        df = pd.read_sql(query, conn)
        print("=== GPT 原始回答示例 ===")
        print(df.to_string())
    finally:
        conn.close()

if __name__ == "__main__":
    debug()

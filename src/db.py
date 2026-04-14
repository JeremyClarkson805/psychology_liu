import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def get_connection():
    """获取数据库连接"""
    return psycopg2.connect(**DB_CONFIG)

def init_db():
    """初始化数据库表"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # 创建 AI 执行记录表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_persona_runs (
                    run_id SERIAL PRIMARY KEY,
                    persona_prompt TEXT NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # 创建问卷答题明细表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS questionnaire_answers (
                    answer_id SERIAL PRIMARY KEY,
                    run_id INTEGER REFERENCES ai_persona_runs(run_id) ON DELETE CASCADE,
                    questionnaire_name VARCHAR(100) NOT NULL,
                    question_num VARCHAR(20) NOT NULL,
                    question_text TEXT NOT NULL,
                    answer_content VARCHAR(255),
                    ai_reasoning TEXT
                );
            """)
            conn.commit()
            print("Database initialized successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error initializing DB: {e}")
    finally:
        conn.close()

def create_run(persona_prompt: str, model_name: str) -> int:
    """创建一次新的执行记录，返回 run_id"""
    conn = get_connection()
    run_id = None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ai_persona_runs (persona_prompt, model_name) VALUES (%s, %s) RETURNING run_id",
                (persona_prompt, model_name)
            )
            run_id = cur.fetchone()[0]
            conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error creating run: {e}")
    finally:
        conn.close()
    return run_id

def insert_answer(run_id: int, q_name: str, q_num: str, q_text: str, answer_content: str, ai_reasoning: str):
    """插入一条答题记录"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO questionnaire_answers 
                (run_id, questionnaire_name, question_num, question_text, answer_content, ai_reasoning)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (run_id, q_name, q_num, q_text, answer_content, ai_reasoning))
            conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error inserting answer: {e}")
    finally:
        conn.close()

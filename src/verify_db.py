import psycopg2
import json

DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "REMOVED_PASSWORD",
    "host": "localhost",
    "port": "5432"
}

def verify_db():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            # Check runs
            print("--- Runs ---")
            cur.execute("SELECT * FROM ai_persona_runs ORDER BY run_id DESC LIMIT 1;")
            run = cur.fetchone()
            print(f"Latest Run: ID={run[0]}, Created={run[2]}")
            
            # Check answers
            print("\n--- Answer Sample (First 5) ---")
            cur.execute("""
                SELECT questionnaire_name, question_num, answer_content, ai_reasoning 
                FROM questionnaire_answers 
                WHERE run_id = %s 
                ORDER BY answer_id ASC LIMIT 5;
            """, (run[0],))
            answers = cur.fetchall()
            for ans in answers:
                print(f"[{ans[0]} Q{ans[1]}] Answer: {ans[2]} | Reason: {ans[3]}")
                
            # Count answers
            cur.execute("SELECT COUNT(*) FROM questionnaire_answers WHERE run_id = %s;", (run[0],))
            count = cur.fetchone()[0]
            print(f"\nTotal Answers for Run {run[0]}: {count}")
            
    except Exception as e:
        print(f"Error checking DB: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    verify_db()

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import get_connection, init_db


def db_tables_exist(conn) -> bool:
    """检查数据库表是否存在"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'ai_persona_runs'
                );
            """)
            return cur.fetchone()[0]
    except Exception:
        return False


def verify_db():
    conn = get_connection()
    try:
        # 检查数据库表是否存在，如果不存在则初始化
        if not db_tables_exist(conn):
            print("Database tables not found. Initializing...")
            init_db()

        # 重新连接确保初始化完成
        conn = get_connection()

        with conn.cursor() as cur:
            # Check runs
            print("--- Runs ---")
            cur.execute(
                "SELECT run_id, model_name, created_at FROM ai_persona_runs ORDER BY run_id DESC LIMIT 1;"
            )
            run = cur.fetchone()

            if not run:
                print(
                    "No runs found in the database. (This is normal for a fresh initialization)"
                )
            else:
                print(f"Latest Run: ID={run[0]}, Model={run[1]}, Created={run[2]}")

                # Check answers
                print("\n--- Answer Sample (First 5) ---")
                cur.execute(
                    """
                    SELECT questionnaire_name, question_num, answer_content, ai_reasoning 
                    FROM questionnaire_answers 
                    WHERE run_id = %s 
                    ORDER BY answer_id ASC LIMIT 5;
                """,
                    (run[0],),
                )
                answers = cur.fetchall()

                if not answers:
                    print("No answers found for this run.")
                else:
                    for ans in answers:
                        print(
                            f"[{ans[0]} Q{ans[1]}] Answer: {ans[2]} | Reason: {ans[3]}"
                        )

                # Count answers
                cur.execute(
                    "SELECT COUNT(*) FROM questionnaire_answers WHERE run_id = %s;",
                    (run[0],),
                )
                count = cur.fetchone()[0]
                print(f"\nTotal Answers for Run {run[0]}: {count}")

        print("\n--- Database verification complete ---")

    except Exception as e:
        print(f"Error checking DB: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    verify_db()

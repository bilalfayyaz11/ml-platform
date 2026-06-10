import json
import time
import psycopg2
import psycopg2.extras


class PredictionLogger:
    def __init__(self):
        self.conn = None

    def connect(self, dsn: str) -> None:
        deadline = time.time() + 30
        last_error = None

        while time.time() < deadline:
            try:
                self.conn = psycopg2.connect(dsn)
                self.conn.autocommit = False
                return
            except Exception as exc:
                last_error = exc
                time.sleep(2)

        raise ConnectionError(f"Database unreachable after 30 seconds: {last_error}")

    def log_prediction(self, features: list, prediction: int, confidence: float) -> int:
        if self.conn is None:
            raise ConnectionError("Database connection is not initialized")

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions (features, prediction, confidence)
                VALUES (%s, %s, %s)
                RETURNING id;
                """,
                (json.dumps(features), prediction, confidence),
            )
            row_id = cur.fetchone()[0]

        self.conn.commit()
        return row_id

    def fetch_recent(self, limit: int) -> list[dict]:
        if self.conn is None:
            raise ConnectionError("Database connection is not initialized")

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, features, prediction, confidence, created_at
                FROM predictions
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()

        return [dict(row) for row in rows]

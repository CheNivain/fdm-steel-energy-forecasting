"""Small local SQLite store. SQL values are always parameterized."""

import json
import sqlite3
from pathlib import Path


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS forecasts (id TEXT PRIMARY KEY, created TEXT NOT NULL, body TEXT NOT NULL);
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def save_dataset(self, body):
        with self.connect() as db:
            count = db.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
            if count >= 20:
                raise ValueError("The local workspace holds 20 datasets. Remove an old dataset before importing another.")
            db.execute("INSERT INTO datasets VALUES (?, ?)", (body["id"], json.dumps(body, allow_nan=False)))

    def datasets(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT body FROM datasets ORDER BY rowid DESC")]

    def dataset(self, identifier):
        with self.connect() as db:
            row = db.execute("SELECT body FROM datasets WHERE id = ?", (identifier,)).fetchone()
            return json.loads(row[0]) if row else None

    def delete_dataset(self, identifier):
        with self.connect() as db:
            return db.execute("DELETE FROM datasets WHERE id = ?", (identifier,)).rowcount > 0

    def update_dataset(self, body):
        with self.connect() as db:
            db.execute("UPDATE datasets SET body = ? WHERE id = ?", (json.dumps(body, allow_nan=False), body["id"]))

    def save_forecast(self, body):
        with self.connect() as db:
            db.execute("INSERT INTO forecasts VALUES (?, ?, ?)",
                       (body["id"], body["created_utc"], json.dumps(body, allow_nan=False)))

    def forecasts(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT body FROM forecasts ORDER BY created DESC")]

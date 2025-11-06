import sqlite3
import os

class Database:
    def __init__(self, db_path='indexador.db'):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pdfs (
                    id INTEGER PRIMARY KEY,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    extracted_text TEXT,
                    metadata TEXT,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def insert_pdf(self, filename, filepath, extracted_text, metadata=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO pdfs (filename, filepath, extracted_text, metadata)
                VALUES (?, ?, ?, ?)
            ''', (filename, filepath, extracted_text, metadata))
            conn.commit()

    def get_all_pdfs(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT id, filename, filepath FROM pdfs')
            return cursor.fetchall()

    def get_pdf_text(self, pdf_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT extracted_text FROM pdfs WHERE id = ?', (pdf_id,))
            result = cursor.fetchone()
            return result[0] if result else None

    def get_pdf_metadata(self, pdf_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT metadata FROM pdfs WHERE id = ?', (pdf_id,))
            result = cursor.fetchone()
            return result[0] if result else None

import logging
import sqlite3
from datetime import datetime
from typing import Dict


class DatabaseManager:
    """Manages SQLite database for tracking processed papers"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        logging.info(f"🗄️  Initializing database: {db_path}")
        self.init_db()

    def init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_papers (
                arxiv_id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                processed_date TEXT,
                is_relevant BOOLEAN,
                is_israeli BOOLEAN,
                israeli_authors TEXT,
                confidence REAL,
                similarity_score REAL
            )
        ''')
        conn.commit()
        conn.close()
        logging.info(f"✅ Database initialized successfully!")
        logging.info(f"Database initialized: {self.db_path}")

    def is_paper_processed(self, arxiv_id: str) -> bool:
        """Check if paper was already processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT arxiv_id FROM processed_papers WHERE arxiv_id = ?', (arxiv_id,))
        result = cursor.fetchone()
        conn.close()
        return result is not None

    def save_processed_paper(self, arxiv_id: str, title: str, authors: str,
                             is_relevant: bool, is_israeli: bool = False,
                             israeli_authors: str = "", confidence: float = 0.0,
                             similarity_score: float = 0.0):
        """Save processed paper to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO processed_papers 
            (arxiv_id, title, authors, processed_date, is_relevant, is_israeli, israeli_authors, confidence, similarity_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (arxiv_id, title, authors, datetime.now().isoformat(),
              is_relevant, is_israeli, israeli_authors, confidence, similarity_score))
        conn.commit()
        conn.close()

    def get_stats(self) -> Dict:
        """Get processing statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM processed_papers')
        total = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM processed_papers WHERE is_relevant = 1')
        relevant = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM processed_papers WHERE is_israeli = 1')
        israeli = cursor.fetchone()[0]

        conn.close()

        return {
            'total_processed': total,
            'genai_relevant': relevant,
            'israeli_papers': israeli
        }

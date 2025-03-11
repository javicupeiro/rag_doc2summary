import os
import logging
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SQLiteStore:
    """
    Persistent docstore using SQLite to store original documents.
    Each document is saved with a unique identifier and its content.
    Compatible with Langchain's document store interface.
    """

    def __init__(self, db_path):
        """
        Initialize the SQLite document store.
        
        Args:
            db_path (str): Path to the SQLite database file
        """        
        self.db_path = db_path
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        try:
            self._create_table()
            logger.info(f"SQLite document store initialized at: {db_path}")
        except Exception as e:
            logger.error(f"Error initializing SQLite store: {e}")
            raise

    def _create_table(self):
        """
        Create the document store table if it doesn't exist.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS docstore (
                doc_id TEXT PRIMARY KEY,
                content TEXT
            )
            ''')
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"SQLite error creating table: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating table: {e}")
            raise

    def mset(self, docs):
         """
         Save multiple documents to the store.
         
         Args:
             docs: List of tuples (doc_id, content)
         
         Returns:
             bool: True if successful, False otherwise
         """
         try:
             if not docs:
                 logger.warning("No documents to save")
                 return True
                 
             conn = sqlite3.connect(self.db_path)
             cursor = conn.cursor()
             
             for doc_id, content in docs:
                 # Convert content to string if it's not already
                 if not isinstance(content, str):
                     content = str(content)
                     
                 cursor.execute(
                     'INSERT OR REPLACE INTO docstore (doc_id, content) VALUES (?, ?)',
                     (doc_id, content)
                 )
             
             conn.commit()
             conn.close()
             logger.info(f"Stored {len(docs)} documents in SQLite")
             return True
             
         except sqlite3.Error as e:
             logger.error(f"SQLite error storing documents: {e}")
             return False
         except Exception as e:
             logger.error(f"Error storing documents: {e}")
             return False
     
    def get(self, doc_id):
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id (str): The unique document identifier
        
        Returns:
            str: The document content or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT content FROM docstore WHERE doc_id = ?', (doc_id,))
            row = cursor.fetchone()
            conn.close()
            
            return row[0] if row else None
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error retrieving document {doc_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
            return None
    
    def delete(self, doc_id):
        """
        Delete a document by its ID.
        
        Args:
            doc_id (str): The unique document identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM docstore WHERE doc_id = ?', (doc_id,))
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted document with ID: {doc_id}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error deleting document {doc_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def count(self):
        """
        Count the number of documents in the store.
        
        Returns:
            int: The number of documents
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM docstore')
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error counting documents: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
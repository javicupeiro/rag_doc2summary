import os
import logging
import sqlite3


# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SQLiteStore():
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
     
    def get(self, ids):
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id (str): The unique document identifier
        
        Returns:
            str: The document content or None if not found
        """
        if not ids:
            return {}

        # Ensure ids is a list (if a single str is passed)
        if isinstance(ids, str):
            ids = [ids]

        placeholders = ','.join('?' for _ in ids)
        query = f"SELECT doc_id, content FROM docstore WHERE doc_id IN ({placeholders})"

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query, ids)
            rows = cursor.fetchall()

            # Return a dictionary {id: original_data}
            return {row[0]: row[1] for row in rows}
        except sqlite3.Error as e:
            logging.error(f"SQLite error retrieving document {ids}: {e}")
            return {}


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
        
    def clear_sqlite_data(self,db_path="processed_files.db"):    
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM files")
        cursor.execute("DELETE FROM summaries")
        conn.commit()
        conn.close()
        
    def clear_sqlite_data(self, db_path=None):
            if db_path is None:
                db_path = self.db_path

            try:
                # Connect to the SQLite database
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Check if the 'files' table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='files';")
                table_exists = cursor.fetchone()

                if table_exists:
                    # If the 'files' table exists, clear its data
                    cursor.execute("DELETE FROM files;")
                    conn.commit()
                    logger.info("Successfully cleared the 'files' table data.")
                else:
                    logger.warning("Table 'files' does not exist, skipping clearing.")

                # Close the connection
                conn.close()

            except sqlite3.Error as e:
                logger.error(f"SQLite error: {e}")
                raise
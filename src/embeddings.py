import os
from dotenv import load_dotenv
import logging
import base64
import uuid
# Chroma DB
from chromadb.config import Settings
from chromadb import EmbeddingFunction, PersistentClient
from chromadb.api.types import Documents, Embeddings
# SQLite
import sqlite3
from src.sqlite_processor import SQLiteStore
# AI
from openai import AzureOpenAI, OpenAI
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AZURE_API_KEY= os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT= os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT_NAME= os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_API_VERSION= os.getenv("AZURE_API_VERSION")

# Define prompt directory
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "../data/prompts/")
# Default database directories
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "../data/database/vector_db/vector_db")
SQL_DB_DIR = os.path.join(os.path.dirname(__file__), "../data/database/sql/docstore.db")

class OpenAIEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function using OpenAI's API, conforming to ChromaDB's interface.
    """
    def __init__(self, 
                 api_key: str, 
                 model_name: str = "text-embedding-3-small"):
        if not api_key:
            raise ValueError("OpenAI API Key not found.")
        
        # It's generally better practice to initialize the client within the class
        # that uses it directly, especially if this class might be used independently.
        self._client = OpenAI(api_key=api_key) 
        self._model_name = model_name
        logger.info(f"Initialized OpenAIEmbeddingFunction with model: {self._model_name}")

    # This is the required method by ChromaDB's EmbeddingFunction interface
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generates embeddings for a list of documents.

        Args:
            input (Documents): A list of strings to embed.

        Returns:
            Embeddings: A list of embeddings (list of lists of floats).
        """
        if not isinstance(input, list):
             # Although ChromaDB should always pass a list, adding a safeguard
             logger.warning("Input to embedding function was not a list. Converting.")
             input = [str(input)] # Ensure it's a list of strings

        if not input:
             return [] # Handle empty input list

        try:
            logger.debug(f"Generating embeddings for {len(input)} documents.")
            response = self._client.embeddings.create(
                input=input,
                model=self._model_name
            )
            # Extract embeddings from the response
            embeddings = [item.embedding for item in response.data]
            logger.debug(f"Successfully generated {len(embeddings)} embeddings.")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {e}", exc_info=True)
            # Decide how to handle errors: re-raise, return empty, return None?
            # Re-raising is often best to signal failure upstream.
            raise

class EmbeddingsGenerator:
    """
    A class for generating embeddings and summaries from text, tables, and images.
    Uses various LLM models for processing different types of content.
    """

    def __init__(self,
                 embedding_model: str = 'openai',
                 vector_db_dir: str = VECTOR_DB_DIR, # Use the constant default
                 sqlite_db_path: str = SQL_DB_DIR
                 ):
        """
        Initialize the embeddings generator. It will load an existing ChromaDB
        from vector_db_dir if it exists, otherwise, it will create a new one.
        """
        self.vector_db_dir = vector_db_dir
        self.sqlite_db_path = sqlite_db_path
        self.embedding_model = embedding_model
        self.text_summaries = []
        self.retrieved_docs = []
        self.id_key= "doc_id"

        logger.info(f"Initializing EmbeddingsGenerator...")
        # Log the absolute paths for clarity
        abs_vector_db_dir = os.path.abspath(self.vector_db_dir)
        abs_sqlite_db_path = os.path.abspath(self.sqlite_db_path)
        logger.info(f"Vector DB Persistence Path: {abs_vector_db_dir}")
        logger.info(f"SQLite DB Path: {abs_sqlite_db_path}")

        try:
            # --- Ensure Parent Directories Exist ---
            # Make sure the directory *containing* the vector_db_dir exists
            vector_parent_dir = os.path.dirname(abs_vector_db_dir)
            if not os.path.exists(vector_parent_dir):
                logger.warning(f"Parent directory for vector DB does not exist: {vector_parent_dir}. Creating it.")
                os.makedirs(vector_parent_dir, exist_ok=True)

            # Make sure the directory *containing* the sqlite_db_path exists
            sql_parent_dir = os.path.dirname(abs_sqlite_db_path)
            if not os.path.exists(sql_parent_dir):
                 logger.warning(f"Parent directory for SQLite DB does not exist: {sql_parent_dir}. Creating it.")
                 os.makedirs(sql_parent_dir, exist_ok=True)

            # --- Initialize Embedding Function ---
            logger.info("Initializing embedding function...")
            if embedding_model == 'openai':
                if not OPENAI_API_KEY:
                    raise ValueError("OpenAI API Key not found in environment variables")
                self.embedding_fn = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-3-small")
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                logger.info("Initialized OpenAI embedding function.")
            else:
                raise NotImplementedError(f"Embedding model '{embedding_model}' not implemented")

            # --- Initialize ChromaDB ---
            logger.info(f"Attempting to initialize ChromaDB client using PersistentClient.")
            logger.info(f"Persistence directory provided: '{self.vector_db_dir}'")

            # Check if the persistence directory exists before initializing
            if os.path.exists(self.vector_db_dir):
                logger.info(f"Persistence directory '{self.vector_db_dir}' exists. Attempting to load.")
                # Check for the expected SQLite file as a basic sanity check
                sqlite_file_path = os.path.join(self.vector_db_dir, "chroma.sqlite3")
                if not os.path.isfile(sqlite_file_path):
                     logger.warning(f"Persistence directory exists, but expected file '{sqlite_file_path}' not found. Chroma might re-initialize.")
            else:
                logger.info(f"Persistence directory '{self.vector_db_dir}' does not exist. Chroma will create it.")

            # Use PersistentClient for clarity when using on-disk storage.
            # It handles both loading if path exists and creating if not.
            # Settings are often implicitly handled by PersistentClient's path argument,
            # but we can pass them for explicit configuration like telemetry.
            chroma_settings = Settings(
                anonymized_telemetry=False,
                is_persistent=True # Explicitly confirm persistence
                # persist_directory=self.vector_db_dir # This is implicitly set by PersistentClient's path
            )
            self.chroma_client = PersistentClient(path=self.vector_db_dir, settings=chroma_settings)
            logger.info("ChromaDB PersistentClient initialized.")

            # Verify connection by listing collections (optional debug step)
            try:
                existing_collections = self.chroma_client.list_collections()
                logger.info(f"Available collections after client init: {[c.name for c in existing_collections]}")
            except Exception as e:
                logger.error(f"Could not list collections after client init: {e}")
                # This might indicate a deeper problem with the DB state

            # Get or create the specific collection
            collection_name = "multi_modal_rag"
            logger.info(f"Getting or creating collection: '{collection_name}'...")
            # This is the standard way to ensure the collection exists
            self.vectorstore = self.chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
                # metadata={"hnsw:space": "cosine"} # Optional: Specify distance metric
            )
            logger.info(f"Successfully obtained collection '{self.vectorstore.name}'.")
            # Check count immediately after getting/creating
            logger.info(f"Initial item count in collection '{self.vectorstore.name}': {self.vectorstore.count()}")

            # --- Initialize SQLite Document Store ---
            logger.info("Initializing SQLite document store...")
            self.docstore = SQLiteStore(db_path=self.sqlite_db_path)
            logger.info(f"SQLite document store initialized. Path: {abs_sqlite_db_path}")

            # --- Init LLM Clients ---
            logger.info("Initializing LLM clients...")
            # Init Azure client
            self.azure_deployment_name = AZURE_DEPLOYMENT_NAME
            self.azure_client = AzureOpenAI(
                api_version=AZURE_API_VERSION,
                api_key=AZURE_API_KEY,
                azure_endpoint=AZURE_ENDPOINT
            )
            # Init Groq client
            if not GROQ_API_KEY:
                raise ValueError("Groq API Key not found in environment variables")
            self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            logger.info("LLM clients initialized.")

            logger.info("EmbeddingsGenerator initialized successfully.")

        except sqlite3.OperationalError as e:
             # Catch the specific error during initialization
             logger.error(f"SQLite Operational Error during ChromaDB initialization: {e}", exc_info=True)
             logger.error("!!!!! IMPORTANT !!!!!")
             logger.error(f"This usually means the ChromaDB database file at '{os.path.join(abs_vector_db_dir, 'chroma.sqlite3')}' is corrupted or incomplete.")
             logger.error(f"Please STOP the application, COMPLETELY DELETE the directory '{abs_vector_db_dir}' and all its contents, then restart.")
             logger.error("Do NOT just empty it, delete the folder itself.")
             logger.error("!!!!! IMPORTANT !!!!!")
             raise # Re-raise the error to stop execution
        except Exception as e:
            logger.error(f"Unexpected error during EmbeddingsGenerator initialization: {e}", exc_info=True)
            raise


    def get_embedding(self, text):
        """
        Get embedding for a given text using OpenAI's API.
        
        Args:
            text (str): The input text.
            
        Returns:
            list: Embedding vector.
        """
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small" #"text-embedding-ada-002"
            )
            # If text is a list, return a list of embeddings for each element.
            if isinstance(text, list):
                embeddings = [item.embedding for item in response.data]
                # Log information from the first embedding (optional)
                logger.info(f"Batch embedding: Returned {len(embeddings)} embeddings; each of length {len(embeddings[0])}")
                return embeddings
            else:
                embedding = response.data[0].embedding
                logger.info(f"Embedding type: {type(embedding)}; Length: {len(embedding)}")
                logger.info(f"Embedding text: {text}")
                if isinstance(embedding, float):
                    logger.error("Embedding returned as float for text: " + text)
                    raise ValueError("Embedding is a float, expected list")
                return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise


    def count_docstore_elements(self):
        """
        Counts the number of elements stored in the docstore.
    
        Returns:
            int: Number of elements stored in the docstore.
        """
        return self.docstore.count()


    def load_prompt(self, file_path: str) -> str:
        """
        Load a prompt from a file.
        
        Args:
            file_path (str): Path to the prompt file.
            
        Returns:
            str: The prompt text.
        """
        try:
            logger.debug(f"Loading prompt from: {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Prompt file not found: {file_path}")
                
            with open(file_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()                
            return prompt
        except Exception as e:
            logger.error(f"Error loading prompt: {e}")
            raise

    def call_groq_model(self, prompt: str, content: str) -> str:
        """
        Call the Groq model directly (stub function).
        Replace with an actual HTTP request to the Groq API.
        """
        if not GROQ_API_KEY:
            raise ValueError("Groq API Key not found in environment variables")
        
        prompt = prompt.format(element=content)
        logger.info("Calling Groq model (stub) for content summarization")
        completion = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role":"user",
                    "content":prompt
                }
            ]
        )        
        logger.info(f"Groq summarization: {completion.choices[0].message.content}")
        return completion.choices[0].message.content

    def summary_text_and_tables(self, text_list, table_list):
            """
            Generate summaries for text and tables using the Groq model.
            """
            try:
                summary_prompt_path = os.path.join(PROMPT_DIR, 'summary_text.txt')
                summary_prompt = self.load_prompt(summary_prompt_path)

                text_summaries = []
                if text_list:
                    logger.info(f"Summarizing {len(text_list)} text elements")
                    text_summaries = [self.call_groq_model(summary_prompt, txt) for txt in text_list]
                    logger.info(f"Generated {len(text_summaries)} text summaries")

                table_summaries = []
                if table_list:
                    logger.info(f"Summarizing {len(table_list)} table elements")
                    table_summaries = [self.call_groq_model(summary_prompt, tbl) for tbl in table_list]
                    logger.info(f"Generated {len(table_summaries)} table summaries")

                return text_summaries, table_summaries
            except Exception as e:
                logger.error(f"Error summarizing text and tables: {e}")
                raise

    def call_azure_openai_for_image_summary(self, image: str, prompt: str) -> str:
        """
        Call Azure OpenAI for image summarization (stub function).
        Replace with an actual API call.
        """
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API Key not found in environment variables")
        logger.info("Calling Azure OpenAI (stub) for image summarization")
        # Validate base64 image format
        base64.b64decode(image)
        # Build prompt
        prompt_message = [
            { "role": "system", 
              "content": "Responde a la pregunta en español basándote únicamente en el siguiente contexto, que puede incluir texto, tablas e imágenes." 
            },
            {
              "role": "user",
              "content": [
                  {
                      "type": "text", 
                      "text": prompt
                  },
                  { 
                      "type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                  }
              ]
            }
        ]
        response = self.azure_openai_query(prompt_message)
        return response

    def summary_images(self, image_list):
        """
        Generate summaries for base64-encoded images using Azure OpenAI.
        """
        try:
            if not image_list:
                logger.info("No images to summarize")
                return []
                
            summary_prompt_path = os.path.join(PROMPT_DIR, 'summary_image.txt')
            summary_prompt = self.load_prompt(summary_prompt_path)
            
            logger.info(f"Summarizing {len(image_list)} images")
            image_summaries = [self.call_azure_openai_for_image_summary(img, summary_prompt)
                               for img in image_list]
            logger.info(f"Generated {len(image_summaries)} image summaries")
            return image_summaries
        except Exception as e:
            logger.error(f"Error summarizing images: {e}")
            raise

    def persist_data(self, text_summaries, table_summaries, image_summaries,
                    original_texts, original_tables, original_images):
       """
       Persist summaries in the Chromadb vector store and original data in SQLite.
       """
       # Inner function definition remains the same
       def add_documents(summaries, originals, content_type):
           if not summaries or not originals:
               logger.info(f"No {content_type} to persist")
               return
           if len(summaries) != len(originals):
               # Log clearly if mismatch happens
               logger.error(
                   f"CRITICAL MISMATCH: Cannot persist {content_type}. "
                   f"Number of summaries ({len(summaries)}) does not match "
                   f"number of originals ({len(originals)}). Aborting add for this type."
               )
               # Optionally raise an error or just return to prevent adding partial data
               # raise ValueError(f"Mismatch between {content_type} summaries and originals.")
               return # Safer to just skip adding this type if counts mismatch

           logger.info(f"Generating {len(summaries)} unique IDs for {content_type} documents...")
           doc_ids = [str(uuid.uuid4()) for _ in range(len(summaries))]

           logger.info(f"Preparing {len(summaries)} metadata entries for {content_type}...")
           metadatas = [{"doc_id": doc_ids[i], "type": content_type} for i in range(len(summaries))]

           try:
               # Add summaries to Chromadb
               logger.info(f"Adding {len(summaries)} '{content_type}' summaries to Chromadb collection '{self.vectorstore.name}'...")
               # Make sure summaries is a list of strings
               if not all(isinstance(s, str) for s in summaries):
                   logger.warning(f"Detected non-string elements in '{content_type}' summaries. Attempting conversion.")
                   summaries = [str(s) for s in summaries]

               self.vectorstore.add(
                   documents=summaries,
                   metadatas=metadatas,
                   ids=doc_ids
               )
               logger.info(f"Successfully added {len(summaries)} '{content_type}' summaries to Chromadb.")
               logger.info(f"Collection count is now: {self.vectorstore.count()}")


               # Persist the original content in SQLite
               logger.info(f"Persisting {len(summaries)} original '{content_type}' items in SQLite...")
               # Ensure originals are suitable for SQLite (likely strings or bytes if images)
               # No major change needed here assuming mset handles the data types
               self.docstore.mset(list(zip(doc_ids, originals)))
               logger.info(f"Successfully persisted {len(summaries)} original '{content_type}' items in SQLite.")

           except sqlite3.OperationalError as e:
               logger.error(f"SQLite Operational Error during ChromaDB ADD operation: {e}", exc_info=True)
               logger.error(f"This likely means the DB state is inconsistent. Path: {self.vector_db_dir}")
               raise # Re-raise to signal failure
           except Exception as e:
               logger.error(f"Error adding {content_type} documents to stores: {e}", exc_info=True)
               raise # Re-raise other errors

       # --- Call add_documents for each type ---
       try:
           logger.info("Persisting text data...")
           add_documents(text_summaries, original_texts, "text")

           logger.info("Persisting table data...")
           add_documents(table_summaries, original_tables, "table")

           logger.info("Persisting image data...")
           add_documents(image_summaries, original_images, "image")

           logger.info("Data persistence process completed.")
           # Return vectorstore and docstore for further retrieval if needed.
           return self.vectorstore, self.docstore

       except Exception as e:
           # Catch errors from the add_documents calls
           logger.error(f"Error during data persistence phase: {e}", exc_info=True)
           # Depending on requirements, you might want to raise this or handle it
           raise

 
    def retrieve_data(self, query: str, k: int = 3):
        """
        Retrieve documents by performing a similarity search in the vector store
        and then fetching the original data from the SQLite document store.

        Returns a list of dicts with keys "page_content" and "metadata".
        """
        logger.info(f"Retrieving top {k} documents for query: '{query[:100]}...'") # Log snippet of query
        try:
            # Query Chromadb directly
            logger.info(f"Querying Chroma collection '{self.vectorstore.name}'...")
            result = self.vectorstore.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas"] # Only request needed fields
            )
            logger.info(f"Chroma query returned {len(result.get('ids', [[]])[0])} results.")

            # Process results carefully, checking structure
            docs = []
            retrieved_ids = result.get("ids", [[]])[0] # Get list of IDs or empty list
            retrieved_docs = result.get("documents", [[]])[0]
            retrieved_metadatas = result.get("metadatas", [[]])[0]

            if not retrieved_ids:
                 logger.warning(f"Chroma query for '{query[:50]}...' returned no results.")
                 return [] # Return empty list if no results

            if not (len(retrieved_ids) == len(retrieved_docs) == len(retrieved_metadatas)):
                logger.error("Mismatch in lengths of retrieved ids, documents, and metadatas from Chroma query!")
                # Handle this inconsistency - perhaps return empty or try to process matched pairs
                return [] # Safer to return empty

            logger.info(f"Processing {len(retrieved_ids)} retrieved items from Chroma.")
            doc_id_map = {} # To store doc_ids needed from SQLite
            for i in range(len(retrieved_ids)):
                chroma_id = retrieved_ids[i]
                doc_text = retrieved_docs[i]
                meta = retrieved_metadatas[i]

                doc_entry = {"page_content": doc_text, "metadata": meta}
                docs.append(doc_entry)

                # Get the specific doc_id stored in metadata for SQLite lookup
                sqlite_doc_id = meta.get("doc_id")
                if sqlite_doc_id:
                    doc_id_map[sqlite_doc_id] = None # Just need the keys
                else:
                    logger.warning(f"Retrieved item with Chroma ID '{chroma_id}' is missing 'doc_id' in metadata.")


            # Retrieve original data from SQLite using stored doc_ids
            sqlite_doc_ids_to_fetch = list(doc_id_map.keys())
            original_data_dict = {}
            if sqlite_doc_ids_to_fetch:
                logger.info(f"Fetching original data for {len(sqlite_doc_ids_to_fetch)} doc_ids from SQLite...")
                try:
                    original_data_dict = self.docstore.get(sqlite_doc_ids_to_fetch)
                    logger.info(f"Successfully fetched {len(original_data_dict)} items from SQLite.")
                except Exception as e:
                    logger.error(f"Error fetching data from SQLite store: {e}", exc_info=True)
                    # Decide how to proceed - maybe continue without original data?
            else:
                logger.warning("No valid 'doc_id' found in retrieved metadata to fetch from SQLite.")


            # Add original data back to the docs list
            for doc in docs:
                doc_id = doc["metadata"].get("doc_id")
                if doc_id in original_data_dict:
                    doc["metadata"]["original_data"] = original_data_dict[doc_id]
                    # logger.debug(f"Attached original data for doc_id: {doc_id}")
                else:
                    # This case covers if doc_id was missing or SQLite fetch failed for it
                    doc["metadata"]["original_data"] = None
                    if doc_id: # Only log warning if we expected data
                       logger.warning(f"Original data not found in SQLite results for doc_id: {doc_id}")


            logger.info(f"Successfully retrieved and processed {len(docs)} documents for the query.")
            return docs

        except sqlite3.OperationalError as e:
            # Catch specific error during query
            logger.error(f"SQLite Operational Error during ChromaDB QUERY operation: {e}", exc_info=True)
            logger.error(f"This likely means the DB state is inconsistent. Path: {self.vector_db_dir}")
            raise
        except Exception as e:
            logger.error(f"Error retrieving data for query '{query[:50]}...': {e}", exc_info=True)
            raise


    def parse_docs(self, docs):
        """
        Split retrieved documents into images and texts based on whether the
        'original_data' in their metadata can be base64-decoded.
        Returns a dictionary {"images": [...], "texts": [...]}.
        """
        images = []
        texts = [] # This 'texts' list isn't directly used in the corrected build_prompt context
                  # but the function correctly separates images based on original_data.
        for doc in docs:
            # CRUCIAL: Use 'original_data' from metadata
            original = doc.get("metadata", {}).get("original_data")

            if not original:
                # logger.debug("Document lacks original_data in metadata, cannot parse type.")
                continue # Skip if no original data to parse

            # Attempt to decode. If it works, assume image. Otherwise, assume text.
            try:
                # Add strict=True for more robust validation if needed
                base64.b64decode(original, validate=True)
                # Check if it's likely a string representation of an image
                if isinstance(original, str) and original.startswith(('data:image', '/9j/')):# Common base64 image starts
                     images.append(original)
                     # logger.debug("Identified image based on base64 decode.")
                elif isinstance(original, bytes): # If original data was stored as bytes
                     images.append(base64.b64encode(original).decode('utf-8')) # Re-encode to base64 string
                     # logger.debug("Identified image (from bytes) based on base64 decode.")
                else:
                     # Decoded successfully but doesn't look like image string/bytes? Treat as text.
                     if isinstance(original, str):
                         texts.append(original)
                     # logger.debug("Base64 decoded but doesn't seem like image format, treated as text.")

            except (base64.binascii.Error, ValueError, TypeError):
                # If decoding fails, it's likely text (or other non-base64 data)
                if isinstance(original, str):
                    texts.append(original)
                    # logger.debug("Identified text (base64 decode failed).")
                elif isinstance(original, bytes):
                     try:
                         texts.append(original.decode('utf-8')) # Try decoding bytes as UTF-8 text
                         # logger.debug("Identified text (decoded bytes to utf-8).")
                     except UnicodeDecodeError:
                         logger.warning("Could not decode original_data bytes as UTF-8 text. Skipping.")
                else:
                     logger.warning(f"Original data is neither string nor bytes after failing base64 decode. Type: {type(original)}. Skipping.")

        logger.info(f"Parsed documents into {len(images)} potential images and {len(texts)} potential texts based on original_data.")
        return {"images": images, "texts": texts}
 
    def azure_openai_query(self, message_prompt):
        """
        Send the prompt to Azure OpenAI (using a direct API call).
        This implementation uses a placeholder based on your previous usage.
        """
        logger.info(f"Sending prompt to AzureOpenAI...")
        response = self.azure_client.beta.chat.completions.parse(
            model=self.azure_deployment_name,
            messages=message_prompt,
            temperature = 0.0
        )
        if not response.choices:
                logger.info("Did not receive any answer from Azure OpenAI")
                return "error"        
        #response_content = response.choices[0].message["content"].strip()
        response_content = response.choices[0].message.content.strip()
        logger.info(f"Azure OpenAI response received: {response_content}")
        return response_content

    def build_prompt(self, user_query, docs):
        """
        Build a prompt message using the retrieved context (original text and images).
        """
        logger.info(f"Building prompt using retrieved original content...")

        # Retrieve original texts/tables for the context
        original_texts_for_context = []
        processed_doc_ids = set() # Keep track to avoid duplicates if somehow needed

        for doc in docs:
            metadata = doc.get("metadata", {})
            doc_id = metadata.get("doc_id")
            doc_type = metadata.get("type")
            original_content = metadata.get("original_data") # <-- OBTENER EL ORIGINAL

            # Ensure we have original content and it's for text/table types
            if original_content and doc_type in ("text", "table"):
                # Optional: Check if it's actually a string, just in case
                if isinstance(original_content, str):
                     # Avoid adding duplicates if the same doc_id appeared multiple times in retrieval
                     if doc_id not in processed_doc_ids:
                         original_texts_for_context.append(original_content)
                         if doc_id: # Only add if doc_id exists
                             processed_doc_ids.add(doc_id)
                else:
                    logger.warning(f"Original content for doc_id {doc_id} (type: {doc_type}) is not a string. Skipping.")
            elif not original_content and doc_type in ("text", "table"):
                 logger.warning(f"Original content missing in metadata for retrieved doc_id {doc_id} (type: {doc_type}). Cannot add to context.")


        # Join the collected *original* texts
        context_text = "\n\n".join(original_texts_for_context) # Use newline separation for clarity

        if not context_text:
            logger.warning("No valid text/table original content found in retrieved docs to build context.")
            # Decide how to handle: maybe send a message saying no context found, or proceed without text context?
            # For now, we'll proceed with an empty context_text, the LLM will rely only on images if any.

        # Log a snippet of the context being used
        logger.info(f"Context Text for Prompt (Originals - Snippet): {context_text[:500]}...") # Log first 500 chars

        # --- Load prompt template and format ---
        summary_prompt_path = os.path.join(PROMPT_DIR, 'user_query.txt')
        summary_prompt_template = self.load_prompt(summary_prompt_path)

        filled_prompt = summary_prompt_template.format(
            context_text=context_text, # <-- Ahora contiene los ORIGINALES
            user_query=user_query
        )

        # --- Prepare final message list ---
        prompt_message = [
            {
                "role": "system",
                "content": "Responde a la pregunta en español basándote únicamente en el contexto proporcionado (texto e imágenes)." # Clarified prompt
            },
            {"role": "user", "content": filled_prompt}
        ]

        # --- Handle Images ---
        # This part correctly uses original_data via parse_docs, but let's re-verify parse_docs
        docs_by_type = self.parse_docs(docs) # Separates originals into images/texts based on base64 decode

        if docs_by_type["images"]:
            logger.info(f"Adding {len(docs_by_type['images'])} images to the prompt.")
            # Add image content separately, Azure format usually expects images in the 'user' turn content list
            # Check Azure documentation for the exact multi-modal format they expect.
            # Often it's adding the image_url dict within the 'content' list of the *last* user message.

            # Let's try adding images to the main user message content list:
            if isinstance(prompt_message[-1]["content"], str): # If current content is just text
                 prompt_message[-1]["content"] = [{"type": "text", "text": prompt_message[-1]["content"]}] # Convert to list
            elif not isinstance(prompt_message[-1]["content"], list):
                 logger.error("Unexpected format for last user message content. Cannot add images.")
                 # Handle error appropriately

            # Append image URLs to the content list
            for img_b64 in docs_by_type["images"]:
                 # Ensure it's a valid base64 string before adding
                 try:
                     base64.b64decode(img_b64) # Quick validation
                     prompt_message[-1]["content"].append({
                         "type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                     })
                 except Exception as decode_err:
                     logger.error(f"Failed to decode base64 image data before adding to prompt: {decode_err}. Skipping image.")

            # --- Alternative way to add images (separate message - less common for context): ---
            # for img in docs_by_type["images"]:
            #     prompt_message.append({
            #         "role": "user", # Or potentially 'system' if providing image context? Check Azure docs.
            #         "content": [
            #             # Optional text preamble for the image
            #             # {"type": "text", "text": "Context Image:"},
            #             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
            #         ]
            #     })

        logger.debug(f"Final prompt message structure: {prompt_message}")
        return prompt_message


    def process_user_query(self, 
                           user_query:str, 
                           k:int = 3):
        """
        Retrieve relevant context from the stores, build a prompt, and send it to Azure OpenAI.
        """
        logger.info(f"Processing query: {user_query}")
        # Retrieve data
        self.retrieved_docs = self.retrieve_data(user_query, k=k)
        # Build prompt
        prompt_message = self.build_prompt(user_query, self.retrieved_docs)
        # Send prompt
        response = self.azure_openai_query(prompt_message)
        return response

    def get_retrieved_context_data(self):
        """
        Returns the ORIGINAL context text and image data from the retrieved documents
        stored in self.retrieved_docs.
        """
        if not self.retrieved_docs:
             logger.warning("get_retrieved_context_data called but self.retrieved_docs is empty.")
             return [], [] # Return empty lists if no docs were retrieved

        logger.info(f"Extracting original context for display from {len(self.retrieved_docs)} retrieved documents.")

        original_texts_for_display = []
        processed_doc_ids_display = set() # Avoid duplicates in display

        for doc in self.retrieved_docs:
            metadata = doc.get("metadata", {})
            doc_id = metadata.get("doc_id")
            doc_type = metadata.get("type")
            
            original_content = metadata.get("original_data")            

            # Extraer solo textos y tablas originales para mostrar como 'context_text'
            if doc_type in ("text", "table"):
                if original_content:
                    if isinstance(original_content, str):
                         # Evitar duplicados si el mismo doc_id fue recuperado varias veces
                         if doc_id not in processed_doc_ids_display:
                             original_texts_for_display.append(original_content)
                             if doc_id:
                                 processed_doc_ids_display.add(doc_id)
                    else:
                        logger.warning(f"Original content for display (doc_id {doc_id}, type {doc_type}) is not a string. Skipping.")
                else:
                    
                    logger.warning(f"Original content missing in metadata for display (doc_id {doc_id}, type {doc_type}). Skipping.")
        
        docs_by_type = self.parse_docs(self.retrieved_docs)
        context_img = docs_by_type['images']

        logger.info(f"Returning {len(original_texts_for_display)} original text/table contexts and {len(context_img)} image contexts for display.")

        return original_texts_for_display, context_img
     
    def clear_chroma_data(self):
        import shutil
        shutil.rmtree(self.vector_db_dir, ignore_errors=True)
        
    def clear_sqlite_data(self): 
        self.docstore.clear_sqlite_data(self.sqlite_db_path)
    
    def clear_databases(self):
        logger.info(f"Cleaning Chroma db...")
        self.clear_chroma_data()
        logger.info(f"Cleaning SQlite...")
        self.clear_sqlite_data()
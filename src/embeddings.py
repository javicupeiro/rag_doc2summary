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
            # Filter out any non-string or empty string inputs before sending to API
            processed_input = [doc for doc in input if isinstance(doc, str) and doc.strip()]
            if not processed_input:
                logger.warning("All input documents were empty or non-string after filtering.")
                # Return empty embeddings for each original input to maintain length consistency if needed by Chroma,
                # or handle as appropriate for your use case. Here, returning empty for empty valid input.
                # Chroma might handle mismatch if we return less than expected.
                # For simplicity, if all are empty, return an empty list.
                return []


            response = self._client.embeddings.create(
                input=processed_input, # Send only valid, non-empty strings
                model=self._model_name
            )
            # Extract embeddings from the response
            embeddings = [item.embedding for item in response.data]
            logger.debug(f"Successfully generated {len(embeddings)} embeddings for {len(processed_input)} documents.")
            
            # If ChromaDB expects an embedding for every original item, including empty ones,
            # we might need to map back or insert dummy embeddings.
            # For now, assume Chroma handles cases where len(embeddings) < len(original_input) if empty strings were filtered.
            # Or, more robustly, create a placeholder for empty strings if the API can't process them.
            # Let's assume the current behavior (only embedding non-empty) is acceptable.

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
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY) # Standard OpenAI client
                logger.info("Initialized OpenAI embedding function and client.")
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

            chroma_settings = Settings(
                anonymized_telemetry=False,
                is_persistent=True 
            )
            self.chroma_client = PersistentClient(path=self.vector_db_dir, settings=chroma_settings)
            logger.info("ChromaDB PersistentClient initialized.")

            try:
                existing_collections = self.chroma_client.list_collections()
                logger.info(f"Available collections after client init: {[c.name for c in existing_collections]}")
            except Exception as e:
                logger.error(f"Could not list collections after client init: {e}")

            collection_name = "multi_modal_rag"
            logger.info(f"Getting or creating collection: '{collection_name}'...")
            self.vectorstore = self.chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
            logger.info(f"Successfully obtained collection '{self.vectorstore.name}'.")
            logger.info(f"Initial item count in collection '{self.vectorstore.name}': {self.vectorstore.count()}")

            # --- Initialize SQLite Document Store ---
            logger.info("Initializing SQLite document store...")
            self.docstore = SQLiteStore(db_path=self.sqlite_db_path)
            logger.info(f"SQLite document store initialized. Path: {abs_sqlite_db_path}")

            # --- Init LLM Clients ---
            logger.info("Initializing LLM clients...")
            # Init Azure client (for image summarization)
            self.azure_deployment_name = AZURE_DEPLOYMENT_NAME
            self.azure_client = AzureOpenAI(
                api_version=AZURE_API_VERSION,
                api_key=AZURE_API_KEY,
                azure_endpoint=AZURE_ENDPOINT
            )
            # Init Groq client (for text/table summarization if selected)
            if not GROQ_API_KEY:
                raise ValueError("Groq API Key not found in environment variables")
            self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            logger.info("LLM clients initialized.")

            logger.info("EmbeddingsGenerator initialized successfully.")

        except sqlite3.OperationalError as e:
             logger.error(f"SQLite Operational Error during ChromaDB initialization: {e}", exc_info=True)
             logger.error("!!!!! IMPORTANT !!!!!")
             logger.error(f"This usually means the ChromaDB database file at '{os.path.join(abs_vector_db_dir, 'chroma.sqlite3')}' is corrupted or incomplete.")
             logger.error(f"Please STOP the application, COMPLETELY DELETE the directory '{abs_vector_db_dir}' and all its contents, then restart.")
             logger.error("Do NOT just empty it, delete the folder itself.")
             logger.error("!!!!! IMPORTANT !!!!!")
             raise 
        except Exception as e:
            logger.error(f"Unexpected error during EmbeddingsGenerator initialization: {e}", exc_info=True)
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
        Call the Groq model for content summarization.
        """
        if not GROQ_API_KEY:
            raise ValueError("Groq API Key not found in environment variables")
        
        formatted_prompt = prompt.format(element=content)
        logger.info("Calling Groq model (llama-3.1-8b-instant) for content summarization")
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role":"user",
                        "content": formatted_prompt
                    }
                ]
            )        
            summary = completion.choices[0].message.content
            logger.info(f"Groq summarization successful. Length: {len(summary)}")
            return summary
        except Exception as e:
            logger.error(f"Error calling Groq model: {e}", exc_info=True)
            raise

    def call_openai_for_summary(self, prompt: str, content: str) -> str:
        """
        Call the OpenAI model (GPT-4o) for content summarization.
        """
        if not self.openai_client: 
            raise ValueError("OpenAI client not initialized.")
        
        formatted_prompt = prompt.format(element=content)
        logger.info("Calling OpenAI model (gpt-4o) for content summarization")
        messages_for_api = [
            {"role": "user", "content": formatted_prompt}
        ]
        summary = self.azure_openai_query(messages_for_api)
        return summary
    

    def summary_text_and_tables(self, text_list, table_list, summarization_model_choice: str = "Groq (Llama3-8b)"):
            """
            Generate summaries for text and tables using the chosen model.
            Args:
                text_list (list): List of text strings.
                table_list (list): List of table strings.
                summarization_model_choice (str): The model to use for summarization 
                                                  ("Groq (Llama3-8b)" or "GPT4o").
            """
            try:
                summary_prompt_path = os.path.join(PROMPT_DIR, 'summary_text.txt')
                summary_prompt = self.load_prompt(summary_prompt_path)

                text_summaries = []
                if text_list:
                    logger.info(f"Summarizing {len(text_list)} text elements using {summarization_model_choice}")
                    if summarization_model_choice == "GPT4o":
                        text_summaries = [self.call_openai_for_summary(summary_prompt, txt) for txt in text_list]
                    elif summarization_model_choice == "Groq (Llama3-8b)":
                        text_summaries = [self.call_groq_model(summary_prompt, txt) for txt in text_list]
                    else:
                        raise ValueError(f"Unsupported summarization model: {summarization_model_choice}")
                    logger.info(f"Generated {len(text_summaries)} text summaries")

                table_summaries = []
                if table_list:
                    logger.info(f"Summarizing {len(table_list)} table elements using {summarization_model_choice}")
                    if summarization_model_choice == "GPT4o":
                        table_summaries = [self.call_openai_for_summary(summary_prompt, tbl) for tbl in table_list]
                    elif summarization_model_choice == "Groq (Llama3-8b)":
                        table_summaries = [self.call_groq_model(summary_prompt, tbl) for tbl in table_list]
                    else:
                        raise ValueError(f"Unsupported summarization model: {summarization_model_choice}")
                    logger.info(f"Generated {len(table_summaries)} table summaries")

                return text_summaries, table_summaries
            except Exception as e:
                logger.error(f"Error summarizing text and tables: {e}", exc_info=True)
                raise

    def call_azure_openai_for_image_summary(self, image: str, prompt: str) -> str:
        """
        Call Azure OpenAI for image summarization.
        """
        if not self.azure_client:
            raise ValueError("Azure OpenAI client not initialized.")
        logger.info("Calling Azure OpenAI for image summarization")
        try:
            # Validate base64 image format
            base64.b64decode(image) # This will raise an error if invalid
            
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
            response_content = self.azure_openai_query(prompt_message)
            logger.info("Azure OpenAI image summarization successful.")
            return response_content
        except base64.binascii.Error as b64e:
            logger.error(f"Invalid base64 image string for Azure OpenAI: {b64e}", exc_info=True)
            raise ValueError("Invalid base64 image string provided.") from b64e
        except Exception as e:
            logger.error(f"Error calling Azure OpenAI for image summary: {e}", exc_info=True)
            raise

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
            
            logger.info(f"Summarizing {len(image_list)} images using Azure OpenAI")
            image_summaries = [self.call_azure_openai_for_image_summary(img, summary_prompt)
                               for img in image_list]
            logger.info(f"Generated {len(image_summaries)} image summaries")
            return image_summaries
        except Exception as e:
            logger.error(f"Error summarizing images: {e}", exc_info=True)
            raise

    def persist_data(self, text_summaries, table_summaries, image_summaries,
                    original_texts, original_tables, original_images):
       """
       Persist summaries in the Chromadb vector store and original data in SQLite.
       """
       def add_documents(summaries, originals, content_type):
           if not summaries or not originals:
               logger.info(f"No {content_type} to persist")
               return
           if len(summaries) != len(originals):
               logger.error(
                   f"CRITICAL MISMATCH: Cannot persist {content_type}. "
                   f"Number of summaries ({len(summaries)}) does not match "
                   f"number of originals ({len(originals)}). Aborting add for this type."
               )
               return 

           logger.info(f"Generating {len(summaries)} unique IDs for {content_type} documents...")
           # Ensure doc_ids are strings, as Chroma expects. uuid.uuid4() returns UUID objects.
           doc_ids = [str(uuid.uuid4()) for _ in range(len(summaries))]


           logger.info(f"Preparing {len(summaries)} metadata entries for {content_type}...")
           metadatas = [{"doc_id": doc_ids[i], "type": content_type} for i in range(len(summaries))]

           try:
               logger.info(f"Adding {len(summaries)} '{content_type}' summaries to Chromadb collection '{self.vectorstore.name}'...")
               # Ensure summaries are strings
               valid_summaries = [str(s) for s in summaries if isinstance(s, (str, bytes)) and str(s).strip()]
               valid_metadatas = [metadatas[i] for i, s in enumerate(summaries) if isinstance(s, (str, bytes)) and str(s).strip()]
               valid_doc_ids = [doc_ids[i] for i, s in enumerate(summaries) if isinstance(s, (str, bytes)) and str(s).strip()]
               
               if not valid_summaries:
                   logger.warning(f"No valid (non-empty string) summaries found for {content_type}. Skipping add to Chromadb.")
               else:
                   self.vectorstore.add(
                       documents=valid_summaries,
                       metadatas=valid_metadatas,
                       ids=valid_doc_ids # Ensure these IDs match the ones used for SQLite
                   )
                   logger.info(f"Successfully added {len(valid_summaries)} '{content_type}' summaries to Chromadb.")
                   logger.info(f"Collection count is now: {self.vectorstore.count()}")

               # Persist the original content in SQLite using all original doc_ids and originals
               # This ensures that even if a summary was empty, the original is stored.
               logger.info(f"Persisting {len(originals)} original '{content_type}' items in SQLite...")
               # Prepare (doc_id, original_content) pairs. Original content should be string.
               sqlite_data = []
               for i in range(len(doc_ids)): # Iterate using the full list of generated doc_ids
                   original_content = originals[i]
                   if not isinstance(original_content, str):
                       original_content = str(original_content) # Ensure it's a string for SQLite
                   sqlite_data.append((doc_ids[i], original_content))

               if sqlite_data:
                   self.docstore.mset(sqlite_data)
                   logger.info(f"Successfully persisted {len(sqlite_data)} original '{content_type}' items in SQLite.")
               else:
                   logger.warning(f"No original {content_type} items to persist in SQLite.")


           except sqlite3.OperationalError as e:
               logger.error(f"SQLite Operational Error during ChromaDB ADD operation: {e}", exc_info=True)
               logger.error(f"This likely means the DB state is inconsistent. Path: {self.vector_db_dir}")
               raise 
           except Exception as e:
               logger.error(f"Error adding {content_type} documents to stores: {e}", exc_info=True)
               raise 

       try:
           logger.info("Persisting text data...")
           add_documents(text_summaries, original_texts, "text")

           logger.info("Persisting table data...")
           add_documents(table_summaries, original_tables, "table")

           logger.info("Persisting image data...")
           add_documents(image_summaries, original_images, "image")

           logger.info("Data persistence process completed.")
           return self.vectorstore, self.docstore

       except Exception as e:
           logger.error(f"Error during data persistence phase: {e}", exc_info=True)
           raise

    def retrieve_data(self, query: str, k: int = 3):
        """
        Retrieve documents by performing a similarity search in the vector store
        and then fetching the original data from the SQLite document store.

        Returns a list of dicts with keys "page_content" and "metadata".
        """
        logger.info(f"Retrieving top {k} documents for query: '{query[:100]}...'") 
        try:
            logger.info(f"Querying Chroma collection '{self.vectorstore.name}'...")
            result = self.vectorstore.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas"] 
            )
            logger.info(f"Chroma query returned {len(result.get('ids', [[]])[0])} results.")

            docs = []
            retrieved_ids = result.get("ids", [[]])[0] 
            retrieved_docs_content = result.get("documents", [[]])[0] # These are the summaries
            retrieved_metadatas = result.get("metadatas", [[]])[0]

            if not retrieved_ids:
                 logger.warning(f"Chroma query for '{query[:50]}...' returned no results.")
                 return [] 

            if not (len(retrieved_ids) == len(retrieved_docs_content) == len(retrieved_metadatas)):
                logger.error("Mismatch in lengths of retrieved ids, documents, and metadatas from Chroma query!")
                return [] 

            logger.info(f"Processing {len(retrieved_ids)} retrieved items from Chroma.")
            doc_id_map = {} 
            for i in range(len(retrieved_ids)):
                chroma_id = retrieved_ids[i] # This is the ID used in Chroma (the UUID)
                summary_text = retrieved_docs_content[i] # This is the summary
                meta = retrieved_metadatas[i]

                # page_content for RAG should ideally be the summary, or potentially original if preferred
                # For now, let's keep 'page_content' as the summary used for retrieval
                doc_entry = {"page_content": summary_text, "metadata": meta}
                docs.append(doc_entry)

                sqlite_doc_id = meta.get("doc_id") # This is the UUID linking to SQLite
                if sqlite_doc_id:
                    doc_id_map[sqlite_doc_id] = None 
                else:
                    logger.warning(f"Retrieved item with Chroma ID '{chroma_id}' is missing 'doc_id' in metadata.")


            sqlite_doc_ids_to_fetch = list(doc_id_map.keys())
            original_data_dict = {}
            if sqlite_doc_ids_to_fetch:
                logger.info(f"Fetching original data for {len(sqlite_doc_ids_to_fetch)} doc_ids from SQLite...")
                try:
                    original_data_dict = self.docstore.get(sqlite_doc_ids_to_fetch)
                    logger.info(f"Successfully fetched {len(original_data_dict)} items from SQLite.")
                except Exception as e:
                    logger.error(f"Error fetching data from SQLite store: {e}", exc_info=True)
            else:
                logger.warning("No valid 'doc_id' found in retrieved metadata to fetch from SQLite.")


            for doc in docs:
                sqlite_doc_id = doc["metadata"].get("doc_id")
                if sqlite_doc_id in original_data_dict:
                    doc["metadata"]["original_data"] = original_data_dict[sqlite_doc_id]
                else:
                    doc["metadata"]["original_data"] = None
                    if sqlite_doc_id: 
                       logger.warning(f"Original data not found in SQLite results for doc_id: {sqlite_doc_id}")


            logger.info(f"Successfully retrieved and processed {len(docs)} documents for the query.")
            return docs

        except sqlite3.OperationalError as e:
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
        texts_from_originals = [] 
        for doc in docs:
            original = doc.get("metadata", {}).get("original_data")
            doc_type = doc.get("metadata", {}).get("type") # Get content type

            if not original:
                continue 

            if doc_type == "image": # If metadata explicitly says it's an image
                if isinstance(original, str):
                    try:
                        base64.b64decode(original, validate=True) # Validate it's proper base64
                        images.append(original)
                        # logger.debug("Identified image based on 'type' and base64 content.")
                    except (base64.binascii.Error, ValueError):
                        logger.warning(f"Original data marked as 'image' but not valid base64: {original[:50]}...")
                else:
                    logger.warning(f"Original data marked as 'image' but not a string: {type(original)}")

            elif doc_type in ("text", "table"): # If metadata says text or table
                if isinstance(original, str):
                    texts_from_originals.append(original)
                    # logger.debug("Identified text/table based on 'type'.")
                elif isinstance(original, bytes):
                     try:
                         texts_from_originals.append(original.decode('utf-8'))
                         # logger.debug("Identified text/table (decoded bytes to utf-8).")
                     except UnicodeDecodeError:
                         logger.warning("Could not decode original_data bytes as UTF-8 text for text/table type. Skipping.")
                else:
                     logger.warning(f"Original data for text/table is not string or bytes. Type: {type(original)}. Skipping.")
            else: # Fallback if type is unknown or other, try to infer
                try:
                    base64.b64decode(original, validate=True)
                    if isinstance(original, str) and original.startswith(('data:image', '/9j/')):
                         images.append(original)
                    elif isinstance(original, bytes): 
                         images.append(base64.b64encode(original).decode('utf-8'))
                    else:
                         if isinstance(original, str): texts_from_originals.append(original)
                except (base64.binascii.Error, ValueError, TypeError):
                    if isinstance(original, str):
                        texts_from_originals.append(original)
                    elif isinstance(original, bytes):
                         try: texts_from_originals.append(original.decode('utf-8'))
                         except UnicodeDecodeError: logger.warning("Fallback: Could not decode bytes as UTF-8. Skipping.")
                    else:
                         logger.warning(f"Fallback: Original data type {type(original)} unhandled. Skipping.")


        logger.info(f"Parsed documents into {len(images)} images and {len(texts_from_originals)} text/table elements based on original_data and type.")
        return {"images": images, "texts": texts_from_originals}
 
    def azure_openai_query(self, message_prompt):
        """
        Send the prompt to Azure OpenAI.
        """
        logger.info(f"Sending prompt to AzureOpenAI ({self.azure_deployment_name})...")
        try:
            # The beta.chat.completions.parse method is not standard.
            # Assuming it's a custom helper or an older SDK version.
            # Standard way for current openai package (v1.x.x onwards):
            response = self.azure_client.chat.completions.create(
                model=self.azure_deployment_name, # This should be the deployment name for GPT-4V or similar
                messages=message_prompt,
                temperature = 0.0,
                # max_tokens, etc. can be added if needed
            )
            if not response.choices:
                    logger.warning("Did not receive any answer from Azure OpenAI")
                    return "Error: No response from Azure OpenAI."        
            response_content = response.choices[0].message.content
            if response_content is None: # Handle cases where content might be None (e.g. finish_reason 'content_filter')
                logger.warning("Azure OpenAI response content is None.")
                logger.warning(f"Finish reason: {response.choices[0].finish_reason}")
                # You might want to inspect response.choices[0].message for more details if it's a non-standard structure
                # or if there's an error object.
                return "Error: Azure OpenAI returned no content (possibly filtered)."
            
            response_content = response_content.strip()
            logger.info(f"Azure OpenAI response received successfully.")
            # logger.debug(f"Azure OpenAI response content: {response_content[:200]}...")
            return response_content
        except Exception as e:
            logger.error(f"Error during Azure OpenAI query: {e}", exc_info=True)
            # Provide a more generic error to the user, but log specifics
            return f"Error communicating with Azure OpenAI: {str(e)}"


    def build_prompt(self, user_query, docs):
        """
        Build a prompt message using the retrieved context (original text and images).
        """
        logger.info(f"Building prompt using retrieved original content...")

        # Parse docs to separate original images and texts/tables
        docs_by_type = self.parse_docs(docs)
        original_texts_for_context = docs_by_type["texts"] # These are actual original texts/tables
        original_images_for_context = docs_by_type["images"]

        context_text = "\n\n".join(original_texts_for_context) 

        if not context_text and not original_images_for_context:
            logger.warning("No valid original content (text/table/image) found in retrieved docs to build context.")
        elif not context_text:
            logger.info("No text/table original content for context, but images might be present.")
        elif not original_images_for_context:
            logger.info("No image original content for context, but text/tables might be present.")


        logger.info(f"Context Text for Prompt (Originals - Snippet): {context_text[:300]}...") 
        logger.info(f"Number of original images for prompt: {len(original_images_for_context)}")

       
        summary_prompt_path = os.path.join(PROMPT_DIR, 'user_query.txt')
        summary_prompt_template = self.load_prompt(summary_prompt_path)

        # Ensure context_text is a string, even if empty
        filled_prompt_text = summary_prompt_template.format(
            context_text=str(context_text), # Ensure it's a string
            user_query=user_query
        )

        # --- Prepare final message list ---
        # System prompt should be general
        system_message = "Responde a la pregunta en español basándote únicamente en el contexto proporcionado (texto e imágenes). Si la respuesta no se encuentra en el contexto, indica que no tienes suficiente información."
        
        # User message content starts with the text part
        user_content_list = [{"type": "text", "text": filled_prompt_text}]

        # Add images to the user message content
        if original_images_for_context:
            logger.info(f"Adding {len(original_images_for_context)} images to the prompt's user message.")
            for img_b64 in original_images_for_context:
                 try:
                     base64.b64decode(img_b64, validate=True) # Validate before adding
                     user_content_list.append({
                         "type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                     })
                 except (base64.binascii.Error, ValueError) as decode_err:
                     logger.error(f"Failed to decode/validate base64 image data for prompt: {decode_err}. Skipping image.")

        prompt_message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content_list} # Content is now a list
        ]
        
        # logger.debug(f"Final prompt message structure for Azure: {prompt_message}")
        return prompt_message


    def process_user_query(self, 
                           user_query:str, 
                           k:int = 3):
        """
        Retrieve relevant context from the stores, build a prompt, and send it to Azure OpenAI.
        """
        logger.info(f"Processing user query: {user_query}")
        # Retrieve data (summaries + metadata linking to originals)
        self.retrieved_docs = self.retrieve_data(user_query, k=k)
        if not self.retrieved_docs:
            logger.warning("No documents retrieved for the query. Cannot build prompt.")
            # Return a message indicating no context was found
            return "No se encontró información relevante en los documentos para responder a tu pregunta."

        # Build prompt using original content from retrieved_docs
        prompt_message = self.build_prompt(user_query, self.retrieved_docs)
        
        # Send prompt to Azure OpenAI (which handles multimodal input)
        response = self.azure_openai_query(prompt_message)
        return response

    def get_retrieved_context_data(self):
        """
        Returns the ORIGINAL context text and image data from the retrieved documents
        stored in self.retrieved_docs.
        """
        if not self.retrieved_docs:
             logger.warning("get_retrieved_context_data called but self.retrieved_docs is empty.")
             return [], [] 

        logger.info(f"Extracting original context for display from {len(self.retrieved_docs)} retrieved documents.")
        
        # Use parse_docs to get original texts and images directly
        parsed_originals = self.parse_docs(self.retrieved_docs)
        
        original_texts_for_display = parsed_originals["texts"]
        context_img_for_display = parsed_originals["images"]

        logger.info(f"Returning {len(original_texts_for_display)} original text/table contexts and {len(context_img_for_display)} image contexts for display.")

        return original_texts_for_display, context_img_for_display
     
    def clear_chroma_data(self):
        import shutil
        # Check if the directory exists before attempting to remove it
        if os.path.exists(self.vector_db_dir):
            shutil.rmtree(self.vector_db_dir, ignore_errors=True)
            logger.info(f"ChromaDB directory '{self.vector_db_dir}' removed.")
            # Recreate the directory for future use by PersistentClient
            os.makedirs(self.vector_db_dir, exist_ok=True)
            logger.info(f"ChromaDB directory '{self.vector_db_dir}' recreated.")
        else:
            logger.info(f"ChromaDB directory '{self.vector_db_dir}' does not exist. No need to remove.")
        
        # Re-initialize the Chroma client and collection to ensure a clean state
        # This is important because just deleting files might not be enough if client holds state.
        try:
            chroma_settings = Settings(anonymized_telemetry=False, is_persistent=True)
            self.chroma_client = PersistentClient(path=self.vector_db_dir, settings=chroma_settings)
            collection_name = "multi_modal_rag"
            self.vectorstore = self.chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
            logger.info("ChromaDB client and collection re-initialized after clearing data.")
        except Exception as e:
            logger.error(f"Error re-initializing ChromaDB after clearing data: {e}", exc_info=True)
            # This could leave the system in an inconsistent state for ChromaDB operations.
            raise # Re-raise to make it evident there's an issue.


    def clear_sqlite_data(self): 
        # The SQLiteStore.clear_sqlite_data seems to clear a 'files' table,
        # but our docstore table is named 'docstore'.
        # Let's adapt it to clear the 'docstore' table.
        try:
            conn = sqlite3.connect(self.sqlite_db_path)
            cursor = conn.cursor()
            
            # Clear the 'docstore' table
            cursor.execute("DELETE FROM docstore;")
            conn.commit()
            logger.info("Successfully cleared the 'docstore' table data in SQLite.")
            
            # Optionally, vacuum to reclaim space, though may not be necessary for small dbs
            # cursor.execute("VACUUM;")
            # conn.commit()
            
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"SQLite error clearing 'docstore' table: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error clearing 'docstore' table: {e}", exc_info=True)
            raise

    
    def clear_databases(self):
        logger.info(f"Attempting to clear ChromaDB data...")
        try:
            self.clear_chroma_data()
            logger.info(f"ChromaDB data cleared successfully.")
        except Exception as e:
            logger.error(f"Failed to clear ChromaDB data: {e}", exc_info=True)
            # Decide if you want to stop or continue to clear SQLite
            # For now, let's continue

        logger.info(f"Attempting to clear SQLite data...")
        try:
            self.clear_sqlite_data()
            logger.info(f"SQLite data cleared successfully.")
        except Exception as e:
            logger.error(f"Failed to clear SQLite data: {e}", exc_info=True)
import os
from dotenv import load_dotenv
import logging
import base64
import uuid
import chromadb
from chromadb.config import Settings
from src.sqlite_processor import SQLiteStore
#from sqlite_processor import SQLiteStore
import openai
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


class EmbeddingsGenerator:
    """
    A class for generating embeddings and summaries from text, tables, and images.
    Uses various LLM models for processing different types of content.
    """

    def __init__(self,
                 embedding_model: str = 'openai',
                 vector_db_dir: str = VECTOR_DB_DIR,
                 sqlite_db_path: str = SQL_DB_DIR
                 ):
        """
        Initialize the embeddings generator.
        
        Args:
            vector_db_dir (str): Directory to save vector database
            embedding_model (str): Type of embedding model ('openai' or 'nomic-embed-text')
        """
        self.vector_db_dir = vector_db_dir
        self.sqlite_db_path = sqlite_db_path
        self.embedding_model = embedding_model
        self.text_summaries = []
        self.retrieved_docs = []
        self.id_key= "doc_id"
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(vector_db_dir, exist_ok=True)
            logger.info(f"Vector database directory: {vector_db_dir}")
            os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)
            logger.info(f"SQLite database path: {sqlite_db_path}")

            # Initialize the embedding function
            if embedding_model == 'openai':
                if not OPENAI_API_KEY:
                    raise ValueError("OpenAI API Key not found in environment variables")
                # Set the API key for openai
                openai.api_key = OPENAI_API_KEY
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                self.embedding_fn = self.get_embedding
                logger.info("Initialized OpenAI embedding function")
            else:
                raise NotImplementedError(f"Embedding model '{embedding_model}' not implemented")


            # Initialize Chroma client and collection directly
            chroma_settings = Settings(
                persist_directory=self.vector_db_dir,
                anonymized_telemetry=False  
            )
            self.chroma_client = chromadb.Client(settings=chroma_settings)
            self.vectorstore = self.chroma_client.get_or_create_collection(
                name="multi_modal_rag",
                embedding_function=self.embedding_fn
            )
            logger.info("Initialized Chromadb vector store")

            # Initialize SQLite document store using custom implementation
            self.docstore = SQLiteStore(db_path=self.sqlite_db_path)
            logger.info("Initialized SQLite document store")

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
            
        except ValueError as e:
            logger.error(f"Initialization error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
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
       def add_documents(summaries, originals, content_type):
           if not summaries or not originals:
               logger.info(f"No {content_type} to persist")
               return
           if len(summaries) != len(originals):
               logger.warning(
                   f"Mismatch between {content_type} summaries ({len(summaries)}) and originals ({len(originals)})"
               )
           # Generate unique IDs for each document
           logger.info(f"Generating unique IDs for each document...")
           doc_ids = [str(uuid.uuid4()) for _ in range(len(summaries))]
           # Prepare metadata for each document
           logger.info(f"Preparing metadata for each document...")
           metadatas = [{"doc_id": doc_ids[i], "type": content_type} for i in range(len(summaries))]
           # Add summaries to Chromadb
           logger.info(f"Adding summaries to Chromadb..")
           self.vectorstore.add(
               documents=summaries,
               metadatas=metadatas,
               ids=doc_ids
           )
           # Persist the original content in SQLite
           logger.info(f"Originales: {originals}")
           logger.info(f"Persisting the original content in SQLite...")
           self.docstore.mset(list(zip(doc_ids, originals)))
           logger.info(f"{content_type.capitalize()} persisted: {len(summaries)} documents")
              
       logger.info(f"Persisting text summaries...")
       add_documents(text_summaries, original_texts, "text")
       logger.info(f"Persisting table summaries...")
       add_documents(table_summaries, original_tables, "table")
       logger.info(f"Persisting image summaries...")
       add_documents(image_summaries, original_images, "image")
       # Return vectorstore and docstore for further retrieval if needed.
       return self.vectorstore, self.docstore

 
    def retrieve_data(self, query: str, k: int = 3):
        """
        Retrieve documents by performing a similarity search in the vector store
        and then fetching the original data from the SQLite document store.
        
        Returns a list of dicts with keys "page_content" and "metadata".
        """
        try:
            # Query Chromadb directly (the query returns a dict with lists)
            result = self.vectorstore.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas"]
            )
            docs = []
            # TODO: Assume result format: {"documents": [[...]], "metadatas": [[...]]}
            for doc_text, meta in zip(result["documents"][0], result["metadatas"][0]):
                docs.append({"page_content": doc_text, "metadata": meta})
            # Retrieve original data from SQLite using stored doc_ids
            doc_ids = [d["metadata"].get("doc_id") for d in docs if d["metadata"].get("doc_id")]
            if doc_ids:
                original_data_dict = self.docstore.get(doc_ids)
                for doc in docs:
                    doc_id = doc["metadata"].get("doc_id")
                    logger.info(f"***\nOriginal data: {doc_id}: {original_data_dict.get(doc_id)}")
                    doc["metadata"]["original_data"] = original_data_dict.get(doc_id)
            logger.info(f"Retrieved {len(docs)} documents for query: '{query}'.")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving data for query '{query}': {e}")
            raise

    def parse_docs(self, docs):
        """
        Split documents into images and texts based on whether the original data
        can be base64-decoded.
        """
        images = []
        texts = []
        for doc in docs:
            original = doc["metadata"].get("original_data", "")
            try:
                # If base64 decoding succeeds, assume it's an image
                base64.b64decode(original)
                images.append(original)
            except Exception:
                texts.append(original)
        return {"images": images, "texts": texts}

    def get_context_text_content(self):
        """
        Extract the text content from the retrieved documents.
        """
        if not self.retrieved_docs:
            logger.warning("No retrieved documents found.")
            return []
        return [doc["page_content"] for doc in self.retrieved_docs]

 
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
        Build a prompt message using the retrieved context (text and images).
        """
        logger.info(f"Building prompt...")
        docs_by_type = self.parse_docs(docs)
        # Combine text contents from documents as context
        context_text = " ".join([doc["page_content"]
                                 for doc in docs 
                                 if doc["metadata"].get("type") in ("text", "table")])
        
        summary_prompt_path = os.path.join(PROMPT_DIR, 'user_query.txt')
        summary_prompt_template = self.load_prompt(summary_prompt_path)

        filled_prompt = summary_prompt_template.format(
            context_text=context_text,
            user_query=user_query
        )
        prompt_message = [
            {
                "role": "system",
                "content": "Responde a la pregunta en español basándote únicamente en el contexto proporcionado."
            },
            {"role": "user", "content": filled_prompt}
        ]
        # If images are present, attach them 
        if docs_by_type["images"]:
            for img in docs_by_type["images"]:
                prompt_message.append({
                    "role": "user",
                    "content": [
                        #{"type": "text", "text": "Analiza la siguiente imagen:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                    ]
                })
        logger.info(f"Final prompt: {prompt_message}")
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
        logger.info(f"Retrieved docs: {self.retrieved_docs}")
        prompt_message = self.build_prompt(user_query, self.retrieved_docs)
        # Send prompt
        response = self.azure_openai_query(prompt_message)
        return response

    def get_retrieved_context_data(self):    
        """
        Returns the context text and image data from the retrieved documents.
        """    
        docs_by_type = self.parse_docs(self.retrieved_docs)
        context_text = self.get_context_text_content() # docs_by_type['texts']
        context_img  = docs_by_type['images']
        return context_text, context_img
    
 
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
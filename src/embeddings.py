import os
from dotenv import load_dotenv
import base64
import logging
from src.sqlite_processor import SQLiteStore

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import uuid
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(vector_db_dir, exist_ok=True)
            logger.info(f"Vector database directory: {vector_db_dir}")
            os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)
            logger.info(f"SQLite database path: {sqlite_db_path}")

            # Initialize embeddings model
            if embedding_model == 'openai':
                if not OPENAI_API_KEY:
                    raise ValueError("OpenAI API Key not found in environment variables")
                    
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=OPENAI_API_KEY
                    #openai_api_key=OPENAI_API_KEY
                )
                logger.info(f"Initialized OpenAI embedding model: text-embedding-3-small")
            else:
                raise NotImplementedError(f"Embedding model '{embedding_model}' not implemented")

            # Initialize Chroma vector store with Langchain
            vectorstore = Chroma(
                collection_name="multi_modal_rag",
                embedding_function=self.embeddings,
                persist_directory=self.vector_db_dir
            )
            logger.info("Initialized Chroma vector store")

            # Initialize SQLite document store using custom implementation
            #docstore = SQLiteStore(db_path=self.sqlite_db_path)
            #logger.info("Initialized SQLite document store")

            docstore = InMemoryStore()
            logger.info("Initialized document store")

            # The retriever (empty to start)
            self.id_key= "doc_id"
            self.retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                docstore=docstore,
                id_key=self.id_key,
            )
            
            
        except ValueError as e:
            logger.error(f"Initialization error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            raise
    
    def count_docstore_elements(self):
        """
        Counts the number of elements stored in the docstore.
            
        Returns:
            int: number of elements stored in the docstore.
        """
        return sum(1 for _ in self.retriever.docstore.yield_keys())
    
    def load_prompt(self, file_path):
        """
        Load a prompt from a file.
        
        Args:
            file_path (str): Path to the prompt file
            
        Returns:
            str: The prompt text
            
        Raises:
            FileNotFoundError: If the prompt file does not exist
            IOError: If there is an error reading the file
        """
        try:
            logger.debug(f"Loading prompt from: {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Prompt file not found: {file_path}")
                
            with open(file_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()                
            return prompt
        except FileNotFoundError as e:
            logger.error(f"Prompt file not found: {e}")
            raise
        except IOError as e:
            logger.error(f"Error reading prompt file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading prompt: {e}")
            raise

    def summary_text_and_tables(self, text_list, table_list):
        """
        Generate summaries for text and tables.
        
        Args:
            text_list (list): List of texts to summarize
            table_list (list): List of tables to summarize
            
        Returns:
            tuple: (text_summaries, table_summaries)
            
        Raises:
            ValueError: If API keys are missing
            RuntimeError: If summarization fails
        """
        try:
            if not GROQ_API_KEY:
                raise ValueError("Groq API Key not found in environment variables")
                
            # Load summary prompt
            summary_prompt_path = os.path.join(PROMPT_DIR, 'summary_text.txt')
            summary_prompt = self.load_prompt(summary_prompt_path)

            prompt = ChatPromptTemplate.from_template(summary_prompt)
            
            # Initialize the model
            logger.info("Initializing Groq model for summarization")
            model = ChatGroq(
                temperature=0.5, 
                model="llama-3.1-8b-instant",
                api_key=GROQ_API_KEY
            )
            
            # Create summarization chain
            summarize_chain = (
                {"element": lambda x: x} | prompt | model | StrOutputParser()
            )
            
            # Process texts if present
            text_summaries = []
            if text_list:
                logger.info(f"Summarizing {len(text_list)} text elements")
                text_summaries = summarize_chain.batch(
                    text_list, 
                    {"max_concurrency": 3}
                )
                logger.info(f"Generated {len(text_summaries)} text summaries")
            
            # Process tables if present
            table_summaries = []
            if table_list:
                logger.info(f"Summarizing {len(table_list)} tables")
                table_summaries = summarize_chain.batch(
                    table_list, 
                    {"max_concurrency": 3}
                )
                logger.info(f"Generated {len(table_summaries)} table summaries")
            
            return text_summaries, table_summaries
            
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise
        except FileNotFoundError as e:
            logger.error(f"Prompt file error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error summarizing text and tables: {e}")
            raise RuntimeError(f"Summarization failed: {e}")
    
    def summary_images(self, image_list):
        """
        Generate summaries for base64-encoded images.
        
        Args:
            image_list (list): List of base64-encoded images
            
        Returns:
            list: List of image summaries
            
        Raises:
            ValueError: If API keys are missing
            RuntimeError: If image summarization fails
        """
        try:
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API Key not found in environment variables")
                
            if not image_list:
                logger.info("No images to summarize")
                return []
                
            # Load summary prompt
            summary_prompt_path = os.path.join(PROMPT_DIR, 'summary_image.txt')
            summary_prompt = self.load_prompt(summary_prompt_path)
            
            logger.info(f"Summarizing {len(image_list)} images with GPT-4o-mini")
            
            # Process each image
            processed_images = []
            for image in image_list:
                try:
                    # Validate base64 format
                    base64.b64decode(image)
                    
                    # Create message template
                    messages = [
                        (
                            "user",
                            [
                                {"type": "text", "text": summary_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                                },
                            ],
                        )
                    ]
                    processed_images.append(messages)
                except Exception as e:
                    logger.warning(f"Skipping invalid image: {str(e)[:100]}...")
            
            # Create the chain
            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY) | StrOutputParser()
            
            # Process images in batch
            image_summaries = chain.batch(processed_images)
            logger.info(f"Generated {len(image_summaries)} image summaries")
            
            return image_summaries
            
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise
        except FileNotFoundError as e:
            logger.error(f"Prompt file error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error summarizing images: {e}")


    def persist_data_langchain(self, text_summaries, table_summaries, image_summaries,
                               original_texts, original_tables, original_images):
        """
        Persist summaries in the Chroma vector store and original data in SQLite.
        Generates a unique identifier for each document, creates Document objects,
        and associates the originals in the document store.
        
        Args:
            text_summaries (list): List of text summaries
            table_summaries (list): List of table summaries
            image_summaries (list): List of image summaries
            original_texts (list): List of original texts
            original_tables (list): List of original tables
            original_images (list): List of original images
            
        Returns:
            tuple: (vectorstore, docstore) - The configured stores
            
        Raises:
            RuntimeError: If persistence fails
        """
        try:
            id_key = self.id_key

            def add_documents(summaries, originals, content_type):
                """Helper function to add documents to both stores"""
                if not summaries or not originals:
                    logger.info(f"No {content_type} to persist")
                    return
                
                if len(summaries) != len(originals):
                    logger.warning(
                        f"Mismatch between {content_type} summaries ({len(summaries)}) " 
                        f"and originals ({len(originals)})"
                    )
                
                # Generate unique IDs for each document
                doc_ids = [str(uuid.uuid4()) for _ in range(len(summaries))]
                
                # Create Document objects with summaries and metadata
                docs = [
                    Document(page_content=summary, metadata={id_key: doc_ids[i], "type": content_type})
                    for i, summary in enumerate(summaries)
                ]
                
                # Persist summaries in Chroma vector store
                self.retriever.vectorstore.add_documents(docs)                
                # Persist original data in docstore
                self.retriever.docstore.mset(list(zip(doc_ids, originals)))
                
                logger.info(f"{content_type.capitalize()} persisted: {len(docs)} documents")

            # Process each content type
            add_documents(text_summaries, original_texts, "text")
            add_documents(table_summaries, original_tables, "table")
            add_documents(image_summaries, original_images, "image")

            # Return both components for retrieval
            return self.retriever

        except Exception as e:
            logger.error(f"Error persisting data using Langchain: {e}")
            raise RuntimeError(f"Data persistence failed: {e}")



    def retrieve_data(self, query: str, k: int = 3):
        """
        Recupera documentos relevantes para la consulta a partir del vectorstore y asocia los datos originales almacenados en SQLite.
        
        Args:
            query (str): La consulta para la búsqueda.
            k (int): Número de documentos a recuperar.
        
        Returns:
            list: Lista de objetos Document con la metadata extendida que incluye los datos originales.
        """
        try:
            # Realizar búsqueda en el vectorstore de Chroma
            docs = self.vectorstore.similarity_search(query, k=k)
            
            # Para cada documento, recuperar los datos originales desde SQLite utilizando el doc_id
            for doc in docs:
                doc_id = doc.metadata.get("doc_id")
                if doc_id:
                    original_data = self.retriever.docstore.get(doc_id)
                    # Se añade el contenido original a la metadata del documento
                    doc.metadata["original_data"] = original_data
            logger.info(f"Retrieved {len(docs)} documents for query: '{query}'.")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving data for query '{query}': {e}")
            raise


    def parse_docs(self, docs):
        """Split base64-encoded images and texts"""
        b64 = []
        text = []
        for doc in docs:
            try:
                base64.b64decode(doc)
                b64.append(doc)
            except Exception as e:
                text.append(doc)
        return {"images": b64, "texts": text}

    def build_prompt(self, kwargs):
    
        docs_by_type = kwargs["context"]
        user_question = kwargs["question"]
    
        context_text = ""
        if len(docs_by_type["texts"]) > 0:
            for text_element in docs_by_type["texts"]:
                context_text += text_element.text
    
        # construct prompt with context (including images)
        prompt_template = f"""
        Responde la pregunta basándote únicamente en el siguiente contexto, que puede incluir texto, tablas e imágenes.
        Contexto: {context_text}
        Pregunta: {user_question}
        """
    
        prompt_content = [{"type": "text", "text": prompt_template}]
    
        if len(docs_by_type["images"]) > 0:
            for image in docs_by_type["images"]:
                prompt_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                )
    
        return ChatPromptTemplate.from_messages(
            [
                HumanMessage(content=prompt_content),
            ]
        )

    def process_user_query(self, query:str):
        logger.info(f"Processing query: {query}")
        chain_with_sources = {
                "context": self.retriever | RunnableLambda(self.parse_docs),
                "question": RunnablePassthrough(),
            } | RunnablePassthrough().assign(
                  response=(
                    RunnableLambda(self.build_prompt)
                    | ChatOpenAI(model="gpt-4o-mini")
                    | StrOutputParser()
                )
            )
        
        response = chain_with_sources.invoke(query)
        logger.info(f"Chain executed")

        return response
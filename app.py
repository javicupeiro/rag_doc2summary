#!/usr/bin/env python3
"""
Streamlit application for PDF processing, embedding generation, and interactive chat.
This app allows users to upload PDFs, process them, and chat with the extracted content.
"""

import os
import uuid
import logging
import streamlit as st
import tempfile
from dotenv import load_dotenv
import base64
from PIL import Image
import io

from src.pdf_processor import PDFProcessor
from src.embeddings import EmbeddingsGenerator


# The next code avoids some problems during init time between Streamlit and Torch
# Fix the path attribute of torch.classes to prevent problematic inspection
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="PDF Processor & RAG Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define directory paths
@st.cache_resource
def get_directories():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'data', 'pdfs')
    output_dir = os.path.join(base_dir, 'data', 'markdown')
    vector_db_dir = os.path.join(base_dir, 'data', 'database', 'vector_db')
    sqlite_db_path = os.path.join(base_dir, 'data', 'database', 'sql', 'docstore.db')
    
    # Ensure directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vector_db_dir, exist_ok=True)
    os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)
    
    return {
        "base_dir": base_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "vector_db_dir": vector_db_dir,
        "sqlite_db_path": sqlite_db_path
    }

# Initialize the PDF processor and embeddings generator
@st.cache_resource
def initialize_processors(dirs):
    try:
        processor = PDFProcessor(
            input_dir=dirs["input_dir"], 
            output_dir=dirs["output_dir"]
        )
        
        embeddings_generator = EmbeddingsGenerator(
            embedding_model='openai',
            vector_db_dir=dirs["vector_db_dir"],
            sqlite_db_path=dirs["sqlite_db_path"]
        )
        
        return processor, embeddings_generator
    except Exception as e:
        logger.error(f"Error initializing processors: {e}", exc_info=True)
        return None, None

# Function to display base64 image
def display_image(base64_image):
    """Display a base64 encoded image."""
    try:
        logger.info(f"Displaying image in streamlit")
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        st.image(image, use_column_width=True)
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

# Function to process PDF
def process_pdf(file_path, processor, embeddings_generator):
    """Process a PDF file and generate embeddings."""
    with st.spinner("Processing PDF..."):
        success = processor.process_one_pdf(file_path)
        
        if not success:
            st.error(f"Failed to process the PDF file.")
            return False
        
        # Get processed data
        text_chunks = processor.get_chunked_text()
        table_chunks = processor.get_chunked_tables()
        image_chunks = processor.get_chunked_images_b64()

        # Display results
        st.success(f"✅ Extracted {len(text_chunks)} text chunks")
        st.success(f"✅ Extracted {len(table_chunks)} tables")
        st.success(f"✅ Extracted {len(image_chunks)} images")
        
        # Display a sample of each type in an expander
        with st.expander("Preview Extracted Content"):
            # Show a text sample
            if text_chunks:
                st.subheader("Sample Text")
                st.text(str(text_chunks[0])[:500] + "..." if len(str(text_chunks[0])) > 500 else str(text_chunks[0]))
            
            # Show a table sample
            if table_chunks:
                st.subheader("Sample Table")
                st.text(table_chunks[0][:500] + "..." if len(table_chunks[0]) > 500 else table_chunks[0])
            
            # Show an image sample
            if image_chunks:
                st.subheader("Sample Image")
                for img in image_chunks:
                    display_image(img)
        
        # Generate summaries
        with st.spinner("Generating summaries..."):
            try:
                text_summaries, table_summaries = embeddings_generator.summary_text_and_tables(
                    text_chunks, table_chunks
                )
                
                st.success(f"✅ Generated {len(text_summaries)} text summaries")
                st.success(f"✅ Generated {len(table_summaries)} table summaries")
                
                # Generate image summaries if images are available
                image_summaries = []
                if image_chunks:
                    image_summaries = embeddings_generator.summary_images(image_chunks)
                    st.success(f"✅ Generated {len(image_summaries)} image summaries")
                
                # Persist the data using Chroma and SQLite
                with st.spinner("Persisting data to vector store and document store..."):
                    retriever = embeddings_generator.persist_data(
                        text_summaries=text_summaries,
                        table_summaries=table_summaries,
                        image_summaries=image_summaries,
                        original_texts=text_chunks,
                        original_tables=table_chunks,
                        original_images=image_chunks
                    )
                    
                    st.success("✅ Successfully persisted data to vector store and document store")
                    st.session_state['pdf_processed'] = True
                    return True
                    
            except Exception as e:
                st.error(f"Error in summary generation or persistence: {str(e)}")
                logger.error(f"Error in summary generation or persistence: {e}", exc_info=True)
                return False


def chat_with_documents(user_query, embeddings_generator):
    """Retrieve and display relevant document chunks based on the query."""
    try:        
        results = embeddings_generator.process_user_query(user_query)
        print(f"Result: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}", exc_info=True)
        return f"Error processing your query: {str(e)}"


# Main function
def main():
    """
    Main Streamlit application.
    """
    # Get directories
    dirs = get_directories()
    
    # Initialize processors
    processor, embeddings_generator = initialize_processors(dirs)
    
    # Application title
    st.title("📄 PDF Processor & RAG Chat")
    
    # Initialize session state variables
    if 'pdf_processed' not in st.session_state:
        st.session_state['pdf_processed'] = False
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Create tabs
    tab1, tab2 = st.tabs(["📤 Upload & Process", "💬 Chat with Documents"])
    
    # Tab 1: Upload & Process
    with tab1:
        if processor is None or embeddings_generator is None:
            st.error("Error: Failed to initialize processors. Please check your API keys in the .env file.")
        else:
            st.header("Upload & Process PDF")
            
            # Option to choose existing PDF or upload new one
            upload_option = st.radio(
                "Choose PDF source:",
                ["Upload New PDF", "Use Existing PDF"]
            )
            
            if upload_option == "Upload New PDF":
                uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
                
                if uploaded_file:
                    # Save the uploaded file to a temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        pdf_path = tmp_file.name
                    
                    if st.button("Process PDF", type="primary"):
                        success = process_pdf(pdf_path, processor, embeddings_generator)
                        
                        # Clean up the temporary file
                        os.unlink(pdf_path)
            
            else:  # Use existing PDF
                # List available PDFs in the input directory
                available_pdfs = []
                try:
                    for file in os.listdir(dirs["input_dir"]):
                        if file.lower().endswith('.pdf'):
                            available_pdfs.append(file)
                except Exception as e:
                    st.error(f"Error listing PDF files: {str(e)}")
                
                if available_pdfs:
                    selected_pdf = st.selectbox("Select a PDF", available_pdfs)
                    
                    if st.button("Process PDF", type="primary"):
                        pdf_path = os.path.join(dirs["input_dir"], selected_pdf)
                        success = process_pdf(pdf_path, processor, embeddings_generator)
                else:
                    st.info("No PDFs found in the input directory.")
            
            # Document store status
            if embeddings_generator:
                st.subheader("Document Store Status")
                # Button to clear all stored data
                st.warning("⚠️ Esta acción eliminará todos los documentos procesados de las bases de datos.")
                if st.button("🗑️ Borrar toda la información de la base de datos"):
                    try:                        
                        embeddings_generator.clear_databases()
                        st.success("✅ Todas las bases de datos han sido limpiadas exitosamente.")
                        st.session_state['pdf_processed'] = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error al borrar las bases de datos: {str(e)}")

                try:
                    doc_count = embeddings_generator.count_docstore_elements()
                    st.info(f"Document store contains {doc_count} documents.")
                except Exception as e:
                    st.warning(f"Could not retrieve document count: {str(e)}")
    
    # Tab 2: Chat
    with tab2:
        st.header("Chat with Documents")
        
        if processor is None or embeddings_generator is None:
            st.error("Error: Failed to initialize processors. Please check your API keys in the .env file.")
        else:
            # Check if documents are available for chat
            doc_count = 0
            try:
                doc_count = embeddings_generator.count_docstore_elements()
            except:
                pass
                
            if doc_count == 0 and not st.session_state['pdf_processed']:
                st.info("No documents have been processed yet. Please upload and process a PDF first.")
            else:
                # Chat controls
                col1, col2 = st.columns([4, 1])
                with col2:
                    if st.button("Clear Chat History"):
                        st.session_state['chat_history'] = []
                        st.rerun()
                
                # Display chat history
                for message in st.session_state['chat_history']:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Get user query
                user_query = st.chat_input("Ask a question about your documents...")
                
                if user_query:
                    # Add user message to chat history
                    st.session_state['chat_history'].append({"role": "user", "content": user_query})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(user_query)
                    
                    # Process the query and display results
                    with st.chat_message("assistant"):
                        with st.spinner("Searching documents..."):
                            response = chat_with_documents(user_query, embeddings_generator)
                            # Display LLM response        
                            #response_text = format_response_text(response) 
                            st.markdown(response)

                            # Get context data
                            text_context, img_context = embeddings_generator.get_retrieved_context_data()
                            # Display context text
                            print(f"text_context: {text_context}")
                            # Display context image
                            print(f"img_context: {img_context}")
                            for img in img_context:
                                display_image(img)

                            # Add assistant message to chat history
                            st.session_state['chat_history'].append({"role": "assistant", "content": response})

                            
                            
if __name__ == "__main__":
    main()
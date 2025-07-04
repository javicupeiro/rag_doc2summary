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
try:
    import torch
    if hasattr(torch, 'classes') and hasattr(torch.classes, '__file__'):
         torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
    else:
         # Handle cases where torch.classes might not be structured as expected
         pass # Or add specific logging/handling if needed
except ImportError:
    pass # Torch not installed or import failed


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

# --- Caching Resources ---
@st.cache_resource
def get_directories():
    """Gets and creates necessary directories. Cached."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'data', 'pdfs')
    output_dir = os.path.join(base_dir, 'data', 'markdown')
    # Adjusted ChromaDB path to be a directory as PersistentClient expects
    vector_db_dir = os.path.join(base_dir, 'data', 'database', 'vector_db')
    sqlite_db_path = os.path.join(base_dir, 'data', 'database', 'sql', 'docstore.db')

    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Input PDF directory: {input_dir}")
    logger.info(f"Output Markdown directory: {output_dir}")
    logger.info(f"Vector DB directory: {vector_db_dir}")
    logger.info(f"SQLite DB path: {sqlite_db_path}")

    # Ensure directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vector_db_dir, exist_ok=True) # Ensure vector DB directory itself exists
    os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True) # Ensure parent dir for SQLite exists

    return {
        "base_dir": base_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "vector_db_dir": vector_db_dir, # Use the dedicated directory path
        "sqlite_db_path": sqlite_db_path
    }

@st.cache_resource
def initialize_processors(_dirs): # Use _dirs to indicate it's accessing the cached dict
    """Initializes PDFProcessor and EmbeddingsGenerator. Cached."""
    try:
        logger.info("Initializing PDFProcessor...")
        processor = PDFProcessor(
            input_dir=_dirs["input_dir"],
            output_dir=_dirs["output_dir"]
        )
        logger.info("PDFProcessor initialized.")

        logger.info("Initializing EmbeddingsGenerator...")
        # Ensure correct paths are passed from the cached dictionary
        embeddings_generator = EmbeddingsGenerator(
            embedding_model='openai',
            vector_db_dir=_dirs["vector_db_dir"],
            sqlite_db_path=_dirs["sqlite_db_path"]
        )
        logger.info("EmbeddingsGenerator initialized.")

        return processor, embeddings_generator
    except Exception as e:
        # Log the error and display a user-friendly message in Streamlit
        logger.error(f"Fatal error initializing processors: {e}", exc_info=True)
        st.error(f"Failed to initialize application components. Please check logs and API keys. Error: {e}")
        # Return None to indicate failure, which should be checked in main()
        return None, None

# --- Helper Functions ---
def display_image(base64_image, caption=""):
    """Display a base64 encoded image with optional caption."""
    try:
        # logger.debug(f"Attempting to display image (first 50 chars): {base64_image[:50]}...")
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        st.image(image, caption=caption, use_container_width=True)
        # logger.debug("Image displayed successfully.")
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")
        logger.error(f"Error decoding or displaying image: {e}", exc_info=True)

# --- Core Logic Functions ---
def process_pdf(file_path, processor, embeddings_generator, summarization_model_choice):
    """Process a PDF file: extract, summarize, and persist."""
    if not processor or not embeddings_generator:
         st.error("Processors not initialized correctly.")
         return False

    try:
        st.info(f"Starting processing for: {os.path.basename(file_path)}")
        with st.spinner("Step 1/4: Extracting content from PDF..."):
            # Clear previous chunks from processor instance before processing a new PDF
            processor.texts = []
            processor.tables = []
            processor.images_b64 = []
            
            success = processor.process_one_pdf(file_path)
            if not success:
                st.error(f"Failed to process the PDF file.")
                return False
            text_chunks = processor.get_chunked_text()
            table_chunks = processor.get_chunked_tables()
            image_chunks = processor.get_chunked_images_b64()
            st.success(f"✅ Extracted: {len(text_chunks)} text, {len(table_chunks)} table, {len(image_chunks)} image chunks.")

        # Display preview
        with st.expander("Preview Extracted Content (Samples)"):
            if text_chunks: st.text_area("Sample Text", text_chunks[0][:500]+"...", height=100)
            if table_chunks: st.text_area("Sample Table (as text)", table_chunks[0][:500]+"...", height=100)
            if image_chunks: display_image(image_chunks[0], caption="Sample Image")

        with st.spinner(f"Step 2/4: Generating summaries using {summarization_model_choice}..."):
            text_summaries, table_summaries = embeddings_generator.summary_text_and_tables(
                text_chunks, 
                table_chunks,
                summarization_model_choice=summarization_model_choice # Pass the choice here
            )
            st.success(f"✅ Summarized: {len(text_summaries)} text, {len(table_summaries)} table chunks.")
            image_summaries = []
            if image_chunks:
                image_summaries = embeddings_generator.summary_images(image_chunks) # Image summarization is fixed to Azure OpenAI for now
                st.success(f"✅ Summarized: {len(image_summaries)} image chunks.")

        with st.spinner("Step 3/4: Persisting data to stores..."):
             # Note: persist_data returns the stores, but retriever is often managed within the class now
             _, _ = embeddings_generator.persist_data(
                text_summaries=text_summaries, table_summaries=table_summaries, image_summaries=image_summaries,
                original_texts=text_chunks, original_tables=table_chunks, original_images=image_chunks
            )
             st.success("✅ Data persisted successfully.")

        with st.spinner("Step 4/4: Finalizing..."):
             st.session_state['pdf_processed'] = True # Mark as processed
             st.balloons()
        st.success("PDF Processing Complete!")
        return True

    except Exception as e:
        st.error(f"An error occurred during PDF processing: {str(e)}")
        logger.error(f"Error in process_pdf function: {e}", exc_info=True)
        return False


def chat_with_documents(user_query, embeddings_generator):
    """Handles the RAG process: retrieve context, query LLM."""
    if not embeddings_generator:
        return "Error: Embeddings generator not initialized."
    try:
        # 1. Get LLM response using the RAG pipeline
        logger.info(f"Processing user query via RAG: {user_query}")
        llm_response = embeddings_generator.process_user_query(user_query)
        logger.info("RAG processing complete.")

        # 2. Get the context used for the response (now retrieves originals)
        logger.info("Retrieving context data used for the response...")
        text_context, img_context = embeddings_generator.get_retrieved_context_data()
        logger.info(f"Retrieved context: {len(text_context)} text/table items, {len(img_context)} images.")

        # 3. Return all parts needed for display
        return {
            "answer": llm_response,
            "text_context": text_context,
            "image_context": img_context
        }

    except Exception as e:
        logger.error(f"Error during chat_with_documents: {e}", exc_info=True)
        # Return a dictionary indicating error, preserving structure
        return {
            "answer": f"Sorry, an error occurred processing your query: {str(e)}",
            "text_context": [],
            "image_context": []
        }

# --- Streamlit UI ---
def main():
    """Main Streamlit application layout and logic."""
    st.title("📄 PDF Processor & RAG Chat")
    st.caption("Upload PDFs, extract content, and chat using Retrieval-Augmented Generation.")

    # --- Initialization and Sidebar ---
    dirs = get_directories() # Get paths first
    # Initialize processors (cached) using the obtained directories
    # Using the placeholder '_' for the argument name passed to the cached function
    processor, embeddings_generator = initialize_processors(dirs)

    with st.sidebar:
        st.header("⚙️ Options")
        # Add any configuration options here later (e.g., select LLM model, k value)

        # Document store status display
        st.subheader("📊 Document Store Status")
        if embeddings_generator:
            try:
                doc_count = embeddings_generator.count_docstore_elements()
                st.metric(label="Stored Documents (Chunks)", value=doc_count)
            except Exception as e:
                st.warning(f"Could not get document count: {e}")
                logger.warning(f"Failed to get docstore count: {e}", exc_info=True)

            st.warning("Clearing databases will remove all processed content.")
            if st.button("⚠️ Clear All Databases", key="clear_db_button"):
                try:
                    with st.spinner("Clearing databases..."):
                        embeddings_generator.clear_databases()
                    st.success("✅ Databases cleared successfully.")
                    st.session_state['pdf_processed'] = False
                    st.session_state['chat_history'] = [] # Clear chat history too
                    st.rerun() # Rerun to reflect changes
                except Exception as e:
                    st.error(f"❌ Error clearing databases: {str(e)}")
                    logger.error(f"Error clearing databases: {e}", exc_info=True)
        else:
            st.error("Processor initialization failed. Cannot display DB status.")


    # --- Initialize Session State ---
    if 'pdf_processed' not in st.session_state:
        st.session_state['pdf_processed'] = False
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [] # Store dicts: {"role": "user/assistant", "content": message, "context": context_dict (optional)}
    if 'selected_summarization_model' not in st.session_state:
        st.session_state['selected_summarization_model'] = "Groq (Llama3-8b)" # Default value


    # --- Main App Tabs ---
    tab1, tab2 = st.tabs(["📤 Sube y Procesa un PDF", "💬 Chatea con los Documentos"])

    # == Tab 1: Upload & Process ==
    with tab1:
        st.header("1. Select or Upload PDF")
        if processor is None or embeddings_generator is None:
            # Error message already shown by initialize_processors if it fails
            st.warning("Application components failed to load. Please check setup and logs.")
        else:
            pdf_to_process = None
            upload_option = st.radio(
                "Choose PDF source:",
                ["Upload New PDF", "Use Existing PDF in `data/pdfs`"],
                horizontal=True, key="pdf_source_radio"
            )

            if upload_option == "Upload New PDF":
                uploaded_file = st.file_uploader("Select PDF file", type=["pdf"], key="pdf_uploader")
                if uploaded_file:
                    # Save temporarily to process
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        pdf_to_process = tmp_file.name # Path to temp file
                    st.info(f"Selected for upload: {uploaded_file.name}")

            else: # Use Existing PDF
                try:
                    available_pdfs = [f for f in os.listdir(dirs["input_dir"]) if f.lower().endswith('.pdf')]
                    if available_pdfs:
                        selected_pdf = st.selectbox("Select existing PDF:", available_pdfs, key="existing_pdf_select")
                        if selected_pdf:
                            pdf_to_process = os.path.join(dirs["input_dir"], selected_pdf)
                            st.info(f"Selected existing: {selected_pdf}")
                    else:
                        st.info("No PDF files found in the `data/pdfs` directory.")
                except Exception as e:
                    st.error(f"Error listing PDF files: {str(e)}")
                    logger.error(f"Error listing existing PDFs: {e}", exc_info=True)
            
            st.header("2. Choose Summarization Model for Texts/Tables")
            summarization_model_choice = st.selectbox(
                "Elige el modelo para resúmenes de texto/tabla:",
                options=["Groq (Llama3-8b)", "GPT4o"],
                index=["Groq (Llama3-8b)", "GPT4o"].index(st.session_state.get('selected_summarization_model', "Groq (Llama3-8b)")),
                key="summarization_model_select"
            )
            st.session_state['selected_summarization_model'] = summarization_model_choice


            st.header("3. Process Selected PDF")
            if pdf_to_process:
                process_button_disabled = False
            else:
                 process_button_disabled = True
                 st.warning("Please select or upload a PDF file first.")

            if st.button("🚀 Process PDF", type="primary", key="process_pdf_button", disabled=process_button_disabled):
                if pdf_to_process:
                    # Process the selected/uploaded PDF
                    success = process_pdf(
                        pdf_to_process, 
                        processor, 
                        embeddings_generator,
                        st.session_state['selected_summarization_model'] # Pass the selected model
                    )
                    # Clean up temp file if it was created from upload
                    if upload_option == "Upload New PDF" and os.path.exists(pdf_to_process):
                        try:
                            os.unlink(pdf_to_process)
                            logger.info(f"Removed temporary file: {pdf_to_process}")
                        except OSError as e:
                             logger.error(f"Error removing temporary file {pdf_to_process}: {e}")

                    if success:
                         # Maybe switch to chat tab automatically?
                         # st.experimental_set_query_params(tab="chat") # Example, syntax might change
                         pass
                else:
                     st.error("No PDF file was selected or uploaded.")

    # == Tab 2: Chat ==
    with tab2:
        st.header("💬 Chat with Your Documents")

        if processor is None or embeddings_generator is None:
             st.warning("Application components failed to load. Chat is unavailable.")
        else:
            # Check if documents are available
            doc_count = 0
            try:
                doc_count = embeddings_generator.count_docstore_elements()
            except Exception:
                 pass # Ignore error if count fails, main check below

            if doc_count == 0 and not st.session_state.get('pdf_processed', False):
                 st.info("No documents found in the database. Please process a PDF on the 'Upload & Process PDF' tab first.")
                 if not st.session_state.get('pdf_processed', False): # Only stop if not even one PDF processed in session
                    st.stop()

            # --- Chat Interface ---
            st.subheader("Conversation")
            if st.button("Clear Chat History", key="clear_chat_button"):
                st.session_state['chat_history'] = []
                st.rerun()

            # Display chat history
            # Use a container for chat messages for better scrolling perhaps
            chat_container = st.container()
            with chat_container:
                for i, message_data in enumerate(st.session_state['chat_history']):
                    role = message_data["role"]
                    content = message_data["content"] # The user query or the LLM answer text
                    context = message_data.get("context") # Optional context dictionary

                    with st.chat_message(role):
                         st.markdown(content) # Display the main message content
                         # If it's an assistant message AND has context, display it in an expander
                         if role == "assistant" and context:
                              with st.expander("Show Context Used"):
                                   st.markdown("**Text/Table Context:**")
                                   if context.get("text_context"):
                                        bullets_md = "\n".join(f"- {str(point)[:200]}..." if len(str(point)) > 200 else str(point) for point in context["text_context"])
                                        st.markdown(bullets_md)
                                   else:
                                        st.markdown("_No text/table context found or used._")

                                   st.markdown("**Image Context:**")
                                   if context.get("image_context"):
                                        img_cols = st.columns(min(3, len(context["image_context"]))) # Max 3 images per row
                                        for idx, img_b64 in enumerate(context["image_context"]):
                                             with img_cols[idx % 3]:
                                                  display_image(img_b64, caption=f"Context Image {idx+1}")
                                   else:
                                        st.markdown("_No image context found or used._")


            # --- Chat Input (Rendered after history) ---
            user_query = st.chat_input("Ask a question about the processed documents...")

            # --- Process New Query ---
            if user_query:
                 # 1. Add and display user message immediately
                 st.session_state['chat_history'].append({"role": "user", "content": user_query})
                 with st.chat_message("user"):
                      st.markdown(user_query)

                 # 2. Process query and get response + context
                 with st.chat_message("assistant"):
                      message_placeholder = st.empty() # Placeholder for streaming later
                      message_placeholder.markdown("Thinking... 🤔")
                      with st.spinner("Retrieving context and generating response..."):
                          response_data = chat_with_documents(user_query, embeddings_generator)

                      # Display final answer
                      message_placeholder.markdown(response_data["answer"])

                      # Display context in expander (similar to history display)
                      if response_data["text_context"] or response_data["image_context"]:
                          with st.expander("Show Context Used"):
                               st.markdown("**Text/Table Context:**")
                               if response_data["text_context"]:
                                    bullets_md = "\n".join(f"- {str(point)[:200]}..." if len(str(point)) > 200 else str(point) for point in response_data["text_context"])
                                    st.markdown(bullets_md)
                               else:
                                    st.markdown("_No text/table context found or used._")

                               st.markdown("**Image Context:**")
                               if response_data["image_context"]:
                                    img_cols = st.columns(min(3, len(response_data["image_context"]))) # Max 3 images per row
                                    for idx, img_b64 in enumerate(response_data["image_context"]):
                                         with img_cols[idx % 3]:
                                              display_image(img_b64, caption=f"Context Image {idx+1}")
                               else:
                                    st.markdown("_No image context found or used._")


                 # 3. Add assistant message (with context) to history
                 st.session_state['chat_history'].append({
                       "role": "assistant",
                       "content": response_data["answer"],
                       "context": { # Store context with the message for later display
                            "text_context": response_data["text_context"],
                            "image_context": response_data["image_context"]
                       }
                 })

                 # No explicit rerun needed here, Streamlit will rerun automatically
                 # because the state ('chat_history') changed implicitly before the script ended.
                 # However, sometimes an explicit rerun helps ensure rendering updates cleanly.
                 # Consider adding if visual updates seem inconsistent: st.rerun()


if __name__ == "__main__":
    # Basic check for essential environment variables
    essential_keys = ["OPENAI_API_KEY", "GROQ_API_KEY", "AZURE_API_KEY", "AZURE_ENDPOINT", "AZURE_DEPLOYMENT_NAME", "AZURE_API_VERSION"]
    missing_keys = [key for key in essential_keys if not os.getenv(key)]
    if missing_keys:
        logger.warning(f"Missing essential environment variables: {', '.join(missing_keys)}. Application might not function correctly.")
        # st.warning(f"Warning: Missing environment variables: {', '.join(missing_keys)}") # Show warning in UI too

    main()
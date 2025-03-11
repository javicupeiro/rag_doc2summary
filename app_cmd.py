#!/usr/bin/env python3
"""
Main application file to demonstrate PDF processing and embedding functionality.
This script initializes and uses the PDFProcessor and EmbeddingsGenerator classes
with persistence to Chroma vectorstore and SQLite document store.
"""

import os
import uuid
import logging
from dotenv import load_dotenv

from src.pdf_processor import PDFProcessor
from src.embeddings import EmbeddingsGenerator

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

def main():
    """
    Main function to initialize and run the PDF processor and embeddings generator.
    Demonstrates the entire pipeline: PDF processing, summary generation, and persistence.
    """
    try:
        # Get directory paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(base_dir, 'data', 'pdfs')
        output_dir = os.path.join(base_dir, 'data', 'markdown')
        vector_db_dir = os.path.join(base_dir, 'data', 'database', 'vector_db')
        sqlite_db_path = os.path.join(base_dir, 'data', 'database', 'sql', 'docstore.db')
        
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Vector DB directory: {vector_db_dir}")
        logger.info(f"SQLite DB path: {sqlite_db_path}")
        
        # Initialize PDF processor
        processor = PDFProcessor(input_dir=input_dir, 
                                 output_dir=output_dir)
        
        # Initialize Embeddings generator with vector and SQLite paths
        embeddings_generator = EmbeddingsGenerator(
            embedding_model='openai',
            vector_db_dir=vector_db_dir,
            sqlite_db_path=sqlite_db_path
        )
        
        # Option 1: Process all PDFs in the input directory
        # num_processed = processor.process_all_pdfs()
        # logger.info(f"Processed {num_processed} PDF files")
        
        # Option 2: Process a specific PDF file
        pdf_filename = "RECETA_DE_LA_PAELLA.pdf"  # Replace with your PDF filename
        pdf_path = os.path.join(input_dir, pdf_filename)
        
        if os.path.exists(pdf_path):
            logger.info(f"Processing file: {pdf_path}")
            success = processor.process_one_pdf(pdf_path)
            
            if success:
                # Get processed data
                text_chunks = processor.get_chunked_text()
                table_chunks = processor.get_chunked_tables()
                image_chunks = processor.get_chunked_images_b64()

                # Display results
                logger.info(f"Extracted {len(text_chunks)} text chunks")
                logger.info(f"Extracted {len(table_chunks)} tables")
                logger.info(f"Extracted {len(image_chunks)} images")
                
                # Generate summaries
                try:
                    logger.info("Generating text and table summaries...")
                    text_summaries, table_summaries = embeddings_generator.summary_text_and_tables(
                        text_chunks, table_chunks
                    )
                    
                    logger.info(f"Generated {len(text_summaries)} text summaries")
                    logger.info(f"Generated {len(table_summaries)} table summaries")
                    
                    # Print first summary as example
                    if text_summaries:
                        logger.info(f"Example text summary: {text_summaries[0][:100]}...")
                        
                    # Generate image summaries if images are available
                    image_summaries = []
                    if image_chunks:
                        logger.info("Generating image summaries...")
                        image_summaries = embeddings_generator.summary_images(image_chunks)
                        logger.info(f"Generated {len(image_summaries)} image summaries")
                        
                        # Print first image summary as example
                        if image_summaries:
                            logger.info(f"Example image summary: {image_summaries[0][:100]}...")
                    
                    # Persist the data using Langchain with Chroma and SQLite
                    logger.info("Persisting data to vector store and document store...")
                    vectorstore, docstore = embeddings_generator.persist_data_langchain(
                        text_summaries=text_summaries,
                        table_summaries=table_summaries,
                        image_summaries=image_summaries,
                        original_texts=text_chunks,
                        original_tables=table_chunks,
                        original_images=image_chunks
                    )
                    
                    logger.info(f"Successfully persisted data to vector store and document store")
                    
                    # Optional: perform a sample query to test retrieval
                    # This can be uncommented if you want to test the retrieval functionality
                    
                    if vectorstore:
                        logger.info("Testing retrieval with a sample query...")
                        query = "What is this document about?"
                        results = vectorstore.similarity_search(query, k=3)
                        logger.info(f"Found {len(results)} relevant chunks for query: '{query}'")
                        
                        for i, doc in enumerate(results):
                            doc_id = doc.metadata.get("doc_id")
                            logger.info(f"Result {i+1} - ID: {doc_id}")
                            logger.info(f"Summary: {doc.page_content[:100]}...")
                            
                            # Retrieve original content from SQLite
                            if doc_id and docstore:
                                original = docstore.get(doc_id)
                                if original:
                                    logger.info(f"Original content available: {len(str(original))} characters")
                    
                    
                except Exception as e:
                    logger.error(f"Error in summary generation or persistence: {e}", exc_info=True)
                
                # Save extracted text to file if needed
                # output_file = os.path.join(output_dir, f"{pdf_filename.split('.')[0]}_text.txt")
                # processor.save_text_to_file(output_file)
            else:
                logger.error(f"Failed to process {pdf_path}")
        else:
            logger.error(f"File not found: {pdf_path}")
            logger.info("Available PDF files:")
            try:
                for file in os.listdir(input_dir):
                    if file.lower().endswith('.pdf'):
                        logger.info(f"  - {file}")
            except Exception as e:
                logger.error(f"Error listing PDF files: {e}")
    
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
    except Exception as e:
        logger.error(f"Error in main application: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("Starting PDF processing and embedding application")
    main()
    logger.info("Processing completed")
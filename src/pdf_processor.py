import os
from dotenv import load_dotenv
import base64
from IPython.display import Image, display
from unstructured.partition.pdf import partition_pdf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PDFProcessor:
    """
    A class for processing PDF files to extract text, tables, and images.
    Uses the unstructured library to partition PDFs into meaningful chunks.
    """

    def __init__(self, input_dir='../data/pdfs/', output_dir='../data/database/'):
        """
        Initialize the PDF processor.
        
        Args:
            input_dir (str): Directory containing PDF files
            output_dir (str): Directory to save processed markdown files
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.texts = []
        self.tables = []
        self.images_b64 = []

        if output_dir and not os.path.exists(output_dir):
            logger.warning(f"Output directory '{output_dir}' does not exist. Creating it.")
            try:
                os.makedirs(output_dir)
            except OSError as e:
                logger.error(f"Failed to create output directory: {e}")
    
    def process_all_pdfs(self):
        """
        Process all PDF files in the input directory.

        Returns:
            int: Number of files processed
        """
        pass

    def _process_chunks(self, chunks):
        """
        Process the chunks extracted from a PDF.
        
        Args:
            chunks (list): List of chunks from partition_pdf
        """
        try:
            for chunk in chunks:
                # Check if this is a composite element (contains multiple elements)
                if "CompositeElement" in str(type(chunk)):
                    # Add text to the text list
                    self.texts.append(chunk.text)
                    
                    # Process original elements in the chunk
                    chunk_els = chunk.metadata.orig_elements
                    for el in chunk_els:
                        # Extract tables
                        if "Table" in str(type(el)):
                            self.tables.append(el.text)                        
                        #Extract images as base64
                        if "Image" in str(type(el)):
                           self.images_b64.append(el.metadata.image_base64)
        except AttributeError as e:
            logger.error(f"Error accessing chunk attributes: {e}")
        except Exception as e:
            logger.error(f"Error processing chunks: {e}")

    # def _process_chunks(self, chunks):
    #     """
    #     Process the chunks or raw elements extracted from a PDF.

    #     Args:
    #         chunks (list): List of elements from partition_pdf
    #     """
    #     try:
    #         for chunk in chunks:
    #             chunk_type = str(type(chunk))

    #             # CompositeElement (cuando hay chunking)
    #             if "CompositeElement" in chunk_type:
    #                 if hasattr(chunk, "text"):
    #                     self.texts.append(chunk.text)
    #                 if hasattr(chunk.metadata, "orig_elements"):
    #                     for el in chunk.metadata.orig_elements:
    #                         el_type = str(type(el))
    #                         if "Table" in el_type:
    #                             self.tables.append(el.text)
    #                         elif "Image" in el_type and hasattr(el.metadata, "image_base64"):
    #                             self.images_b64.append(el.metadata.image_base64)

    #             # Procesamiento plano
    #             else:
    #                 if "Table" in chunk_type and hasattr(chunk, "text"):
    #                     self.tables.append(chunk.text)
    #                 elif "Image" in chunk_type and hasattr(chunk.metadata, "image_base64"):
    #                     logger.info(f"chunk.metadata attributes: {vars(chunk.metadata)}")

    #                     #if getattr(chunk.metadata, "image_width_px", 1000) > 500: #filtrar imágenes por tamaño
    #                     #    logger.info(f"image_width_px: {...}")
    #                     #    self.images_b64.append(chunk.metadata.image_base64)
    #                 elif ("Text" in chunk_type or "Title" in chunk_type) and hasattr(chunk, "text"):
    #                     self.texts.append(chunk.text)

    #     except Exception as e:
    #         logger.error(f"Error processing chunks: {e}")



    def get_chunked_text(self):
        """
        Get the extracted text chunks.
        
        Returns:
            list: List of text chunks
        """
        return self.texts

    def get_chunked_tables(self):
        """
        Get the extracted tables.
        
        Returns:
            list: List of tables
        """
        return self.tables

    def get_chunked_images_b64(self):
        """
        Get the extracted images as base64 strings.
        
        Returns:
            list: List of base64-encoded images
        """
        return self.images_b64

    def process_one_pdf(self, file_path):
        """
        Process one input PDF files in the input directory.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        if not file_path:
            logger.error("No file path provided")
            return False
            
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        try:
            logger.info(f"Partitioning PDF: {file_path}")
            
            # Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
            # Reference2: https://docs.unstructured.io/api-reference/partition/extract-image-block-types
            chunks = partition_pdf(
                filename=file_path,
                infer_table_structure=True,  # Extract tables
                strategy="hi_res",  # Mandatory to infer tables
                extract_image_block_types=["Image", "Table"],  # TODO: Add 'Table' to list to extract image of tables
                # image_output_dir_path=output_path,  # If None, images and tables will be saved in base64
                extract_image_block_to_payload=True,  # If true, will extract base64 for API usage
                chunking_strategy="by_title",  # Alternative: 'basic'
                max_characters=10000,  # Defaults to 500
                combine_text_under_n_chars=2000,  # Defaults to 0
                new_after_n_chars=6000,
                # extract_images_in_pdf=True,  # Deprecated
            )

            
            
            # Process the extracted chunks
            self._process_chunks(chunks)
            return True
            
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return False
        except ValueError as e:
            logger.error(f"Invalid parameter: {e}")
            return False
        except Exception as e:
            logger.error(f"Error processing PDF '{file_path}': {e}")
            return False

    def display_base64_image(self, base64_code):
        """
        Display an image from its base64 encoding.
        
        Args:
            base64_code (str): Base64-encoded image data
        """
        try:
            # Decode the base64 string to binary
            image_data = base64.b64decode(base64_code)
            
            # Display the image
            display(Image(data=image_data))
        except base64.binascii.Error:
            logger.error("Invalid base64 encoding")
        except Exception as e:
            logger.error(f"Error displaying image: {e}")

    def print_tables_as_txt(self):
        """
        Print all extracted tables in text format.
        """
        if not self.tables:
            logger.info("No tables extracted")
            return
            
        for idx, table in enumerate(self.tables):
            print(f"Table {idx + 1}:\n{table}\n\n")

    def print_tables_as_html(self):
        """
        Print all extracted tables in HTML format.
        
        Note: This requires the tables to have HTML metadata.
        """
        if not self.tables:
            logger.info("No tables extracted")
            return
            
        try:
            for idx, table in enumerate(self.tables):
                if hasattr(table, 'metadata') and hasattr(table.metadata, 'text_as_html'):
                    print(f"Table {idx + 1}:\n{table.metadata.text_as_html}\n\n")
                else:
                    print(f"Table {idx + 1}: HTML format not available\n\n")
        except AttributeError:
            logger.error("Tables don't have HTML metadata")
        except Exception as e:
            logger.error(f"Error printing tables as HTML: {e}")
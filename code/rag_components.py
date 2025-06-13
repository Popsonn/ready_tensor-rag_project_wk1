import logging
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma             
from langchain_core.documents import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LehningerDocumentSplitter:
    """Document splitter specifically designed for Lehninger Biochemistry textbook."""
    
    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize the document splitter.
        
        Args:
            max_chunk_size (int): Maximum size for secondary splitting
            chunk_overlap (int): Overlap between chunks for secondary splitting
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        
        if chunk_overlap >= max_chunk_size:
            logger.warning(f"chunk_overlap ({chunk_overlap}) should be less than max_chunk_size ({max_chunk_size})")
        
        self.headers_to_split_on = [
            ("#", "Main Title"),
            ("##", "Section"),
        ]
        
        try:
            self.markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.headers_to_split_on,
                strip_headers=False
            )
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            logger.info(f"Document splitter initialized with max_chunk_size={max_chunk_size}, chunk_overlap={chunk_overlap}")
            
        except Exception as e:
            logger.error(f"Failed to initialize document splitters: {e}")
            raise
    
    def split_single_document(self, markdown_text: str, filename: Optional[str] = None) -> List[Document]:
        """
        Split a single markdown document.
        
        Args:
            markdown_text (str): The markdown content
            filename (str): Optional filename for metadata
            
        Returns:
            list: List of Document objects
        """
        if not markdown_text.strip():
            logger.warning(f"Empty or whitespace-only content for file: {filename}")
            return []
            
        try:
            md_header_splits = self.markdown_splitter.split_text(markdown_text)
            logger.debug(f"Generated {len(md_header_splits)} header-based splits for {filename}")
            
            final_splits = []
            
            for doc in md_header_splits:
                if filename:
                    doc.metadata['source_file'] = filename
                
                # Check if document needs secondary splitting
                if len(doc.page_content) > self.max_chunk_size:
                    sub_splits = self.text_splitter.split_documents([doc])
                    final_splits.extend(sub_splits)
                    logger.debug(f"Split large chunk into {len(sub_splits)} smaller chunks")
                else:
                    final_splits.append(doc)
            
            logger.debug(f"Final split count for {filename}: {len(final_splits)}")
            return final_splits
            
        except Exception as e:
            logger.error(f"Error splitting document {filename}: {e}")
            return []
    
    def process_directory(self, data_dir: str) -> List[Document]:
        """
        Process all markdown files in a directory.
        
        Args:
            data_dir (str): Path to directory containing .md files
            
        Returns:
            list: List of all Document objects from all files
        """
        all_splits = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.error(f"Directory not found: {data_dir}")
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        
        md_files = list(data_path.glob("*.md"))
        
        if not md_files:
            logger.warning(f"No .md files found in {data_dir}")
            return all_splits
        
        logger.info(f"Found {len(md_files)} markdown files in {data_dir}")
        
        for md_file in md_files:
            logger.info(f"Processing: {md_file.name}")
            
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.strip():
                    logger.warning(f"File {md_file.name} is empty or contains only whitespace")
                    continue
                
                splits = self.split_single_document(content, md_file.name)
                all_splits.extend(splits)
                
                logger.info(f"Generated {len(splits)} chunks from {md_file.name}")
                
            except UnicodeDecodeError as e:
                logger.error(f"Encoding error reading {md_file.name}: {e}")
            except IOError as e:
                logger.error(f"IO error reading {md_file.name}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing {md_file.name}: {e}")
        
        logger.info(f"Total chunks generated: {len(all_splits)}")
        return all_splits
    
    def get_chunks_summary(self, splits: List[Document]) -> Dict[str, Any]:
        """
        Get a summary of the chunks for analysis.
        
        Args:
            splits (list): List of Document objects
            
        Returns:
            dict: Summary information
        """
        if not splits:
            logger.warning("No splits provided for summary")
            return {
                'total_chunks': 0,
                'files': {},
                'sections': {},
                'size_stats': {'min': 0, 'max': 0, 'avg': 0, 'total': 0}
            }
        
        summary = {
            'total_chunks': len(splits),
            'files': {},
            'sections': {},
            'size_stats': {
                'min': float('inf'),
                'max': 0,
                'avg': 0,
                'total': 0
            }
        }
        
        sizes = []
        
        for doc in splits:
            size = len(doc.page_content)
            sizes.append(size)
            
            source_file = doc.metadata.get('source_file', 'Unknown')
            summary['files'][source_file] = summary['files'].get(source_file, 0) + 1
            
            section = doc.metadata.get('Section', 'No Section')
            summary['sections'][section] = summary['sections'].get(section, 0) + 1
        
        if sizes:
            summary['size_stats']['min'] = min(sizes)
            summary['size_stats']['max'] = max(sizes)
            summary['size_stats']['avg'] = sum(sizes) // len(sizes)
            summary['size_stats']['total'] = sum(sizes)
        
        logger.debug(f"Generated summary for {len(splits)} splits")
        return summary


def get_langchain_huggingface_embeddings(
    model_name: str = "all-MiniLM-L6-v2", 
    device: Optional[str] = None
) -> HuggingFaceEmbeddings:
    """
    Initialize and return a LangChain-compatible HuggingFace Embeddings object.
    
    Args:
        model_name (str): The name of the HuggingFace model to use
        device (str, optional): The device to load the model on ('cpu', 'cuda')
    
    Returns:
        HuggingFaceEmbeddings: A LangChain embeddings object
        
    Raises:
        Exception: If model initialization fails
    """
    logger.info(f"Initializing HuggingFace embeddings with model: {model_name}")
    
    try:
        model_kwargs = {}
        if device is not None:
            model_kwargs['device'] = device
            
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name, 
            model_kwargs=model_kwargs
        )
        
        logger.info(f"HuggingFace embeddings loaded successfully. Model: {model_name}, Device: {device}")
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace embeddings: {e}")
        raise


def get_langchain_chroma_vector_store(
    embedding_function: HuggingFaceEmbeddings, 
    persist_directory: str = "./biochem_chroma_db", 
    collection_name: str = "lehninger_biochem",
    reset_db: bool = False
) -> Chroma:
    """
    Initialize and return a LangChain-compatible Chroma vector store.
    
    Args:
        embedding_function: A LangChain-compatible embedding function
        persist_directory (str): Directory to persist the database
        collection_name (str): Name of the collection
        reset_db (bool): If True, deletes and recreates the collection
        
    Returns:
        Chroma: A LangChain Chroma vector store instance
        
    Raises:
        OSError: If directory operations fail
        Exception: If vector store initialization fails
    """
    logger.info(f"Initializing ChromaDB at: {persist_directory} for collection: {collection_name}")
    
    # Ensure the directory exists
    db_path = Path(persist_directory)
    
    try:
        db_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {persist_directory}: {e}")
        raise

    if reset_db and db_path.exists() and any(db_path.iterdir()): 
        logger.info(f"Clearing existing ChromaDB at {persist_directory}")
        try:
            shutil.rmtree(persist_directory)
            db_path.mkdir(parents=True, exist_ok=True) 
            logger.info("ChromaDB directory cleared successfully")
        except OSError as e:
            logger.error(f"Error clearing directory {persist_directory}: {e}")
            raise

    try:
        vector_store = Chroma(
            persist_directory=str(db_path), 
            collection_name=collection_name,
            embedding_function=embedding_function 
        )
        
        current_count = vector_store._collection.count()
        logger.info(f"ChromaDB collection '{collection_name}' ready. Current document count: {current_count}")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to initialize Chroma vector store: {e}")
        raise
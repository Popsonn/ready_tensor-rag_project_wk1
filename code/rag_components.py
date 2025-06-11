from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma             
from langchain_core.documents import Document                   
import chromadb 
import numpy as np 
from typing import List, Dict, Any, Optional, Union
import uuid
import json
import os
from pathlib import Path
import shutil 

class LehningerDocumentSplitter:
    def __init__(self, max_chunk_size=1000, chunk_overlap=100):
        """
        
        Args:
            max_chunk_size (int): Maximum size for secondary splitting
            chunk_overlap (int): Overlap between chunks for secondary splitting
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.headers_to_split_on = [
            ("#", "Main Title"),
            ("##", "Section"),
        ]
        
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def split_single_document(self, markdown_text, filename=None):
        """
        Split a single markdown document.
        
        Args:
            markdown_text (str): The markdown content
            filename (str): Optional filename for metadata
            
        Returns:
            list: List of Document objects
        """
        md_header_splits = self.markdown_splitter.split_text(markdown_text)
        
        final_splits = []
        
        for doc in md_header_splits:
            if filename:
                doc.metadata['source_file'] = filename
            
            # Check if document needs secondary splitting
            if len(doc.page_content) > self.max_chunk_size:
                sub_splits = self.text_splitter.split_documents([doc])
                final_splits.extend(sub_splits)
            else:
                final_splits.append(doc)
        
        return final_splits
    
    def process_directory(self, data_dir):
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
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        
        md_files = list(data_path.glob("*.md"))
        
        if not md_files:
            print(f"No .md files found in {data_dir}")
            return all_splits
        
        print(f"Found {len(md_files)} markdown files:")
        
        for md_file in md_files:
            print(f"  - {md_file.name}")
            
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                splits = self.split_single_document(content, md_file.name)
                all_splits.extend(splits)
                
                print(f"    → Generated {len(splits)} chunks")
                
            except Exception as e:
                print(f"    → Error processing {md_file.name}: {e}")
        
        print(f"\nTotal chunks generated: {len(all_splits)}")
        return all_splits
    
    def get_chunks_summary(self, splits):
        """
        Get a summary of the chunks for analysis.
        
        Args:
            splits (list): List of Document objects
            
        Returns:
            dict: Summary information
        """
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
            if source_file not in summary['files']:
                summary['files'][source_file] = 0
            summary['files'][source_file] += 1
            
            section = doc.metadata.get('Section', 'No Section')
            if section not in summary['sections']:
                summary['sections'][section] = 0
            summary['sections'][section] += 1
        
        if sizes:
            summary['size_stats']['min'] = min(sizes)
            summary['size_stats']['max'] = max(sizes)
            summary['size_stats']['avg'] = sum(sizes) // len(sizes)
            summary['size_stats']['total'] = sum(sizes)
        
        return summary


# for HuggingFace Embeddings ---
def get_langchain_huggingface_embeddings(model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None) -> HuggingFaceEmbeddings:
    """
    Initializes and returns a LangChain-compatible HuggingFace Embeddings object.
    This object can then be passed directly to LangChain's vector store classes.
    
    Args:
        model_name (str): The name of the HuggingFace model to use (e.g., "all-MiniLM-L6-v2").
        device (str, optional): The device to load the model on (e.g., 'cpu', 'cuda').
                                If None, LangChain will try to infer.
    
    Returns:
        HuggingFaceEmbeddings: A LangChain embeddings object.
    """
    print(f"Initializing LangChain HuggingFaceEmbeddings with model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device})
    print(f"LangChain HuggingFaceEmbeddings loaded. Model: {embeddings.model_name}, Device: {device}")
    return embeddings


# function for Chroma Vector Store ---
def get_langchain_chroma_vector_store(
    embedding_function: HuggingFaceEmbeddings, 
    persist_directory: str = "./biochem_chroma_db", 
    collection_name: str = "lehninger_biochem",
    reset_db: bool = False
) -> Chroma:
    """
    Initializes and returns a LangChain-compatible Chroma vector store.
    
    Args:
        embedding_function: A LangChain-compatible embedding function (e.g., HuggingFaceEmbeddings instance).
        persist_directory (str): Directory to persist the database.
        collection_name (str): Name of the collection.
        reset_db (bool): If True, deletes and recreates the collection.
        
    Returns:
        Chroma: A LangChain Chroma vector store instance.
    """
    print(f"Initializing LangChain ChromaDB at: {persist_directory} for collection: {collection_name}")
    
    # Ensure the directory exists
    db_path = Path(persist_directory)
    db_path.mkdir(parents=True, exist_ok=True)

    if reset_db and db_path.exists() and len(list(db_path.iterdir())) > 0: 
        print(f"Clearing existing ChromaDB at {persist_directory}...")
        try:
            shutil.rmtree(persist_directory)
            db_path.mkdir(parents=True, exist_ok=True) 
            print("ChromaDB directory cleared.")
        except OSError as e:
            print(f"Error clearing directory {persist_directory}: {e}")
            raise

    vector_store = Chroma(
        persist_directory=str(db_path), 
        collection_name=collection_name,
        embedding_function=embedding_function 
    )
    
    print(f"LangChain ChromaDB collection '{collection_name}' ready. Current document count: {vector_store._collection.count()}")
    return vector_store
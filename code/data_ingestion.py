#!/usr/bin/env python3

import logging
import os
import sys
import json 
from pathlib import Path
from typing import Dict, Optional, List, Any

from rag_components import (
    LehningerDocumentSplitter,
    get_langchain_huggingface_embeddings,
    get_langchain_chroma_vector_store
)
import config 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Chapter metadata configuration - could be moved to config file if needed
CHAPTER_METADATA_MAP = {
    "Introduction.md": {
        "chapter_title": "Biosynthesis of Amino Acids, Nucleotides, and Related Molecules",
        "chapter_number": "22",
        "section_root": "22.0 Introduction" 
    },
    "Overview of Nitrogen Metabolism.md": {
        "chapter_title": "Biosynthesis of Amino Acids, Nucleotides, and Related Molecules",
        "chapter_number": "22",
        "section_root": "22.1 Overview of Nitrogen Metabolism" 
    },
    "Biosynthesis of Amino Acids.md": {
        "chapter_title": "Biosynthesis of Amino Acids, Nucleotides, and Related Molecules",
        "chapter_number": "22",
        "section_root": "22.2 Biosynthesis of Amino Acids"
    },
    "Molecules Derived from Amino Acids.md": {
        "chapter_title": "Biosynthesis of Amino Acids, Nucleotides, and Related Molecules",
        "chapter_number": "22",
        "section_root": "22.3 Molecules Derived from Amino Acids"
    },
    "Biosynthesis and Degradation of Nucleotides.md": {
        "chapter_title": "Biosynthesis of Amino Acids, Nucleotides, and Related Molecules",
        "chapter_number": "22",
        "section_root": "22.4 Biosynthesis and Degradation of Nucleotides"
    }
}


def _enrich_documents_with_metadata(document_splits: List, chapter_metadata_map: Dict[str, Dict]) -> List:
    """
    Enrich document splits with chapter metadata.
    
    Args:
        document_splits: List of document splits
        chapter_metadata_map: Mapping of filenames to metadata
    
    Returns:
        List of enriched document splits
    """
    logger.info("Enriching documents with chapter metadata")
    enriched_splits = []
    
    for doc in document_splits:
        file_name = doc.metadata.get('source_file')
        if file_name and file_name in chapter_metadata_map:
            doc.metadata.update(chapter_metadata_map[file_name])
            logger.debug(f"Added metadata for {file_name}")
        else:
            doc.metadata.update({
                'chapter_number': "N/A",
                'chapter_title': "N/A"
            })
            if file_name:
                logger.warning(f"No metadata mapping found for file: {file_name}")
        
        enriched_splits.append(doc)
    
    logger.info(f"Enriched {len(enriched_splits)} documents with metadata")
    return enriched_splits


def _log_processing_summary(summary: Dict[str, Any]) -> None:
    """Log document processing summary."""
    logger.info("Document Processing Summary:")
    logger.info(f"  Total chunks: {summary['total_chunks']}")
    logger.info(f"  Total content: {summary['size_stats']['total']:,} characters")
    logger.info(f"  Average chunk size: {summary['size_stats']['avg']} characters")
    logger.info(f"  Size range: {summary['size_stats']['min']}-{summary['size_stats']['max']} chars")
    
    logger.info("Files processed:")
    for filename, count in summary['files'].items():
        logger.info(f"  • {filename}: {count} chunks")
    
    logger.info("Sections found:")
    for section, count in summary['sections'].items():
        logger.info(f"  • {section}: {count} chunks")


def _prompt_user_for_clear_confirmation(vector_store_config) -> bool:
    """
    Prompt user for confirmation to clear existing database.
    
    Returns:
        bool: True if user confirms, False otherwise
    """
    db_path_obj = Path(vector_store_config.persist_directory)
    if db_path_obj.exists() and any(db_path_obj.iterdir()):
        logger.warning(f"Vector database at '{vector_store_config.persist_directory}' already exists and contains data")
        try:
            response = input("Clear existing documents and rebuild the database? (y/n): ").lower().strip()
            if response == 'y':
                logger.info("User confirmed to clear existing data")
                return True
            else:
                logger.info("User chose to keep existing data - aborting ingestion")
                return False
        except (EOFError, KeyboardInterrupt):
            logger.info("User interrupted prompt - aborting ingestion")
            return False
    return False


def ingest_data_into_vectordb(clear_existing: bool = False) -> Optional[Dict[str, Any]]:
    """
    Complete data ingestion pipeline - processes markdown files into vector database.
    
    Args:
        clear_existing (bool): Whether to clear existing collection without prompting
    
    Returns:
        dict: Summary of the ingestion process, or None if failed/aborted
    """
    logger.info("=" * 60)
    logger.info("🚀 STARTING DATA INGESTION FOR BIOCHEMISTRY DOCUMENTS")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        rag_config = config.get_config()
        embedding_config = rag_config.vector_store 
        vector_store_config = rag_config.vector_store
        document_config = rag_config.document
        
        # Step 1: Initialize Document Splitter
        logger.info("📄 STEP 1: Initializing Document Splitter")
        splitter = LehningerDocumentSplitter(
            max_chunk_size=document_config.max_chunk_size,
            chunk_overlap=document_config.chunk_overlap
        )
        logger.info(f"Configured for chunks: {document_config.max_chunk_size} chars, overlap: {document_config.chunk_overlap}")
        
        # Step 2: Process Documents
        logger.info(f"📂 STEP 2: Processing Documents from '{document_config.data_directory}'")
        
        if not Path(document_config.data_directory).exists():
            raise FileNotFoundError(f"Directory not found: {document_config.data_directory}")
        
        document_splits = splitter.process_directory(document_config.data_directory)
        
        if not document_splits:
            raise ValueError(f"No documents were processed from {document_config.data_directory}. "
                           "Ensure .md files are present and readable.")
        
        # Step 2.5: Enrich documents with metadata
        logger.info("✨ STEP 2.5: Enriching documents with chapter metadata")
        document_splits = _enrich_documents_with_metadata(document_splits, CHAPTER_METADATA_MAP)
        
        # Log processing summary
        summary = splitter.get_chunks_summary(document_splits)
        _log_processing_summary(summary)
        
        # Step 3: Initialize Embedding Function
        logger.info("🧠 STEP 3: Loading Embedding Model")
        try:
            embedding_function = get_langchain_huggingface_embeddings(
                model_name=embedding_config.embedding_model_name, 
                device=embedding_config.embedding_model_device
            )
            
            embedding_dim = embedding_function.client.get_sentence_embedding_dimension()
            device = embedding_function.client.device
            
            logger.info(f"Embedding function loaded: {embedding_config.embedding_model_name}")
            logger.info(f"  Embedding dimension: {embedding_dim}")
            logger.info(f"  Device: {device}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Step 4: Handle database clearing
        logger.info("🗃️  STEP 4: Setting up ChromaDB Vector Database")
        
        perform_clear = clear_existing
        if not clear_existing:
            perform_clear = _prompt_user_for_clear_confirmation(vector_store_config)
            if not perform_clear and Path(vector_store_config.persist_directory).exists():
                logger.info("Aborted - keeping existing documents")
                return None
        
        # Initialize vector store
        try:
            vector_store = get_langchain_chroma_vector_store(
                embedding_function=embedding_function,
                persist_directory=vector_store_config.persist_directory,
                collection_name=vector_store_config.collection_name,
                reset_db=perform_clear
            )
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
        
        # Step 5: Add documents to vector database
        logger.info("💾 STEP 5: Adding Documents to ChromaDB")
        try:
            vector_store.add_documents(document_splits)
            final_count = vector_store._collection.count()
            logger.info(f"Successfully stored {final_count} documents in ChromaDB")
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise
        
        # Step 6: Create and log final summary
        logger.info("=" * 60)
        logger.info("🎉 DATA INGESTION COMPLETE!")
        logger.info("=" * 60)
        
        config_summary = {
            "status": "success",
            "documents": {
                "total_chunks": len(document_splits),
                "files_processed": len(summary['files']),
                "sections_found": len(summary['sections']),
                "total_characters": summary['size_stats']['total']
            },
            "embeddings": {
                "model": embedding_config.embedding_model_name,
                "dimension": embedding_dim,
                "device": str(device),
                "count": final_count
            },
            "vector_store": {
                "path": vector_store_config.persist_directory,
                "collection": vector_store_config.collection_name,
                "document_count": final_count
            },
            "configuration": {
                "max_chunk_size": document_config.max_chunk_size,
                "chunk_overlap": document_config.chunk_overlap,
                "data_directory": document_config.data_directory
            }
        }
        
        logger.info("Configuration Summary:")
        logger.info(json.dumps(config_summary['configuration'], indent=2))
        logger.info(f"Vector database persisted at: {vector_store_config.persist_directory}")
        logger.info("Next: use 'rag_pipeline.py' to build your RAG chain and 'app.py' to query it")
        
        return config_summary
        
    except FileNotFoundError as e:
        logger.error(f"Directory error: {e}")
        logger.error(f"Ensure '{config.get_config().document.data_directory}' exists and contains .md files")
        return None
        
    except ValueError as e:
        logger.error(f"Data processing error: {e}")
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error during data ingestion: {e}", exc_info=True)
        return None


def main() -> bool:
    """Main function to trigger data ingestion."""
    logger.info("Preparing to ingest data for Lehninger Biochemistry textbook")
    
    rag_config = config.get_config()
    logger.info(f"Looking for markdown files in: {rag_config.document.data_directory}")
    
    result = ingest_data_into_vectordb(clear_existing=False) 
    
    if result:
        logger.info("🎯 Data ingestion successful!")
        return True
    else:
        logger.error("❌ Data ingestion failed or was aborted")
        return False


if __name__ == "__main__":
    success = main()
    
    if not success:
        logger.info("💡 Troubleshooting tips:")
        logger.info(f"  • Ensure '{config.get_config().document.data_directory}' directory exists with .md files")
        logger.info("  • Check installed packages: langchain, chromadb, sentence-transformers, pyyaml")
        sys.exit(1)
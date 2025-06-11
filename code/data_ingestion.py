#!/usr/bin/env python3

from rag_components import (
    LehningerDocumentSplitter,
    get_langchain_huggingface_embeddings,
    get_langchain_chroma_vector_store
)
import os
from pathlib import Path
import sys
import json 
import config 

def ingest_data_into_vectordb(
    clear_existing: bool = False
):
    """
    Complete data ingestion pipeline - processes markdown files directly into vector database.
    Configuration is loaded from the central ConfigManager.
    
    Args:
        clear_existing (bool): Whether to clear existing collection. If False, will prompt user.
    
    Returns:
        dict: Summary of the ingestion process, or None if aborted/failed.
    """
    
    rag_config = config.get_config()
    embedding_config = rag_config.vector_store 
    vector_store_config = rag_config.vector_store
    document_config = rag_config.document
    llm_config = config.get_llm_config() 

    print("=" * 60)
    print("🚀 STARTING DATA INGESTION FOR BIOCHEMISTRY DOCUMENTS")
    print("=" * 60)
    
    try:
        # Step 1: Initialize Document Splitter
        print("\n📄 STEP 1: Initializing Document Splitter...")
        splitter = LehningerDocumentSplitter(
            max_chunk_size=document_config.max_chunk_size,
            chunk_overlap=document_config.chunk_overlap
        )
        print(f"✅ Configured for chunks: {document_config.max_chunk_size} chars, overlap: {document_config.chunk_overlap}")
        
        # --- Define Chapter Configuration Map ---
        chapter_metadata_map = {
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

        # Step 2: Process Documents
        print(f"\n📂 STEP 2: Processing Documents from '{document_config.data_directory}'...")
        
        # Check if directory exists
        if not Path(document_config.data_directory).exists():
            raise FileNotFoundError(f"Directory not found: {document_config.data_directory}")
        
        # --- Pass chapter_config to process_directory ---
        document_splits = splitter.process_directory(
            document_config.data_directory,
        )
        
        if not document_splits:
            raise ValueError(f"No documents were processed successfully from {document_config.data_directory}. "
                             "Please ensure .md files are present and readable.")
        # --- Enrich Document Splits with Chapter Metadata ---
        print("\n✨ STEP 2.5: Enriching documents with chapter metadata...")
        enriched_document_splits = []
        for doc in document_splits:
            file_name = doc.metadata.get('source_file')
            if file_name and file_name in chapter_metadata_map:
                doc.metadata['chapter_number'] = chapter_metadata_map[file_name]['chapter_number']
                doc.metadata['chapter_title'] = chapter_metadata_map[file_name]['chapter_title']
            else:
                doc.metadata['chapter_number'] = "N/A"
                doc.metadata['chapter_title'] = "N/A"
            enriched_document_splits.append(doc)
        
        # Overwrite document_splits with the enriched list
        document_splits = enriched_document_splits
        print(f"✅ Documents enriched with chapter metadata. First 5 docs now have metadata like: {document_splits[0].metadata}")
        
        # Get processing summary
        summary = splitter.get_chunks_summary(document_splits)
        
        print(f"\n📊 Document Processing Summary:")
        print(f"   Total chunks: {summary['total_chunks']}")
        print(f"   Total content: {summary['size_stats']['total']:,} characters")
        print(f"   Average chunk size: {summary['size_stats']['avg']} characters")
        print(f"   Size range: {summary['size_stats']['min']}-{summary['size_stats']['max']} chars")
        
        print(f"\n📑 Files processed:")
        for filename, count in summary['files'].items():
            print(f"   • {filename}: {count} chunks")
        
        print(f"\n🏷️ Sections found:")
        for section, count in summary['sections'].items():
            print(f"   • {section}: {count} chunks")
        
        # Step 3: Initialize LangChain Embedding Function
        print(f"\n🧠 STEP 3: Loading LangChain Embedding Model...")
        embedding_function = get_langchain_huggingface_embeddings(
            model_name=embedding_config.embedding_model_name, 
            device=embedding_config.embedding_model_device
        )
        print(f"✅ LangChain Embedding function loaded: {embedding_config.embedding_model_name}")
        print(f"   Embedding dimension (approx): {embedding_function.client.get_sentence_embedding_dimension()}")
        print(f"   Device: {embedding_function.client.device}")
        
        # Step 4: Initialize LangChain Vector Store
        print(f"\n🗃️  STEP 4: Setting up LangChain ChromaDB Vector Database...")
        
        # Handle clearing existing data with user prompt if clear_existing is False
        perform_clear = clear_existing
        if not clear_existing:
            db_path_obj = Path(vector_store_config.persist_directory)
            if db_path_obj.exists() and any(db_path_obj.iterdir()):
                print(f"⚠️  Vector database at '{vector_store_config.persist_directory}' already exists and contains data.")
                response = input("Clear existing documents and rebuild the database? (y/n): ").lower().strip()
                if response == 'y':
                    perform_clear = True
                    print("🧹 User confirmed to clear existing data.")
                else:
                    print("❌ Aborted data ingestion - keeping existing documents.")
                    return None
        
        vector_store = get_langchain_chroma_vector_store(
            embedding_function=embedding_function,
            persist_directory=vector_store_config.persist_directory,
            collection_name=vector_store_config.collection_name,
            reset_db=perform_clear
        )
        
        # Step 5: Add Documents to Vector Database
        print(f"\n💾 STEP 5: Adding Documents to LangChain ChromaDB...")
        vector_store.add_documents(enriched_document_splits)
        
        final_count = vector_store._collection.count()
        print(f"✅ Successfully stored {final_count} documents in LangChain ChromaDB.")
        
        # Step 6: Final Summary
        print("\n" + "=" * 60)
        print("🎉 DATA INGESTION COMPLETE!")
        print("=" * 60)
        
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
                "dimension_approx": embedding_function.client.get_sentence_embedding_dimension(),
                "device": embedding_function.client.device,
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
        
        print(f"📊 Configuration Summary:")
        # Using json.dumps for a cleaner, more readable config output
        print(json.dumps(config_summary['configuration'], indent=2))
        print(f"\n✅ Your vector database is ready!")
        print(f"   • Vector database persisted at: {vector_store_config.persist_directory}")
        print(f"   • Next, use 'rag_pipeline.py' to build your full RAG chain and 'main.py' to query it.")
        
        return config_summary
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Make sure your data directory '{document_config.data_directory}' exists and contains .md files.")
        return None
        
    except ValueError as e:
        print(f"❌ Data processing error: {e}")
        return None

    except Exception as e:
        print(f"❌ Unexpected error during data ingestion: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to trigger data ingestion using global config."""
    print("Preparing to ingest data for Lehninger Biochemistry textbook...")
    
    rag_config = config.get_config()
    
    print(f"Looking for markdown files in: {rag_config.document.data_directory}")
    
    result = ingest_data_into_vectordb(clear_existing=False) 
    
    if result:
        print(f"\n🎯 Data ingestion successful!")
        return True
    else:
        print(f"\n❌ Data ingestion failed or was aborted. Check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n💡 Troubleshooting tips:")
        print(f"   • Make sure '{config.get_config().document.data_directory}' directory exists and contains .md files")
        print("   • Ensure you have installed: langchain, chromadb, sentence-transformers, pyyaml")
        sys.exit(1)
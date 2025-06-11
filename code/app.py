import streamlit as st
import sys
import os
from pathlib import Path
from rag_pipeline import BiochemistryRAGPipeline
from data_ingestion import ingest_data_into_vectordb
import config

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Biochemistry RAG System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize RAG Pipeline in Session State ---
@st.cache_resource 
def get_rag_pipeline():
    """Initializes and returns the BiochemistryRAGPipeline."""
    try:
        pipeline = BiochemistryRAGPipeline()
        return pipeline
    except ValueError as e:
        st.error(f"Error initializing RAG pipeline: {e}")
        st.info("Please ensure your vector database is populated and API key is set correctly.")
        st.session_state.pipeline_ready = False
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during RAG pipeline initialization: {e}")
        st.session_state.pipeline_ready = False
        return None

# --- Data Ingestion Function for Streamlit ---
def run_data_ingestion(clear_existing: bool = False):
    """
    Triggers the data ingestion process and updates Streamlit status.
    """
    st.info("Starting data ingestion...")
    with st.spinner("Processing documents and building vector database... This may take a few minutes."):
        ingestion_success = ingest_data_into_vectordb(clear_existing=clear_existing)
        if ingestion_success:
            st.success("🎉 Data Ingestion Complete! Vector database is ready.")
            st.session_state.rag_pipeline = get_rag_pipeline()
            if st.session_state.rag_pipeline:
                st.session_state.pipeline_ready = True
        else:
            st.error("❌ Data Ingestion Failed or Aborted. Check console for details.")
            st.session_state.pipeline_ready = False

# --- Main Streamlit App Layout ---

st.title("🧬 Biochemistry RAG System")
st.markdown("Ask questions about Lehninger Principles of Biochemistry, Chapter 22-Biosynthesis of Amino Acids, Nucleotides, and Related Molecules!")

# --- Sidebar for Configuration and Ingestion ---
with st.sidebar:
    st.header("Configuration & Data Management")
    
    rag_config = config.get_config()
    document_config = rag_config.document
    vector_store_config = rag_config.vector_store

    st.subheader("Data Directory")
    st.code(f"Path: {document_config.data_directory}")
    st.code(f"Chunk Size: {document_config.max_chunk_size}, Overlap: {document_config.chunk_overlap}")

    st.subheader("Vector Database")
    st.code(f"Path: {vector_store_config.persist_directory}")
    st.code(f"Collection: {vector_store_config.collection_name}")

# --- Initialize or Retrieve RAG Pipeline ---
if 'rag_pipeline' not in st.session_state or st.session_state.rag_pipeline is None:
    with st.spinner("Loading RAG pipeline..."):
        st.session_state.rag_pipeline = get_rag_pipeline()
        if st.session_state.rag_pipeline:
            st.session_state.pipeline_ready = True
        else:
            st.session_state.pipeline_ready = False 

# --- Main Query Interface ---
if st.session_state.get('pipeline_ready', False):
    st.header("Ask a Question")
    user_query = st.text_area("Enter your question about Biosynthesis of Amino Acids, Nucleotides, and Related Molecules:", height=100, key="query_input")

    if st.button("Get Answer", type="primary"):
        if user_query:
            with st.spinner("Thinking..."):
                try:
                    pipeline_instance = st.session_state.rag_pipeline
                    results = pipeline_instance.query(user_query)
                    
                    st.subheader("🤖 Answer:")
                    st.write(results.get('result', "I apologize, I couldn't find a relevant answer in the provided context."))
                    
                    source_documents = results.get('source_documents', [])
                    if source_documents:
                        st.subheader("📄 Sources:")
                        for i, doc in enumerate(source_documents):
                            with st.expander(f"Source Document {i+1} (File: {doc.metadata.get('source_file', 'N/A')}, Section: {doc.metadata.get('Section', 'N/A')}):"):
                                st.code(doc.page_content)
                                st.caption(f"Source File: {doc.metadata.get('source_file', 'N/A')}, Section: {doc.metadata.get('Section', 'N/A')}")
                    else:
                        st.info("No specific sources were retrieved for this query.")
                
                except Exception as e:
                    st.error(f"An error occurred while processing your query: {e}")
                    st.warning("Please check your LLM API key and ensure the vector database is properly initialized.")
        else:
            st.warning("Please enter a question to get an answer.")
else:
    st.warning("RAG pipeline is not ready. Please check configuration/API key and run data ingestion if needed.")

# --- Footer
st.markdown("---")
st.caption("Built with LangChain, Streamlit, ChromaDB, and Together AI")
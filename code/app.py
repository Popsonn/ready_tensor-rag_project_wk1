import logging
import streamlit as st
from typing import Optional, Dict, Any

from rag_pipeline import BiochemistryRAGPipeline
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
APP_TITLE = "🧬 Biochemistry RAG System"
APP_DESCRIPTION = "Ask questions about Lehninger Principles of Biochemistry, Chapter 22-Biosynthesis of Amino Acids, Nucleotides, and Related Molecules!"


class BiochemistryRAGApp:
    """
    Streamlit application for the Biochemistry RAG System.
    
    Provides a user interface for querying biochemistry knowledge
    using a RAG (Retrieval-Augmented Generation) pipeline.
    """
    
    def __init__(self):
        """Initialize the Streamlit app with configuration."""
        self._configure_page()
        self._initialize_session_state()
        
    def _configure_page(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Biochemistry RAG System",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def _initialize_session_state(self) -> None:
        """Initialize session state variables."""
        if 'pipeline_ready' not in st.session_state:
            st.session_state.pipeline_ready = False
        if 'rag_pipeline' not in st.session_state:
            st.session_state.rag_pipeline = None
            
    @st.cache_resource 
    def get_rag_pipeline(_self) -> Optional[BiochemistryRAGPipeline]:
        """
        Initialize and return the BiochemistryRAGPipeline.
        
        Returns:
            Optional[BiochemistryRAGPipeline]: Pipeline instance or None if initialization fails
        """
        try:
            logger.info("Initializing RAG pipeline...")
            pipeline = BiochemistryRAGPipeline()
            logger.info("RAG pipeline initialized successfully")
            return pipeline
            
        except ValueError as e:
            logger.error(f"Configuration error during pipeline initialization: {e}")
            st.error(f"Configuration Error: {e}")
            st.info("💡 Please ensure your vector database is populated and API key is set correctly.")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error during pipeline initialization: {e}")
            st.error(f"Initialization Error: {e}")
            st.info("💡 Please check your configuration and try again.")
            return None

    def render_sidebar(self) -> None:
        """Render the sidebar with configuration information."""
        with st.sidebar:
            st.header("⚙️ System Configuration")
            
            try:
                rag_config = config.get_config()
                document_config = rag_config.document
                vector_store_config = rag_config.vector_store
                
                # Data directory info
                st.subheader("📁 Data Directory")
                st.code(f"Path: {document_config.data_directory}")
                st.code(f"Chunk Size: {document_config.max_chunk_size}")
                st.code(f"Overlap: {document_config.chunk_overlap}")

                # Vector database info
                st.subheader("🗃️ Vector Database")
                st.code(f"Path: {vector_store_config.persist_directory}")
                st.code(f"Collection: {vector_store_config.collection_name}")
                
                # Pipeline status
                st.subheader("🚦 Pipeline Status")
                if st.session_state.get('pipeline_ready', False):
                    st.success("✅ Ready")
                else:
                    st.error("❌ Not Ready")
                    
            except Exception as e:
                logger.error(f"Error loading configuration for sidebar: {e}")
                st.error("❌ Configuration Error")

    def render_main_interface(self) -> None:
        """Render the main query interface."""
        st.title(APP_TITLE)
        st.markdown(APP_DESCRIPTION)
        
        if not st.session_state.get('pipeline_ready', False):
            self._render_pipeline_not_ready()
            return
        
        self._render_query_interface()

    def _render_pipeline_not_ready(self) -> None:
        """Render interface when pipeline is not ready."""
        st.warning("⚠️ RAG pipeline is not ready.")
        st.info("💡 Please ensure data_ingestion.py has been run and your API key is set correctly.")
        
        # Try to initialize pipeline
        if st.button("🔄 Try Initialize Pipeline", type="secondary"):
            with st.spinner("Initializing pipeline..."):
                st.session_state.rag_pipeline = self.get_rag_pipeline()
                st.session_state.pipeline_ready = bool(st.session_state.rag_pipeline)
                st.rerun()

    def _render_query_interface(self) -> None:
        """Render the main query interface when pipeline is ready."""
        st.header("💬 Ask a Question")
        
        user_query = st.text_area(
            "Enter your question about Biosynthesis of Amino Acids, Nucleotides, and Related Molecules:",
            height=100,
            key="query_input",
            placeholder="e.g., What causes gout in humans?"
        )

        if st.button("🔍 Get Answer", type="primary", use_container_width=True):
            if not user_query.strip():
                st.warning("⚠️ Please enter a question to get an answer.")
                return
                
            self._process_query(user_query)

    def _process_query(self, user_query: str) -> None:
        """
        Process user query and display results.
        
        Args:
            user_query (str): The user's question
        """
        logger.info(f"Processing user query: {user_query[:100]}...")
        
        with st.spinner("🤔 Thinking..."):
            try:
                pipeline_instance = st.session_state.rag_pipeline
                results = pipeline_instance.query(user_query)
                
                self._display_results(results)
                logger.info("Query processed successfully")
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                st.error(f"❌ Query Processing Error: {e}")
                st.warning("💡 Please check your API key and ensure the system is properly configured.")

    def _display_results(self, results: Dict[str, Any]) -> None:
        """
        Display query results in the UI.
        
        Args:
            results (Dict[str, Any]): Results from the RAG pipeline
        """
        # Display answer
        st.subheader("🤖 Answer:")
        answer = results.get('result', "I apologize, I couldn't find a relevant answer in the provided context.")
        st.write(answer)
        
        # Display sources
        source_documents = results.get('source_documents', [])
        if source_documents:
            st.subheader("📄 Sources:")
            self._display_sources(source_documents)
        else:
            st.info("ℹ️ No specific sources were retrieved for this query.")

    def _display_sources(self, source_documents: list) -> None:
        """
        Display source documents in expandable sections.
        
        Args:
            source_documents (list): List of source documents
        """
        for i, doc in enumerate(source_documents):
            metadata = doc.metadata
            source_file = metadata.get('source_file', 'N/A')
            section = metadata.get('Section', 'N/A')
            chapter_title = metadata.get('chapter_title', 'N/A')
            chapter_number = metadata.get('chapter_number', 'N/A')
            
            # Create expandable section for each source
            with st.expander(f"📄 Source {i+1}: {source_file} - Chapter {chapter_number}"):
                st.markdown(f"**Section:** {section}")
                st.markdown(f"**Chapter:** {chapter_number} - {chapter_title}")
                st.markdown("**Content:**")
                st.text_area(
                    "Source content",
                    value=doc.page_content,
                    height=200,
                    key=f"source_{i}",
                    label_visibility="collapsed"
                )

    def _render_footer(self) -> None:
        """Render the application footer."""
        st.markdown("---")
        st.caption("🛠️ Built with LangChain, Streamlit, ChromaDB, and Together AI")

    def _load_pipeline(self) -> None:
        """Load the RAG pipeline if not already loaded."""
        if 'rag_pipeline' not in st.session_state or st.session_state.rag_pipeline is None:
            with st.spinner("🔄 Loading RAG pipeline..."):
                st.session_state.rag_pipeline = self.get_rag_pipeline()
                st.session_state.pipeline_ready = bool(st.session_state.rag_pipeline)

    def run(self) -> None:
        """Run the complete Streamlit application."""
        try:
            # Load pipeline
            self._load_pipeline()
            
            # Render UI components
            self.render_sidebar()
            self.render_main_interface()
            self._render_footer()
            
        except Exception as e:
            logger.error(f"Error running application: {e}")
            st.error(f"❌ Application Error: {e}")
            st.info("💡 Please refresh the page and try again.")


def main():
    """Main entry point for the Streamlit application."""
    try:
        app = BiochemistryRAGApp()
        app.run()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        st.error("❌ Failed to start the application. Please check the logs.")


if __name__ == "__main__":
    main()
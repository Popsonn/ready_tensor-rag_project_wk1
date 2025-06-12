import logging
import os
import re
import traceback 
from typing import Optional, Dict, Any, List

from langchain_community.llms import Together
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser

from config import get_config, get_vector_store_config
from rag_components import (
    get_langchain_huggingface_embeddings,
    get_langchain_chroma_vector_store
)
import config 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BiochemistryRAGPipeline:
    """
    A RAG (Retrieval-Augmented Generation) pipeline specifically designed for biochemistry queries.
    
    This pipeline combines LangChain components with ChromaDB for document retrieval
    and Together AI for language generation to answer biochemistry-related questions.
    """
    
    def __init__(self):
        """
        Initializes the RAG pipeline with all necessary components.
        
        Raises:
            ValueError: If configuration is invalid or vector database is empty
            Exception: If initialization fails
        """
        logger.info("=" * 60)
        logger.info("⚙️ INITIALIZING BIOCHEMISTRY RAG PIPELINE")
        logger.info("=" * 60)

        try:
            self._load_configurations()
            self._setup_api_credentials()
            self._initialize_embeddings()
            self._initialize_vector_store()
            self._setup_retriever()
            self._initialize_llm()
            self._setup_prompt_template()
            
            logger.info("✨ RAG Pipeline initialized successfully!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            raise

    def _load_configurations(self) -> None:
        """Load all required configurations from config files."""
        logger.info("📋 Loading configurations...")
        
        self.rag_config = config.get_config()
        self.llm_config = config.get_llm_config()
        self.embedding_config = self.rag_config.vector_store
        self.vector_store_config = self.rag_config.vector_store
        self.retrieval_config = self.rag_config.retrieval
        
        logger.info("✅ Configurations loaded successfully")

    def _setup_api_credentials(self) -> None:
        """Setup API credentials for external services."""
        logger.info("🔑 Setting up API credentials...")
        
        if not self.llm_config.api_key:
            raise ValueError(
                "TOGETHER_API_KEY is not set. Please set it as an environment variable or in config.yaml."
            )
        
        os.environ["TOGETHER_API_KEY"] = self.llm_config.api_key
        logger.info("✅ API credentials configured")

    def _initialize_embeddings(self) -> None:
        """Initialize the embedding model."""
        logger.info(f"🧠 Loading embeddings: {self.embedding_config.embedding_model_name}")
        
        self.embeddings = get_langchain_huggingface_embeddings(
            model_name=self.embedding_config.embedding_model_name,
            device=self.embedding_config.embedding_model_device
        )
        
        logger.info("✅ Embeddings loaded successfully")

    def _initialize_vector_store(self) -> None:
        """Initialize and validate the vector store."""
        logger.info(
            f"🗃️ Connecting to ChromaDB at: "
            f"{self.vector_store_config.persist_directory}/{self.vector_store_config.collection_name}"
        )
        
        self.vector_store = get_langchain_chroma_vector_store(
            embedding_function=self.embeddings,
            persist_directory=self.vector_store_config.persist_directory,
            collection_name=self.vector_store_config.collection_name
        )
        
        # Validate database is populated
        document_count = self.vector_store._collection.count()
        if document_count == 0:
            raise ValueError(
                f"Vector database collection '{self.vector_store_config.collection_name}' is empty. "
                f"Please run 'python data_ingestion.py' first to populate it."
            )
        
        logger.info(f"✅ Vector Store connected. Contains {document_count} documents")

    def _setup_retriever(self) -> None:
        """Setup the document retriever."""
        logger.info("🔎 Creating retriever...")
        
        self.base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.retrieval_config.n_results}
        )
        
        logger.info("✅ Base retriever ready")

    def _initialize_llm(self) -> None:
        """Initialize the language model."""
        logger.info(f"🗣️ Initializing LLM: {self.llm_config.model_name} (Provider: {self.llm_config.provider})")
        
        self.llm = Together(
            model=self.llm_config.model_name,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
        )
        
        logger.info("✅ LLM initialized successfully")

    def _setup_prompt_template(self) -> None:
        """Setup the RAG prompt template from configuration."""
        logger.info("📝 Setting up RAG prompt template...")
        
        prompt_name = "lehninger_rag_prompt_cfg"
        prompt_config_data = config.get_config_manager().get_prompt_template(prompt_name)
        
        if not prompt_config_data:
            raise ValueError(f"Prompt template '{prompt_name}' not found in prompt_config.yaml!")
        
        logger.debug(f"Loaded prompt config: {type(prompt_config_data)}")
        
        # Get reasoning strategy
        cot_strategy_text = self._get_reasoning_strategy(prompt_config_data)
        
        # Build system message
        system_message = self._build_system_message(prompt_config_data, cot_strategy_text)
        
        # Create final prompt template
        combined_template = f"{system_message}\n\nContext:\n{{context}}\n\nQuestion: {{question}}"
        self.rag_prompt = PromptTemplate(
            template=combined_template,
            input_variables=["context", "question"]
        )
        
        logger.debug("Final prompt template:")
        logger.debug("-" * 40)
        logger.debug(self.rag_prompt.template)
        logger.debug("-" * 40)
        
        logger.info("✅ Prompt template configured")

    def _get_reasoning_strategy(self, prompt_config_data: Dict) -> Optional[str]:
        """Extract reasoning strategy from prompt configuration."""
        reasoning_strategy_key = prompt_config_data.get("reasoning_strategy")
        if not reasoning_strategy_key:
            return None
        
        loaded_strategy = config.get_config_manager().get_reasoning_strategy(reasoning_strategy_key)
        
        if loaded_strategy and isinstance(loaded_strategy, dict):
            return loaded_strategy.get(reasoning_strategy_key)
        elif isinstance(loaded_strategy, str):
            return loaded_strategy
        
        logger.warning(
            f"Reasoning strategy '{reasoning_strategy_key}' found in prompt config "
            f"but its content is missing or malformed"
        )
        return None

    def _build_system_message(self, prompt_config_data: Dict, cot_strategy_text: Optional[str]) -> str:
        """Build the system message from prompt configuration components."""
        system_message_parts = []

        # Add core components
        if prompt_config_data.get("role"):
            system_message_parts.append(prompt_config_data["role"].strip())
        
        if prompt_config_data.get("instruction"):
            system_message_parts.append(prompt_config_data["instruction"].strip())
        
        if prompt_config_data.get("goal"):
            system_message_parts.append(f"\nGoal: {prompt_config_data['goal'].strip()}")

        # Add output constraints
        output_constraints = prompt_config_data.get("output_constraints")
        if output_constraints and isinstance(output_constraints, list):
            system_message_parts.append("\nOutput Constraints:")
            system_message_parts.extend([f"  - {c.strip()}" for c in output_constraints])

        # Add style/tone
        style_or_tone = prompt_config_data.get("style_or_tone")
        if style_or_tone and isinstance(style_or_tone, list):
            system_message_parts.append("\nStyle/Tone:")
            system_message_parts.extend([f"  - {s.strip()}" for s in style_or_tone])

        # Add reasoning strategy
        if cot_strategy_text:
            system_message_parts.append(f"\nReasoning Strategy:\n{cot_strategy_text.strip()}")

        return "\n\n".join(filter(None, system_message_parts))

    def query(self, question: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the RAG pipeline with a user question.
        
        Args:
            question (str): The user's question
            filters (Optional[Dict[str, Any]]): Metadata filters for document retrieval
                Example: {"chapter_number": "22"}
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - query: The original question
                - result: The LLM's answer
                - source_documents: List of source documents used
        """
        logger.info(f"🔬 Processing query: '{question}'")
        if filters:
            logger.info(f"Applying filters: {filters}")

        try:
            # Setup retriever with filters
            current_retriever = self._create_filtered_retriever(filters)
            
            # Build RAG chain
            rag_chain = self._build_rag_chain(current_retriever)
            
            # Retrieve and log documents
            retrieved_docs = current_retriever.get_relevant_documents(question)
            self._log_retrieved_documents(retrieved_docs)
            
            # Get response
            response = rag_chain.invoke({"query": question})
            
            # Post-process response
            self._clean_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error during RAG pipeline query: {e}")
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            
            return {
                "query": question, 
                "result": f"An error occurred: {e}", 
                "source_documents": []
            }

    def _create_filtered_retriever(self, filters: Optional[Dict[str, Any]]):
        """Create a retriever with optional filters applied."""
        search_kwargs = {"k": self.base_retriever.search_kwargs['k']}
        if filters:
            search_kwargs["filter"] = filters
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

    def _build_rag_chain(self, retriever):
        """Build the RAG chain with the given retriever."""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            verbose=False,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.rag_prompt}
        )

    def _log_retrieved_documents(self, retrieved_docs: List) -> None:
        """Log information about retrieved documents."""
        logger.debug(f"Retrieved {len(retrieved_docs)} documents")
        
        if not retrieved_docs:
            logger.debug("No documents retrieved with current filters and query")
            return
        
        for i, doc in enumerate(retrieved_docs):
            source_info = self._extract_document_metadata(doc)
            logger.debug(
                f"Doc {i+1}: Source: {source_info['source']}, "
                f"Chapter: {source_info['chapter']}, "
                f"Section: {source_info['section']}, "
                f"Length: {len(doc.page_content)} chars"
            )

    def _extract_document_metadata(self, doc) -> Dict[str, str]:
        """Extract and format document metadata."""
        metadata = doc.metadata
        return {
            'source': metadata.get('source_file', 'N/A'),
            'section': metadata.get('Section', 'N/A'),
            'chapter': f"{metadata.get('chapter_number', 'N/A')} - {metadata.get('chapter_title', 'N/A')}",
        }

    def _clean_response(self, response: Dict[str, Any]) -> None:
        """Clean up the LLM response by removing unwanted patterns."""
        if "result" not in response or not isinstance(response["result"], str):
            return
        
        llm_raw_answer = response["result"]
        cleaned_answer = self._apply_cleaning_patterns(llm_raw_answer)
        
        if cleaned_answer != llm_raw_answer:
            logger.debug("Post-processing cleaned up response")
            response["result"] = cleaned_answer

    def _apply_cleaning_patterns(self, text: str) -> str:
        """Apply regex patterns to clean unwanted text from LLM responses."""
        unwanted_patterns = [
            r'Right Answer:\s*.*?(?=Answer:|$)',
            r'Explanation of the Solution:\s*.*?(?=Related Questions:|$)',
            r'Related Questions:\s*.*$',
            r'Sources:\s*.*$',
            r'^\s*Answer:\s*Answer:\s*',
            r'^\s*Answer:\s*',
            r'^\s*The answer is:\s*',
        ]
        
        cleaned_text = text
        for pattern in unwanted_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
        
        return cleaned_text.strip()


def _run_pipeline_test():
    """Run a comprehensive test of the RAG pipeline."""
    logger.info("--- Running RAG pipeline test ---")
    
    try:
        pipeline = BiochemistryRAGPipeline()
        
        # Test questions
        test_questions = [
            "What are the main functions of ATP in a cell?",
            "Explain the concept of enzyme specificity.",
            "What is glycolysis and where does it occur?",
            "What is the role of the citric acid cycle?",
            "Tell me about quantum physics."  # Out-of-scope question
        ]

        # Test with filters
        logger.info("\n--- Testing with chapter filter (Chapter 22) ---")
        filtered_question = "What is the primary role of transaminases in amino acid metabolism?"
        filtered_results = pipeline.query(filtered_question, filters={"chapter_number": "22"})
        
        _display_results("FILTERED QUERY", filtered_question, filtered_results)

        # Test regular queries
        for i, question in enumerate(test_questions):
            logger.info(f"\n{'-'*80}")
            results = pipeline.query(question)
            _display_results(f"QUESTION {i+1}", question, results)
            logger.info(f"{'-'*80}\n")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please ensure your vector database is populated by running 'python data_ingestion.py' first.")
        logger.error("Also check your API key is set correctly.")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        logger.error(traceback.format_exc())


def _display_results(title: str, question: str, results: Dict[str, Any]) -> None:
    """Display query results in a formatted way."""
    logger.info(f"{title}: {question}")
    logger.info(f"\n🤖 ANSWER:\n{results.get('result', 'No answer found.')}")
    
    source_docs = results.get('source_documents', [])
    if source_docs:
        logger.info("\n📄 SOURCES:")
        for doc in source_docs:
            metadata = doc.metadata
            source_info = (
                f"   - Source: {metadata.get('source_file', 'N/A')}, "
                f"Chapter: {metadata.get('chapter_number', 'N/A')} - {metadata.get('chapter_title', 'N/A')}, "
                f"Section: {metadata.get('Section', 'N/A')}"
            )
            logger.info(source_info)
    else:
        logger.info("\n📄 No specific sources retrieved")


if __name__ == "__main__":
    _run_pipeline_test()
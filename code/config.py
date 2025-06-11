from pathlib import Path
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import yaml
from dotenv import load_dotenv

load_dotenv()

# --- Dataclasses for Configuration ---
@dataclass
class LLMConfig:
    """Configuration for the Language Model."""
    #model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    provider: str = "together" 
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("TOGETHER_API_KEY"))
    temperature: float = 0.0
    max_tokens: int = 512
    streaming: bool = False
    base_url: Optional[str] = None 

@dataclass
class DocumentConfig:
    """Configuration for document processing."""
    data_directory: str = "data" 
    max_chunk_size: int = 800
    chunk_overlap: int = 50

@dataclass
class VectorStoreConfig:
    """Configuration for the vector database."""
    persist_directory: str = "./biochem_chroma_db"
    collection_name: str = "lehninger_biochem"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_device: Optional[str] = None

@dataclass
class RetrievalConfig:
    """Configuration for the document retriever."""
    n_results: int = 4 

@dataclass
class RAGConfig:
    """Overall RAG pipeline configuration."""
    llm: LLMConfig
    document: DocumentConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    reasoning_strategies: Dict[str, Any] = field(default_factory=dict)
    prompt_templates: Dict[str, Any] = field(default_factory=dict)


# --- Configuration Manager Class ---
class ConfigManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config = self._load_default_config() 
        self.reasoning_strategies = {} 
        self.prompt_templates = {} 

        # Check if config directory exists before attempting to load YAMLs
        if self.config_dir.exists():
            self._load_yaml_configs()
        else:
            print(f"⚠️  Config directory '{self.config_dir}' not found. Using defaults.")

    def _load_default_config(self) -> RAGConfig:
        """Load default configuration settings."""
        llm_config = LLMConfig()
        doc_config = DocumentConfig()
        vector_store_config = VectorStoreConfig()
        retrieval_config = RetrievalConfig() 
        return RAGConfig(
            llm=llm_config,
            document=doc_config,
            vector_store=vector_store_config,
            retrieval=retrieval_config,
            reasoning_strategies={}, 
            prompt_templates={} 
        )

    def _load_yaml_configs(self):
        """Load YAML configuration files."""
        try:
            # First, load the entire config.yaml content
            config_data = {}
            config_file = self.config_dir / "config.yaml"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
                print(f"✅ Loaded main config from {config_file}")

            # Example: Update LLM config if 'llm_settings' exists in config.yaml
            if 'llm_settings' in config_data:
                self.config.llm = LLMConfig(**config_data['llm_settings']) 

            if 'document' in config_data:
                self.config.document = DocumentConfig(**config_data['document'])

            if 'vector_store' in config_data:
                self.config.vector_store = VectorStoreConfig(**config_data['vector_store'])

            # Extract reasoning strategies specifically
            self.reasoning_strategies = config_data.get('reasoning_strategies', {})

            # Load prompt templates from prompt_config.yaml
            prompt_file = self.config_dir / "prompt_config.yaml"
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    self.prompt_templates = yaml.safe_load(f) or {}
                print(f"✅ Loaded prompt templates from {prompt_file}")

        except Exception as e:
            # Handle specific YAML parsing errors or other file issues
            print(f"⚠️  Error loading YAML configs: {e}")

    # --- Public Getter Methods ---
    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return self.config.llm

    def get_document_config(self) -> DocumentConfig:
        """Get document processing configuration."""
        return self.config.document

    def get_vector_store_config(self) -> VectorStoreConfig:
        """Get vector store configuration."""
        return self.config.vector_store

    def get_reasoning_strategy(self, strategy_name: str) -> str:
        """
        Get a specific reasoning strategy's content.
        Returns the string content of the strategy, or an empty string if not found.
        """
        strategy_content = self.reasoning_strategies.get(strategy_name, "")
        if not isinstance(strategy_content, str):
            print(f"⚠️  Warning: Reasoning strategy '{strategy_name}' content is not a string. Check config.yaml.")
            return ""
        return strategy_content

    def get_prompt_template(self, template_name: str) -> dict: 
        """
        Retrieves a prompt template by name.
        Expected to return a dictionary containing prompt components.
        Returns an empty dictionary if the template name is not found.
        """
        template_content = self.prompt_templates.get(template_name, {}) 
        
        if not isinstance(template_content, dict):
            print(f"⚠️  Warning: Prompt template '{template_name}' content is not a dictionary. Check prompt_config.yaml.")
            return {} 
            
        return template_content 

# --- Global Config Instance and Accessors ---
_config_manager: Optional[ConfigManager] = None

def get_config_manager(config_dir: str = "config") -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        project_root = Path(__file__).parent.parent
        correct_config_path = project_root / "config"
        
        print(f"DEBUG: Calculated config_dir path: {correct_config_path}")
        print(f"DEBUG: Does config_dir exist? {correct_config_path.exists()}")

        _config_manager = ConfigManager(config_dir=str(correct_config_path))
    return _config_manager

def get_config() -> RAGConfig:
    """Get the overall RAG configuration."""
    return get_config_manager().config

def get_llm_config() -> LLMConfig:
    """Convenience function to get LLM configuration directly."""
    return get_config().llm

def get_document_config() -> DocumentConfig:
    """Convenience function to get document processing configuration directly."""
    return get_config().document

def get_vector_store_config() -> VectorStoreConfig:
    """Convenience function to get vector store configuration directly."""
    return get_config().vector_store
def get_retrieval_config() -> RetrievalConfig:
    return get_config().retrieval

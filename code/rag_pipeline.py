from langchain_community.llms import Together
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import get_config, get_vector_store_config
import traceback 
import re
from langchain.prompts import PromptTemplate
from typing import Optional, Dict, Any # Import for typing filters
from rag_components import (
    get_langchain_huggingface_embeddings,
    get_langchain_chroma_vector_store
)
import config 
import os

class BiochemistryRAGPipeline:
    def __init__(self):
        """
        Initializes the LangChain-based RAG pipeline.
        Pulls configuration from the central ConfigManager and sets up LLM, Embeddings, Vector Store, and Retriever.
        """
        print("=" * 60)
        print("⚙️ INITIALIZING BIOCHEMISTRY RAG PIPELINE")
        print("=" * 60)

        # --- NEW: Get configurations from ConfigManager ---
        rag_config = config.get_config()
        llm_config = config.get_llm_config() 
        embedding_config = rag_config.vector_store
        vector_store_config = rag_config.vector_store
        retrieval_config = rag_config.retrieval

        # 1. Load API Key
        if not llm_config.api_key:
            raise ValueError("TOGETHER_API_KEY is not set. Please set it as an environment variable or in config.yaml.")
        os.environ["TOGETHER_API_KEY"] = llm_config.api_key 

        # 2. Initialize LangChain Embedding Function
        print(f"🧠 Loading LangChain Embeddings: {embedding_config.embedding_model_name}")
        self.embeddings = get_langchain_huggingface_embeddings(
            model_name=embedding_config.embedding_model_name,
            device=embedding_config.embedding_model_device
        )
        print("✅ Embeddings loaded.")

        # 3. Initialize LangChain Chroma Vector Store
        print(f"🗃️ Connecting to LangChain ChromaDB at: {vector_store_config.persist_directory}/{vector_store_config.collection_name}")
        self.vector_store = get_langchain_chroma_vector_store(
            embedding_function=self.embeddings,
            persist_directory=vector_store_config.persist_directory,
            collection_name=vector_store_config.collection_name
        )
        
        # Check if database is populated
        if self.vector_store._collection.count() == 0:
            raise ValueError(
                f"Vector database collection '{vector_store_config.collection_name}' is empty. "
                f"Please run 'python data_ingestion.py' first to populate it."
            )
        print(f"✅ Vector Store connected. Contains {self.vector_store._collection.count()} documents.")

        # 4. Create a LangChain Retriever
        print("🔎 Creating LangChain Retriever...")
        self.base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieval_config.n_results}
        )
        print("✅ Base Retriever ready.")

        # 5. Initialize LangChain LLM
        print(f"🗣️ Initializing LangChain LLM: {llm_config.model_name} (Provider: {llm_config.provider})")
        self.llm = Together(
            model=llm_config.model_name,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
        )
        print("✅ LLM initialized.")

        # 6. Define the RAG Prompt Template from YAML
        print("📝 Setting up RAG Prompt Template from config...")
        
        prompt_name_to_use = "lehninger_rag_prompt_cfg" 
        prompt_config_data = config.get_config_manager().get_prompt_template(prompt_name_to_use)

        print(f"DEBUG: Loaded prompt_config_data: {prompt_config_data}")
        print(f"DEBUG: Type of prompt_config_data: {type(prompt_config_data)}")
        if isinstance(prompt_config_data, dict):
            print(f"DEBUG: Type of 'role': {type(prompt_config_data.get('role'))}")
            print(f"DEBUG: Type of 'instruction': {type(prompt_config_data.get('instruction'))}")
            print(f"DEBUG: Type of 'goal': {type(prompt_config_data.get('goal'))}")
            print(f"DEBUG: Type of 'reasoning_strategy': {type(prompt_config_data.get('reasoning_strategy'))}")
        
        if not prompt_config_data:
            raise ValueError(f"Prompt template '{prompt_name_to_use}' not found in prompt_config.yaml!")

        # Get the reasoning strategy text based on the key specified in the prompt config
        reasoning_strategy_key = prompt_config_data.get("reasoning_strategy")
        cot_strategy_text = None
        
        if reasoning_strategy_key:
            loaded_strategy = config.get_config_manager().get_reasoning_strategy(reasoning_strategy_key)
            if loaded_strategy and isinstance(loaded_strategy, dict):
                cot_strategy_text = loaded_strategy.get(reasoning_strategy_key)
            elif isinstance(loaded_strategy, str):
                cot_strategy_text = loaded_strategy
            
            if not cot_strategy_text:
                print(f"⚠️ Warning: Reasoning strategy '{reasoning_strategy_key}' found in prompt config but its content is missing or malformed in config.yaml.")

        # Assemble the system message using parts from your YAML
        system_message_parts = []

        if prompt_config_data.get("role"):
            system_message_parts.append(prompt_config_data["role"].strip())
        if prompt_config_data.get("instruction"):
            system_message_parts.append(prompt_config_data["instruction"].strip())
        if prompt_config_data.get("goal"):
            system_message_parts.append(f"\nGoal: {prompt_config_data['goal'].strip()}")

        output_constraints = prompt_config_data.get("output_constraints")
        if output_constraints and isinstance(output_constraints, list):
            system_message_parts.append("\nOutput Constraints:")
            system_message_parts.extend([f"  - {c.strip()}" for c in output_constraints])

        style_or_tone = prompt_config_data.get("style_or_tone")
        if style_or_tone and isinstance(style_or_tone, list):
            system_message_parts.append("\nStyle/Tone:")
            system_message_parts.extend([f"  - {s.strip()}" for s in style_or_tone])

        if cot_strategy_text:
            system_message_parts.append(f"\nReasoning Strategy:\n{cot_strategy_text.strip()}")

        base_system_prompt_content = "\n\n".join(filter(None, system_message_parts)) 

        # 7. Construct the LangChain RAG Chain
        print("🔗 Building LangChain RetrievalQA Chain with custom prompt...")

        combined_template = f"{base_system_prompt_content}\n\nContext:\n{{context}}\n\nQuestion: {{question}}"
        self.rag_prompt = PromptTemplate(
            template=combined_template,
            input_variables=["context", "question"]
        )
        print("DEBUG: Final prompt template:")
        print("-" * 40)
        print(self.rag_prompt.template)
        print("-" * 40)

        print("\n✨ RAG Pipeline initialized and ready to answer biochemistry questions!")
        print("=" * 60)

    def query(self, question: str, filters: Optional[Dict[str, Any]] = None) -> dict:
        """
        Queries the RAG pipeline with a user question, optionally applying metadata filters.
        
        Args:
            question (str): The user's question.
            filters (Optional[Dict[str, Any]]): A dictionary of metadata filters to apply.
                                                 E.g., {"chapter_number": "22"}
                                                 These will be passed to ChromaDB's where_clause.
            
        Returns:
            dict: A dictionary containing the LLM's answer and source documents.
                  Example: {"query": "...", "result": "...", "source_documents": [...]}
        """
        print(f"\n🔬 Querying RAG Pipeline for: '{question}'")
        if filters:
            print(f"Applying filters: {filters}")

        try:
            current_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.base_retriever.search_kwargs['k'], "filter": filters}
            )

            # Rebuild the RAG chain with the potentially filtered retriever for this query
            rag_chain_with_filters = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=current_retriever, 
                verbose=False,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.rag_prompt}
            )

            retrieved_docs = current_retriever.get_relevant_documents(question) # Use current_retriever here
            print(f"DEBUG: Retrieved {len(retrieved_docs)} documents.")
            if not retrieved_docs:
                print("DEBUG: No documents retrieved with the current filters and query.")
            for i, doc in enumerate(retrieved_docs):
                source_file = doc.metadata.get('source_file', 'N/A')
                section = doc.metadata.get('Section', 'N/A')
                chapter_title = doc.metadata.get('chapter_title', 'N/A')
                chapter_number = doc.metadata.get('chapter_number', 'N/A')
                print(f"DEBUG: Doc {i+1} (Source: {source_file}, Chapter: {chapter_number} - {chapter_title}, Section: {section}), Length: {len(doc.page_content)} chars")
            
            response = rag_chain_with_filters.invoke({"query": question}) # Use the chain with filters

            # ---  POST-PROCESSING (Safety Net) ---
            if "result" in response and isinstance(response["result"], str):
                llm_raw_answer = response["result"]
    
                unwanted_patterns = [
                    r'Right Answer:\s*.*?(?=Answer:|$)',
                    r'Explanation of the Solution:\s*.*?(?=Related Questions:|$)',
                    r'Related Questions:\s*.*$',
                    r'Sources:\s*.*$',
                    r'^\s*Answer:\s*Answer:\s*',
                    r'^\s*Answer:\s*',
                    r'^\s*The answer is:\s*',
                    ]
    
                cleaned_answer = llm_raw_answer
                for pattern in unwanted_patterns:
                    cleaned_answer = re.sub(pattern, '', cleaned_answer, flags=re.DOTALL | re.IGNORECASE)
    
                cleaned_answer = cleaned_answer.strip()
    
                if cleaned_answer != llm_raw_answer:
                    print(f"DEBUG: Post-processing cleaned up response")
                    response["result"] = cleaned_answer

            return response
        except Exception as e:
            print(f"❌ Error during RAG pipeline query: {e}")
            print("--- FULL TRACEBACK ---")
            traceback.print_exc()
            print("----------------------")
            return {"query": question, "result": f"An error occurred: {e}", "source_documents": []}

if __name__ == "__main__":
    print("--- Running a quick test of the RAG pipeline ---")
    try:
        pipeline = BiochemistryRAGPipeline()
        
        test_questions = [
            "What are the main functions of ATP in a cell?",
            "Explain the concept of enzyme specificity.",
            "What is glycolysis and where does it occur?",
            "What is the role of the citric acid cycle?",
            "Tell me about quantum physics." # Out-of-scope question
        ]

        # Example of using filters
        print("\n--- Testing with a specific chapter filter (Chapter 22) ---")
        filtered_question = "What is the primary role of transaminases in amino acid metabolism?"
        # Assuming transaminases are discussed in chapter 22
        filtered_results = pipeline.query(filtered_question, filters={"chapter_number": "22"})
        print(f"\n🤖 ANSWER (Filtered to Chapter 22):\n{filtered_results.get('result', 'No answer found.')}")
        if filtered_results.get('source_documents'):
            print("\n📄 SOURCES (Filtered):")
            for doc in filtered_results['source_documents']:
                source_file = doc.metadata.get('source_file', 'N/A')
                section = doc.metadata.get('Section', 'N/A')
                chapter_title = doc.metadata.get('chapter_title', 'N/A')
                chapter_number = doc.metadata.get('chapter_number', 'N/A')
                print(f"   - Source: {source_file}, Chapter: {chapter_number} - {chapter_title}, Section: {section}")
        else:
            print("\n📄 No specific sources retrieved with filter.")
        print("-" * 80)
        print("\n")


        for i, q in enumerate(test_questions):
            print(f"\n" + "-"*80)
            print(f"QUESTION {i+1}: {q}")
            results = pipeline.query(q) # No filters applied here, general search
            
            print(f"\n🤖 ANSWER:\n{results.get('result', 'No answer found.')}")
            
            if results.get('source_documents'):
                print("\n📄 SOURCES:")
                for doc in results['source_documents']:
                    source_file = doc.metadata.get('source_file', 'N/A')
                    section = doc.metadata.get('Section', 'N/A')
                    chapter_title = doc.metadata.get('chapter_title', 'N/A')
                    chapter_number = doc.metadata.get('chapter_number', 'N/A')
                    print(f"   - Source: {source_file}, Chapter: {chapter_number} - {chapter_title}, Section: {section}")
            else:
                print("\n📄 No specific sources retrieved.")

            print("-" * 80)
            print("\n")

    except ValueError as e:
        print(f"\n❗ ERROR: {e}")
        print("Please ensure your vector database is populated by running 'python data_ingestion.py' first.")
        print("Also check your API key is set correctly in environment or config.yaml.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
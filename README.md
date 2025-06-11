# Lehninger Biochem QA Assistant 🧬 - Chapter 22 Edition

Welcome to my RAG-powered AI Assistant, specifically trained on Chapter 22 of Lehninger Principles of Biochemistry! This project was quite the journey, and I'm excited to share what I've built.

## 🌟 What is this?

This is a Retrieval-Augmented Generation (RAG) system designed to answer your questions about the biosynthesis of amino acids, nucleotides, and related molecules, purely based on the content of Lehninger Principles of Biochemistry, Chapter 22. Think of it as having a highly focused, super-smart study buddy for one of biochemistry's most intricate chapters.

My main goal with this project was to build an intelligent agent that *truly* sticks to its given context and doesn't make things up – a challenge I quickly learned is easier said than done with LLMs!

## ✨ Features

* **Context-Aware Q&A:** Ask questions about Chapter 22, and the system will fetch relevant information to craft its answer.
* **Strict Grounding:** Designed with extensive prompt engineering to minimize "hallucinations" and ensure answers come directly from the provided text. If the information isn't in Chapter 22, it's trained to tell you exactly that!
* **Custom Knowledge Base:** Uses a PDF of Lehninger Principles of Biochemistry, Chapter 22, as its sole source of truth.
* **Vector Store Integration:** Leverages [ChromaDB / your chosen vector store] for efficient semantic search and retrieval.
* **Powerful LLM:** Powered by the [Mistral 7B Instruct v0.2] Large Language Model for generating coherent and precise responses.
* **CoT Reasoning:** Incorporates Chain-of-Thought (CoT) prompting to encourage more systematic internal reasoning by the LLM.
* **Minimal UI:** A simple Streamlit interface for easy interaction.

## 🛠️ How it Works (The Tech Stack)

This system is built around a classic RAG architecture:

1.  **Document Loading & Chunking:** The Lehninger Chapter 22 PDF is loaded and broken down into smaller, manageable chunks of text.
2.  **Embedding:** Each text chunk is converted into numerical vector embeddings using an embedding model ([e.g., `sentence-transformers/all-MiniLM-L6-v2` - mention the one you're using]).
3.  **Vector Storage:** These embeddings are stored in a [ChromaDB / your chosen vector store], making them searchable.
4.  **Retrieval:** When you ask a question, your query is also embedded, and the system finds the most semantically similar text chunks from the vector store.
5.  **LLM Generation:** The retrieved chunks, along with your original question and a carefully crafted prompt, are fed to the [Mistral 7B Instruct v0.2] LLM. The LLM then synthesizes an answer based *only* on that provided information.

## 🚀 Getting Started

To run this biochemical assistant, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-link]
    cd [your-repo-directory]
    ```
2.  **Set up your environment:**
    ```bash
    # It's highly recommended to use a virtual environment
    python -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3.  **Place your document:**
    * Ensure your Lehninger Chapter 22 PDF (or relevant document) is placed in the `data/` directory.
    * **Important:** This system is specifically designed for content related to Chapter 22. Using vastly different content might require prompt adjustments.
4.  **Configure your settings:**
    * Open `config.py` to adjust core settings like the LLM model name (`Mistral 7B Instruct v0.2`), the `temperature` (I found `0.0` to be crucial for strict adherence!), chunk sizes, and embedding model.
    * Ensure your `TOGETHER_API_KEY` (or relevant API key for your chosen LLM provider) is set as an environment variable or directly in `config.py` (though environment variables are recommended for security).
    * You can also fine-tune the reasoning strategies in `config.yaml`.
5.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
    Your browser should automatically open the application!

## 🤯 The Journey & Key Learnings (This is where it gets personal!)

Building this RAG system, especially enforcing strict grounding, was an adventure! I quickly realized that getting an LLM to *only* use provided context is a significant challenge.

* **The Hallucination Headache:** My initial versions loved to make up answers or pull from general knowledge, even when explicitly told not to. This was the biggest hurdle.
* **The Looping Labyrinth:** At one point, the LLM got stuck in infinite self-correction loops when it couldn't perfectly follow complex instructions, which was both frustrating and fascinating to debug!
* **Prompt Engineering is Key:** I spent a considerable amount of time meticulously crafting and refining the prompt. I learned that even subtle phrasing can dramatically alter an LLM's behavior, and sometimes, less is more when it comes to overly aggressive negative constraints. I experimented with different LLMs too, finding that `Mistral 7B Instruct v0.2` paired with a `temperature` of `0.0` and a finely-tuned prompt provided the best balance of accuracy and adherence.
* **Patience is a Virtue:** Debugging LLM behavior can be unpredictable and exhausting, but the incremental improvements were incredibly rewarding.

While there are still minor artifacts (like a stray `? Answer:` or an occasional "However, it is known from other sources..." when context is completely missing – a small price to pay for otherwise robust grounding!), I'm incredibly proud of how far this system has come in its ability to be a reliable and accurate source of information from its specific knowledge base.

## 📈 Evaluation & Future Work

* **Current Evaluation:** Evaluation during development was primarily **manual and iterative**. I tested numerous queries, carefully analyzing the LLM's responses for accuracy, conciseness, and adherence to the strict grounding rules. This hands-on approach allowed me to quickly identify and address issues like hallucination and unwanted commentary.
* **Future Enhancements (Maybe for another time!):**
    * **Automated Evaluation Loop:** Implementing a formal automated evaluation pipeline with a "gold standard" question-answer dataset would be a valuable next step for systematic performance tracking.
    * **Memory Components:** Adding conversational memory to allow for follow-up questions within a session.
    * **Broader Context:** Expanding the knowledge base to include more chapters or even the entire Lehninger textbook.

## 🙏 Acknowledgements

* My instructor for this project and the valuable insights provided throughout this learning experience.
* The creators of Lehninger Principles of Biochemistry, Chapter 22, for the foundational knowledge!
* [Optional: If you used any specific online resources, tutorials, or even "My ever-patient AI assistant for debugging prompt nightmares" 😉]

---
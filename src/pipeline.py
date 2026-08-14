"""
Document Q&A Pipeline — YOUR WORK GOES HERE.

The knowledge base (loading, chunking, vector store) is already built
for you in knowledge_base.py. Your job is to:

  1. Retrieve relevant chunks and generate an answer
  2. Wire it up into an interactive CLI

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - HuggingFace pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""

import argparse
import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm():
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


def ask_question(vector_store, llm, question: str) -> dict:
    """Retrieve relevant chunks and generate an answer.

    Steps:
      1. Use vector_store.similarity_search(question, k=3) to get
         the top 3 most relevant document chunks.
      2. Combine the chunk text into a single context string.
         (Hint: each chunk has a .page_content attribute)
      3. Format the PROMPT_TEMPLATE with the context and question.
      4. Pass the formatted prompt to llm(...) and extract the
         generated text from the result.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved
    """
    question = (question or "").strip()
    if not question:
        return {"answer": "Please enter a question.", "sources": []}

    docs = vector_store.similarity_search(question, k=3)
    sources = [doc.page_content for doc in docs]
    context = "\n\n".join(sources)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    result = llm(prompt)
    answer = result[0]["generated_text"]
    return {"answer": answer, "sources": sources}


def print_result(result: dict) -> None:
    sources = result.get("sources") or []
    if sources:
        print("\n📄 Sources:")
        for i, source in enumerate(sources, start=1):
            preview = " ".join(source.split())
            if len(preview) > 160:
                preview = preview[:157] + "..."
            print(f"  {i}. {preview}")
    print(f"\n💬 Answer: {result['answer']}\n")


def main(argv=None) -> None:
    """Interactive Q&A loop.

    Steps:
      1. Build the knowledge base using build_knowledge_base()
         with the data/ directory path.
      2. Load the LLM using get_llm().
      3. Start a loop that:
         - Prompts the user for a question with input()
         - Exits if they type "quit"
         - Calls ask_question() with their input
         - Prints the retrieved sources and the answer
    """
    parser = argparse.ArgumentParser(description="Marketing agency Q&A chatbot")
    parser.add_argument("--query", "-q", help="Ask a single question and exit")
    parser.add_argument("--data-dir", default=None, help="Path to the documents folder")
    args = parser.parse_args(argv)

    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "..", "data")
    data_dir = os.path.abspath(data_dir)

    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    if not txt_files:
        print(f"Error: no .txt files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    vector_store = build_knowledge_base(data_dir)
    llm = get_llm()

    if args.query is not None:
        question = args.query.strip()
        if not question:
            print("Error: --query cannot be empty.", file=sys.stderr)
            sys.exit(1)
        print_result(ask_question(vector_store, llm, question))
        return

    print("Ask a question about our services, pricing, or process.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            print("Please enter a question, or type 'quit' to exit.\n")
            continue

        if question.lower() == "quit":
            print("Goodbye!")
            break

        print_result(ask_question(vector_store, llm, question))


if __name__ == "__main__":
    main()

import argparse

from ingestion.youtube_loader import load_transcript
from processing.splitter import split_text
from vector_store.faiss_store import create_vector_store
from retrieval.retriever import get_retriever
from utils.prompt import get_prompt

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

from config import OLLAMA_MODEL


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(retriever, prompt, llm):
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    chain = parallel_chain | prompt | llm | StrOutputParser()
    return chain


def main():
    parser = argparse.ArgumentParser(description="YouTube RAG Chatbot")
    parser.add_argument(
        "--video-id",
        type=str,
        help="YouTube video ID to query (e.g. 8TZMtslA3UY)",
    )
    args = parser.parse_args()

    video_id = args.video_id
    if not video_id:
        video_id = input("Enter YouTube video ID (from youtube.com/watch?v=VIDEO_ID): ").strip()
    if not video_id:
        raise ValueError("A YouTube video ID is required.")

    # Step 1: Load transcript
    print("Loading transcript...")
    transcript = load_transcript(video_id)
    print(f"Transcript length: {len(transcript)} characters")

    # Step 2: Split into chunks
    print("Splitting transcript into chunks...")
    docs = split_text(transcript)
    print(f"Number of chunks: {len(docs)}")

    if not docs:
        raise ValueError("No documents generated. Check transcript loading.")

    # Step 3: Build vector store
    print("Building vector store...")
    vector_store = create_vector_store(docs)

    # Step 4: Get retriever
    retriever = get_retriever(vector_store)

    # Step 5: Get prompt template
    prompt = get_prompt()

    # Step 6: Initialize Ollama LLM
    print(f"Initializing Ollama with model: {OLLAMA_MODEL}")
    llm = OllamaLLM(model=OLLAMA_MODEL)

    # Step 7: Build the RAG chain
    chain = build_chain(retriever, prompt, llm)

    # Step 8: Interactive Q&A loop
    print("\nChatbot ready! Type 'exit' or 'quit' to stop.\n")
    while True:
        query = input("Your question: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        result = chain.invoke(query)
        print("\n===== ANSWER =====")
        print(result)
        print()


if __name__ == "__main__":
    main()
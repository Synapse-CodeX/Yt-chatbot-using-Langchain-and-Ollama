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
    video_id = "8TZMtslA3UY"

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

    # Step 8: Run a query
    query = "Can you summarize the video?"
    print(f"\nRunning query: {query}\n")

    result = chain.invoke(query)

    print("===== FINAL ANSWER =====\n")
    print(result)


if __name__ == "__main__":
    main()
from langchain_core.prompts import PromptTemplate

def get_prompt():
    return PromptTemplate(
        template="""
You are a helpful assistant.

Answer ONLY from the provided context.
If the context is insufficient, say "I don't know."

Context:
{context}

Question: {question}
""",
        input_variables=["context", "question"]
    )
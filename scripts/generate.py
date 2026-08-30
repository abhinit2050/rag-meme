"""
Step 5 of the Meme History RAG pipeline: grounded generation.

Retrieves the top-k chunks for a question (step 4) and asks an LLM to answer
using ONLY those chunks, citing meme name + source URL for every claim. Any
chunk that doesn't clear MIN_RELEVANCE_SCORE is dropped before the LLM ever
sees it -- if nothing clears the bar, the LLM is never called and a fixed
decline message is returned instead. This two-layer approach (score filter +
prompt instruction) is what keeps this from hallucinating on out-of-corpus
questions.
"""

import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from retrieve import retrieve

load_dotenv()

CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
MIN_RELEVANCE_SCORE = 0.72  # from step 4: matching chunks scored ~0.78-0.82, unrelated ones ~0.67
DECLINE_MESSAGE = "I don't have enough information in my sources to answer that."

SYSTEM_PROMPT = """You are a meme historian. Answer the user's question using ONLY the provided context chunks below -- do not use any outside knowledge, even if you know the answer.

Rules:
- Every claim in your answer must be directly supported by the context.
- After your answer, add a line "Sources:" followed by "<meme name> - <source url>" for each distinct meme you drew from.
- If the context does not contain enough information to answer the question, respond with exactly: "I don't have enough information in my sources to answer that." Do not guess or fill gaps with outside knowledge.
"""


def format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c['meme_name']} - {c['section']}] ({c['source_url']})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer(question: str) -> str:
    chunks = retrieve(question, k=TOP_K)
    relevant = [c for c in chunks if c["score"] >= MIN_RELEVANCE_SCORE]

    if not relevant:
        return DECLINE_MESSAGE

    context = format_context(relevant)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n\n{context}\n\nQuestion: {question}"),
    ]
    response = llm.invoke(messages)
    return response.content


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Why is Rickroll called Rickroll?"
    print(f"[question] {question}\n")
    print(answer(question))


if __name__ == "__main__":
    main()

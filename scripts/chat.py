"""
Step 6 of the Meme History RAG pipeline: conversational wrapper.

Adds multi-turn chat on top of step 5's grounded generation. Follow-up
questions ("why is it called that?") get rewritten into a standalone query
using the conversation history before retrieval runs, since vector search
has no notion of prior turns. Generation still grounds strictly in the
retrieved chunks (same score-gated decline path as generate.py) -- chat
history is only used for condensing the query and giving the LLM
conversational tone, never as a substitute for retrieved evidence.
"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from generate import CHAT_MODEL, DECLINE_MESSAGE, MIN_RELEVANCE_SCORE, SYSTEM_PROMPT, TOP_K, format_context
from retrieve import retrieve

load_dotenv()

CONDENSE_PROMPT = """Given the conversation history and a follow-up question, rewrite the \
follow-up into a standalone question that makes sense without the history. Do not answer \
it, just rewrite it. If the follow-up is already standalone, return it unchanged.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""


def condense_question(question: str, history: list[BaseMessage]) -> str:
    if not history:
        return question
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in history
    )
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    response = llm.invoke([HumanMessage(content=CONDENSE_PROMPT.format(history=history_text, question=question))])
    return response.content.strip()


def answer(question: str, history: list[BaseMessage]) -> tuple[str, str]:
    """Returns (response, standalone_question) -- the latter is surfaced so a REPL/caller can show what was actually searched for."""
    standalone = condense_question(question, history)
    chunks = retrieve(standalone, k=TOP_K)
    relevant = [c for c in chunks if c["score"] >= MIN_RELEVANCE_SCORE]

    if not relevant:
        return DECLINE_MESSAGE, standalone

    context = format_context(relevant)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *history,
        HumanMessage(content=f"Context:\n\n{context}\n\nQuestion: {standalone}"),
    ]
    response = llm.invoke(messages)
    return response.content, standalone


def main() -> None:
    print("Meme History RAG chat -- type 'exit' to quit.\n")
    history: list[BaseMessage] = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in ("exit", "quit"):
            break

        response, standalone = answer(question, history)
        if standalone != question:
            print(f"[searched for] {standalone}")
        print(f"Bot: {response}\n")

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=response))


if __name__ == "__main__":
    main()

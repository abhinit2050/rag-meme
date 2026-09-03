"""
Streamlit chat frontend for the Meme History RAG pipeline.

Wraps scripts/chat.py's answer() (query condensation -> retrieval ->
grounded generation) in a chat UI. No RAG logic lives here -- this is
presentation only.
"""

import sys
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from chat import answer  # noqa: E402

st.set_page_config(page_title="Meme History RAG", page_icon="🕵️", layout="centered")

THEMES = {
    "dark": {
        "bg": "#0E0A1A",
        "sidebar_bg": "#160F26",
        "card_bg": "#1A1330",
        "text": "#EDE9FE",
        "muted_text": "#B7ACD6",
        "accent": "#9B6DFF",
        "accent_text": "#FFFFFF",
        "border": "#332658",
        "input_bg": "#1A1330",
    },
    "light": {
        "bg": "#FBF8FF",
        "sidebar_bg": "#F1E9FB",
        "card_bg": "#FFFFFF",
        "text": "#241C3D",
        "muted_text": "#6B5E8C",
        "accent": "#7C3AED",
        "accent_text": "#FFFFFF",
        "border": "#E1D3FA",
        "input_bg": "#FFFFFF",
    },
}


def inject_css(theme: str) -> None:
    t = THEMES[theme]
    st.markdown(
        f"""
        <style>
        html, html body {{
            background-color: {t['bg']} !important;
        }}
        html body [data-testid="stHeader"] {{
            background-color: {t['bg']} !important;
        }}
        .stApp {{
            background-color: {t['bg']};
            color: {t['text']};
        }}
        [data-testid="stSidebar"] {{
            background-color: {t['sidebar_bg']};
            border-right: 1px solid {t['border']};
        }}
        [data-testid="stSidebar"] * {{
            color: {t['text']};
        }}
        h1, h2, h3 {{
            color: {t['text']} !important;
        }}
        p, span, label, .stMarkdown {{
            color: {t['text']};
        }}
        [data-testid="stCaptionContainer"] {{
            color: {t['muted_text']} !important;
        }}
        [data-testid="stChatMessage"] {{
            background-color: {t['card_bg']};
            border: 1px solid {t['border']};
            border-radius: 14px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
        }}
        html body [data-testid="stBottom"] {{
            background-color: {t['bg']} !important;
        }}
        html body [data-testid="stBottom"] * {{
            background-color: transparent !important;
        }}
        html body [data-testid="stChatInput"],
        html body [data-testid="stChatInput"] * {{
            background-color: {t['input_bg']} !important;
        }}
        html body [data-testid="stChatInput"] {{
            border: 1px solid {t['border']};
            border-radius: 12px;
        }}
        html body [data-testid="stChatInput"] textarea,
        html body [data-testid="stChatInputTextArea"] {{
            background-color: {t['input_bg']} !important;
            color: {t['text']} !important;
            -webkit-text-fill-color: {t['text']} !important;
        }}
        html body [data-testid="stChatInput"] textarea::placeholder,
        html body [data-testid="stChatInputTextArea"]::placeholder {{
            color: {t['muted_text']} !important;
            -webkit-text-fill-color: {t['muted_text']} !important;
        }}
        button {{
            border-radius: 10px !important;
        }}
        button[kind="secondary"], button[kind="primary"] {{
            background-color: {t['accent']} !important;
            color: {t['accent_text']} !important;
            border: none !important;
        }}
        [data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {{
            background-color: {t['accent']} !important;
        }}
        hr {{
            border-color: {t['border']} !important;
        }}
        a {{
            color: {t['accent']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("### 🕵️ Meme History RAG")
    st.caption("Grounded answers about internet memes, sourced from Know Your Meme.")
    theme_label = st.radio("Theme", ["Dark", "Light"], horizontal=True, key="theme_radio")
    st.markdown("---")
    if st.button("🗑️ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.caption(
        "Answers are grounded only in the ingested corpus. If a meme isn't in the "
        "sources, the bot says so instead of guessing."
    )

theme = theme_label.lower()
inject_css(theme)

st.title("Meme History RAG")
st.caption(
    "Ask about a meme's origin, spread, or history — every answer cites its source, "
    "or the bot tells you it doesn't know."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about a meme..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Digging through the archives..."):
            response, standalone = answer(question, st.session_state.history)
        if standalone != question:
            st.caption(f"Searched for: _{standalone}_")
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.history.append(HumanMessage(content=question))
    st.session_state.history.append(AIMessage(content=response))

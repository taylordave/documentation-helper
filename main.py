from typing import Set

from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

# Configure page
st.set_page_config(
    page_title="LangChain Documentation Helper",
    page_icon="🦜",  # LangChain's bird emoji
    layout="wide"
)

# Custom CSS to match LangChain's style
st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
        padding: 0;
    }
    
    section[data-testid="stSidebar"] {
        width: 300px !important;
        margin-left: 0;
        padding-right: 1rem;
        background-color: #F0F2F6;
    }
    
    section.main {
        margin-left: 300px;
        padding: 2rem 3rem;
        max-width: 1200px;
    }
    
    .stTitle {
        font-size: 2rem !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stToolbar"] {
        background-color: #F0F2F6;
    }
    
    .stTextInput > div > div > input {
        border-radius: 0.5rem;
    }

    /* Chat message styling */
    [data-testid="stChatMessage"] {
        background-color: white !important;
        max-width: 800px !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
    }
    
    [data-testid="stChatMessage"] > div {
        white-space: pre-wrap !important;       /* preserve line breaks */
        overflow-wrap: break-word !important;
        word-wrap: break-word !important;
        word-break: break-all !important;
    }
    
    [data-testid="stChatMessage"] code {
        white-space: pre-wrap !important;
        word-break: break-all !important;
        overflow-wrap: break-word !important;
    }
    
    .element-container iframe {
        max-width: 800px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Add sidebar with user information
with st.sidebar:
    st.title("User Profile")
    
    # Get user information if available
    if st.experimental_user.email:
        # Show authenticated user info
        st.image("https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp", width=100)
        st.write(f"👤 {st.experimental_user.email}")
    else:
        # Show guest message
        st.info("👋 Welcome Guest! Sign in to access all features.")
    
    # Add a divider for visual separation
    st.divider()

# Main content
st.title("🦜 LangChain Documentation Helper")
st.markdown("""
    <div style='background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
    Ask questions about LangChain and get answers from the documentation.
    </div>
""", unsafe_allow_html=True)

prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")

if (
        "chat_answers_history" not in st.session_state
        or "user_prompt_history" not in st.session_state
        or "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "Sources:\n"
    for i, source in enumerate(sources_list):
        # Add zero-width spaces after slashes to allow breaking
        formatted_source = source.replace("/", "/\u200B")
        sources_string += f"{i + 1}. {formatted_source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt,
            chat_history=st.session_state["chat_history"],
        )
        sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])

        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))


if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True)
        message(generated_response)

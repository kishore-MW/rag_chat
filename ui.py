import streamlit as st
import os
from chat import process_query

# ---------------------- Page Setup -------------------------------------
st.set_page_config(page_title="Chatbot", layout="centered")
st.title("💬 MaintWiz AI Assistant")

# ---------------------- PDF Selection ----------------------------------
pdf_folder = "processed_pdfs"  # Update this to your PDF directory path
available_pdfs = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

st.sidebar.title("📄 Select PDFs to Use")
selected_pdfs = st.sidebar.multiselect("Choose documents", available_pdfs)

# ---------------------- Session State Initialization ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------- Chat Bubble Layout ------------------------------
def chat_bubble_open(role):
    align = "flex-end" if role == "user" else "flex-start"
    bg_color = "#dcf8c6" if role == "user" else "#f1f0f0"
    return f"""
    <div style='display: flex; justify-content: {align}; margin-bottom: 10px;'>
        <div style='background-color: {bg_color}; padding: 10px 15px;
                    border-radius: 15px; max-width: 75%; word-wrap: break-word;
                    font-size: 15px; line-height: 1.4;'>
    """

def chat_bubble_close():
    return "</div></div>"

def full_chat_bubble(content, role):
    return chat_bubble_open(role) + content + chat_bubble_close()

# ---------------------- Display Chat History ----------------------
for msg in st.session_state.messages:
    st.markdown(full_chat_bubble(msg["content"], msg["role"]), unsafe_allow_html=True)

# ---------------------- Chat Input ----------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Save and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(full_chat_bubble(user_input, "user"), unsafe_allow_html=True)

    # Placeholder for streamed bot response
    placeholder = st.empty()

    with st.spinner("Thinking..."):
        # 🧠 Pass selected PDFs into process_query
        response = process_query(user_input, doc_name = selected_pdfs)

        streamed_response = ""
        open_html = chat_bubble_open("bot")
        close_html = chat_bubble_close()

        for word in response:
            streamed_response += word
            placeholder.markdown(open_html + streamed_response + close_html, unsafe_allow_html=True)

    # Save final message
    st.session_state.messages.append({"role": "bot", "content": streamed_response})

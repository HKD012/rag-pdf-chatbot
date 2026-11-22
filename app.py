"""Streamlit chat UI for the RAG PDF chatbot. Run with : streamlit run app.py"""
import streamlit as st

from rag import answer_question, build_vectorstore, get_embeddings, load_pdf, split_documents

st.set_page_config(page_title="Banking Document Q&A", page_icon="💬", layout="wide")
st.title("💬 Banking Document Q&A Chatbot (RAG)")
st.caption("Upload a PDF (for example a bank FAQ or Policy) and ask questions about it. "
           "The chatbot will answer using only the content of the PDF.")


@st.cache_resource  # load the embeddings model only once, not on every click
def cached_embeddings():
    return get_embeddings()


uploaded = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded is not None and st.session_state.get("file_name") != uploaded.name:
    with st.spinner("Reading and indexing the document..."):
        pages = load_pdf(uploaded)
        if not pages:
            st.sidebar.error("No readable text found (scanned PDFs are not supported).")
            st.stop()
        chunks = split_documents(pages)
        st.session_state.vectorstore = build_vectorstore(chunks, embeddings=cached_embeddings())
        st.session_state.file_name = uploaded.name
        st.session_state.messages = []
    st.sidebar.success(f"Indexed {len(pages)} pages into {len(chunks)} chunks.")

if "vectorstore" not in st.session_state:
    st.info("Upload a PDF to start asking questions.")
    st.stop()

# show previous messages
for msg in st.session_state.get("messages", []):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question about the document:")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = answer_question(st.session_state.vectorstore, question)
            except Exception as e:  # network error, quota, bad key ...
                answer, sources = f"Sorry, something went wrong: {e}", []
        st.write(answer)
        if sources:
            with st.expander("Sources"):
                for d in sources:
                    st.markdown(f"**Page {d.metadata['page']}**")
                    st.write(d.page_content[:300] + "...")
    st.session_state.messages.append({"role": "assistant", "content": answer})             

import os

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful banking assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say: "I could not find this in the document."

Context:
{context}

Question: {question}

Answer:"""
)

def load_pdf(file) -> list[Document]:
    """Read a PDF (path or uploaded file object) and return one Document per page."""
    reader = PdfReader(file)
    docs = []
    for page_number , page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={"page": page_number}))
    return docs

def split_documents(docs: list[Document]) -> list[Document]:
    """Cut pages into overlapping chunks (metadata such as a page number is kept)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

def get_embeddings():
    """Free local embedding model: turns text into 384-number vectors."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_vectorstore(chunks: list[Document], embeddings=None) -> FAISS:
    """Embed every chunk and store the vectors in a FAISS index."""
    if embeddings is None:
        embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)

def get_llm():
    """Gemini chat model. temperature=0 -> focused,repeatable answers."""
    return ChatGoogleGenerativeAI(
        model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        temperature = 0,
    )

def answer_question(vectorstore: FAISS, question: str, llm=None):
    """Retrieve the top-k chunks,build the prompt, ask the LLM. Returns (answer,sources)."""
    retrieved = vectorstore.similarity_search(question, k=TOP_K)
    context = "\n\n".join(
        f"[Page {d.metadata['page']}] {d.page_content}" for d in retrieved)
    chain = PROMPT | (llm or get_llm()) | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return answer, retrieved
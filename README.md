# Document Q&A Chatbot using RAG

A chatbot that answers questions from an uploaded PDF (for example bank FAQs or tariff schedules) and shows the source pages. It refuses to answer when the information is not in the document.

## Tech stack
Python, LangChain, Sentence-Transformers (all-MiniLM-L6-v2), FAISS, Google Gemini API, Streamlit

## How it works
1. Extract text from the PDF (pypdf)
2. Split into overlapping chunks (800 characters, 100 overlap)
3. Embed chunks and store them in a FAISS vector index
4. For each question, retrieve the top 4 similar chunks
5. Send the chunks and the question to Gemini with an "answer only from context" prompt
6. Show the answer and the source pages

## Run locally
    pip install -r requirements.txt
    Create a .env file containing: GOOGLE_API_KEY=your_key
    streamlit run app.py

## Limitations
Text PDFs only (no OCR). Table PDFs can be extracted out of order, so values may be matched to the wrong row. The index is in memory and rebuilt for each upload.
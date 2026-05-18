import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from ner_extractor import run_extraction

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


def load_pdf_robust(pdf_path: str) -> list:
    """
    Load a PDF into LangChain Documents using pdfplumber when available.
    Falls back to PyPDFLoader for standard PDFs.
    """
    if PDFPLUMBER_AVAILABLE:
        try:
            docs = []
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": pdf_path, "page": i}
                        ))
            if docs:
                print(f"  PDF loaded via pdfplumber ({len(docs)} pages).")
                return docs
        except Exception as e:
            print(f"  pdfplumber failed ({e}), falling back to PyPDFLoader...")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"  PDF loaded via PyPDFLoader ({len(docs)} pages).")
    return docs


def build_knowledge_base():
    print("1. Running NER extraction on CV...")
    try:
        run_extraction("data/cv.pdf")
    except Exception as e:
        print(f"  NER extraction failed (non-fatal): {e}")

    print("2. Loading documents...")
    cv_docs = load_pdf_robust("data/cv.pdf")

    jd_loader = TextLoader("data/job_description.txt")
    jd_docs = jd_loader.load()

    documents = cv_docs + jd_docs
    ner_path = "data/ner_extracted_profile.txt"
    if os.path.exists(ner_path):
        ner_loader = TextLoader(ner_path)
        ner_docs = ner_loader.load()
        documents += ner_docs
        print(f"  Loaded CV ({len(cv_docs)} pages) + JD + NER profile.")
    else:
        print(f"  Loaded CV ({len(cv_docs)} pages) + JD (no NER profile found).")

    print("3. Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"  {len(chunks)} chunks created.")

    print("4. Embedding and storing in FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vector_db.save_local("./faiss_db")
    print("Knowledge base built successfully.")


if __name__ == "__main__":
    build_knowledge_base()
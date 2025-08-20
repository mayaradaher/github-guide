from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader

# Load JSON
loader = JSONLoader(
    file_path="data/scraper/scraping_github_info.json",
    jq_schema=".[]",
    content_key="content",
)

docs = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_documents(docs)

# Create embeddings and store in FAISS
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store from the chunks and embeddings
vector_store = FAISS.from_documents(chunks, embeddings)

# Store vector
vector_store.save_local("data/processing/vector_store")

print("✓ Vector database created.")

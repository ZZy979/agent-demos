from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load a PDF file into a sequence of Document objects, one Document per page
file_path = 'nke-10k-2023.pdf'
loader = PyPDFLoader(file_path)

docs = loader.load()

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# Convert document texts into numeric vectors (embeddings)
# and store them in a data structure that supports similarity search
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)

# Construct retriever from vector store
# to retrieve documents based on similarity to a string query
retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 1},
)

# Invoke the retriever
result = retriever.batch([
    'How many distribution centers does Nike have in the US?',
    'When was Nike incorporated?',
    "What was Nike's revenue in 2023?",
    "How were Nike's margins impacted in 2023?",
])
print(result)

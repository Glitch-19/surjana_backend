from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
loader= PyPDFLoader("C:\\Users\\91885\\OneDrive\\Desktop\\surjana\\rag\\6science.pdf")
loader = TextLoader("ncert_class6_science.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
docs = text_splitter.split_documents(documents)
print(len(docs))
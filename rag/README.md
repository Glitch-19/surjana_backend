# RAG Pipeline for NCERT Science Textbook

This project implements a Retrieval-Augmented Generation (RAG) pipeline for the NCERT Class 6 Science textbook using LangChain, FAISS, and Google's Gemini API.

## Features

- **Text Processing**: Automatic chunking of the NCERT science textbook
- **Vector Embeddings**: Uses HuggingFace sentence-transformers for local embeddings
- **Vector Database**: FAISS for efficient similarity search
- **Question Answering**: Google Gemini LLM for generating answers
- **Fallback Mode**: Works with retrieval-only when LLM API is not available
- **Comprehensive Logging**: No print statements, uses logging throughout
- **Batch Processing**: Support for processing multiple questions

## Files Structure

```
rag/
├── requirements.txt              # Python dependencies
├── .env                         # Environment variables (API keys)
├── ncert_class6_science.txt     # Source textbook content
├── rag_pipeline_huggingface.py # Main pipeline with HuggingFace embeddings
├── query.py                     # Query interface with LLM integration
├── demo_rag.py                  # Complete demonstration script
├── simple_usage.py              # Simple usage example
├── README.md                    # This file
└── vector_store/                # Generated FAISS vector database
    ├── index.faiss
    └── index.pkl
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- langchain
- langchain-google-genai
- langchain-community
- faiss-cpu
- google-generativeai
- sentence-transformers
- python-dotenv

### 2. Environment Setup

Create a `.env` file with your Google API key:

```
GOOGLE_API_KEY=your_google_api_key_here
```

**Note**: If no Google API key is provided, the system will work in retrieval-only mode.

### 3. Build the Vector Database

Run the pipeline to create embeddings and vector database:

```bash
python rag_pipeline_huggingface.py
```

This will:
- Load and chunk the NCERT textbook
- Generate embeddings using HuggingFace models
- Create and save a FAISS vector store

## Usage

### Basic Query Example

```python
from query import RAGQueryEngine

# Initialize the query engine
query_engine = RAGQueryEngine("vector_store")

# Ask a question
result = query_engine.answer_query("What is photosynthesis?")

print(result['answer'])
print(f"Sources: {len(result['source_documents'])}")
```

### Running the Demo

```bash
python demo_rag.py
```

This will run the complete pipeline demonstration with sample questions.

### Simple Usage

```bash
python simple_usage.py
```

Shows a basic example of querying the system.

## API Reference

### RAGPipeline Class

```python
from rag_pipeline_huggingface import RAGPipeline

# Initialize
rag = RAGPipeline(model_name="all-MiniLM-L6-v2")

# Build complete pipeline
vector_store = rag.build_rag_pipeline(
    text_file_path="ncert_class6_science.txt",
    save_path="vector_store"
)
```

### RAGQueryEngine Class

```python
from query import RAGQueryEngine

# Initialize
query_engine = RAGQueryEngine(
    vector_store_path="vector_store",
    api_key="optional_google_api_key"
)

# Single query
result = query_engine.answer_query("Your question here")

# Batch queries
results = query_engine.batch_query(["Question 1", "Question 2"])

# Retrieve context only
context_docs = query_engine.retrieve_context("Your query", k=5)
```

## Response Format

Each query returns a dictionary with:

```python
{
    "question": "The original question",
    "answer": "Generated or retrieved answer",
    "source_documents": [
        {
            "content": "Document text content",
            "metadata": {"source": "file_info"}
        }
    ],
    "mode": "rag_with_llm" or "retrieval_only"
}
```

## Configuration Options

### Text Chunking
- `chunk_size`: 1000 characters
- `chunk_overlap`: 200 characters
- Uses `RecursiveCharacterTextSplitter`

### Embeddings
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Local processing (no API required)

### Vector Search
- Default retrieval: 5 similar documents
- FAISS for efficient similarity search

### LLM Configuration
- Model: `gemini-pro`
- Temperature: 0.3 (focused responses)
- Custom prompt template for educational content

## Logging

All components use comprehensive logging:
- `rag_pipeline.log`: Pipeline building logs
- `rag_query.log`: Query processing logs
- `rag_demo.log`: Demo script logs

No print statements are used throughout the codebase.

## Operating Modes

### 1. Full RAG Mode (with LLM)
- Retrieves relevant context
- Generates answers using Gemini LLM
- Provides educational, contextual responses

### 2. Retrieval-Only Mode
- Works when no API key is available
- Returns relevant text chunks
- Useful for finding specific information

## Example Questions

The system works well with questions like:
- "What is photosynthesis?"
- "How do plants make their food?"
- "What are the parts of a plant?"
- "Explain the water cycle"
- "What is the difference between living and non-living things?"

## Error Handling

- Graceful fallback to retrieval-only mode
- Comprehensive error logging
- Validation of file paths and API keys
- Exception handling throughout the pipeline

## Performance Notes

- Initial model download may take time (sentence-transformers)
- Vector database creation is one-time setup
- Query responses are typically fast (< 2 seconds)
- Memory usage depends on text corpus size

## Troubleshooting

### Common Issues

1. **"Vector store not found"**
   - Run `python rag_pipeline_huggingface.py` first

2. **"API key expired"**
   - Update `.env` file with valid Google API key
   - System will fallback to retrieval-only mode

3. **"Model download failed"**
   - Ensure internet connection for HuggingFace model download
   - Check available disk space

### Log Files
Check log files for detailed error information:
- `rag_pipeline.log`
- `rag_query.log`
- `rag_demo.log`

## Contributing

When contributing to this project:
1. Use logging instead of print statements
2. Follow the existing error handling patterns
3. Add comprehensive docstrings
4. Test both RAG and retrieval-only modes

## License

This project is for educational purposes using NCERT content.
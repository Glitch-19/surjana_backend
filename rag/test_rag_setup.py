"""
RAG System Test - Validates Setup Without API Calls
This script validates that all components are properly installed and configured
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'edu-game-backend', '.env'))

def test_imports():
    """Test that all required packages are importable"""
    print("Testing imports...")
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI - OK")
    except ImportError as e:
        print(f"❌ Google Generative AI - FAILED: {e}")
        return False
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
        print("✅ LangChain Community (PyPDFLoader) - OK")
    except ImportError as e:
        print(f"❌ LangChain Community - FAILED: {e}")
        return False
    
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        print("✅ LangChain Text Splitters - OK")
    except ImportError as e:
        print(f"❌ LangChain Text Splitters - FAILED: {e}")
        return False
    
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        print("✅ LangChain Google GenAI Embeddings - OK")
    except ImportError as e:
        print(f"❌ LangChain Google GenAI - FAILED: {e}")
        return False
    
    try:
        from langchain_chroma import Chroma
        print("✅ LangChain Chroma - OK")
    except ImportError as e:
        print(f"❌ LangChain Chroma - FAILED: {e}")
        return False
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("✅ LangChain Google GenAI Chat - OK")
    except ImportError as e:
        print(f"❌ LangChain Google GenAI Chat - FAILED: {e}")
        return False
    
    return True

def test_environment():
    """Test environment configuration"""
    print("\nTesting environment configuration...")
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        # Mask the key for security
        masked_key = google_api_key[:10] + "..." + google_api_key[-4:]
        print(f"✅ GOOGLE_API_KEY found: {masked_key}")
        return True
    else:
        print("❌ GOOGLE_API_KEY not found")
        return False

def test_pdf_file():
    """Test PDF file availability"""
    print("\nTesting PDF file...")
    
    pdf_path = os.path.join(os.path.dirname(__file__), "6science.pdf")
    if os.path.exists(pdf_path):
        file_size = os.path.getsize(pdf_path)
        print(f"✅ PDF file found: {pdf_path}")
        print(f"   File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        return True
    else:
        print(f"❌ PDF file not found: {pdf_path}")
        return False

def test_pdf_loading():
    """Test PDF loading without processing"""
    print("\nTesting PDF loading...")
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
        
        pdf_path = os.path.join(os.path.dirname(__file__), "6science.pdf")
        if not os.path.exists(pdf_path):
            print("❌ PDF file not available for testing")
            return False
        
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        print(f"✅ PDF loaded successfully")
        print(f"   Total pages: {len(documents)}")
        
        if documents:
            first_page_content = documents[0].page_content[:200]
            print(f"   First page preview: {first_page_content}...")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF loading failed: {e}")
        return False

def test_text_chunking():
    """Test text chunking without creating embeddings"""
    print("\nTesting text chunking...")
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        pdf_path = os.path.join(os.path.dirname(__file__), "6science.pdf")
        if not os.path.exists(pdf_path):
            print("❌ PDF file not available for testing")
            return False
        
        # Load first few pages only
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()[:3]  # First 3 pages only
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Split documents
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ Text chunking successful")
        print(f"   Input pages: {len(documents)}")
        print(f"   Output chunks: {len(chunks)}")
        
        if chunks:
            first_chunk = chunks[0].page_content[:200]
            print(f"   First chunk preview: {first_chunk}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Text chunking failed: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("RAG SYSTEM COMPONENT VALIDATION")
    print("="*60)
    
    all_tests_passed = True
    
    # Run all tests
    tests = [
        ("Package Imports", test_imports),
        ("Environment Config", test_environment),
        ("PDF File", test_pdf_file),
        ("PDF Loading", test_pdf_loading),
        ("Text Chunking", test_text_chunking)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'-'*20} {test_name} {'-'*20}")
        if not test_func():
            all_tests_passed = False
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nRAG System Components Status:")
        print("✅ All required packages installed")
        print("✅ Google API key configured") 
        print("✅ PDF file available")
        print("✅ PDF loading works")
        print("✅ Text chunking works")
        print("\n📝 Note: Actual embedding and vector storage require API quota")
        print("   The system is ready to work once API limits reset")
        
        print("\n🚀 Next Steps:")
        print("1. Wait for API quota to reset (24 hours)")
        print("2. Run: python gemini_rag_demo.py")
        print("3. Ask questions about NCERT Science content!")
        
    else:
        print("❌ SOME TESTS FAILED")
        print("Please fix the issues above before proceeding")

if __name__ == "__main__":
    main()
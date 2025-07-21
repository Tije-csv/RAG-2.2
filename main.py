from dotenv import load_dotenv
import os
from rag_pipeline import EnhancedRAGPipeline

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment variables
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
    
    # Initialize the RAG pipeline with the API key
    rag = EnhancedRAGPipeline(api_key)
    return rag

if __name__ == "__main__":
    rag = main()
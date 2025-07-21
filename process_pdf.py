from rag_pipeline import EnhancedRAGPipeline
from dotenv import load_dotenv
import os

def process_pdf():
    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    # Initialize RAG pipeline
    rag = EnhancedRAGPipeline(
        api_key=api_key,
        redis_host=os.getenv('REDIS_HOST', 'localhost'),
        redis_port=int(os.getenv('REDIS_PORT', '6379')),
        redis_password=os.getenv('REDIS_PASSWORD', '')
    )

    # Process the PDF
    pdf_path = "/home/ilytije/RAG/documents/Quantum_Innovations_IT_Security_Policy.pdf"
    rag.add_media_documents(file_paths=[pdf_path])
    print(f"Successfully processed {pdf_path}")

if __name__ == "__main__":
    process_pdf()

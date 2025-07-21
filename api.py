from flask import Flask, request, jsonify
from functools import wraps
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from rag_pipeline import EnhancedRAGPipeline
import os
from dotenv import load_dotenv
import datetime

# Load environment variables
load_dotenv()

# Configure Redis connection from environment variables
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
ADMIN_API_KEY = os.getenv('ADMIN_API_KEY')

if not ADMIN_API_KEY:
    raise ValueError("ADMIN_API_KEY not found in environment variables")

app = Flask(__name__)

# Initialize RAG pipeline with Redis configuration
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

rag = EnhancedRAGPipeline(
    api_key=api_key,
    redis_host=REDIS_HOST,
    redis_port=REDIS_PORT,
    redis_password=REDIS_PASSWORD
)

def verify_admin_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("X-Admin-Key")
        if api_key != ADMIN_API_KEY:
            return jsonify({"detail": "Could not validate admin credentials"}), 403
        return f(*args, **kwargs)
    return decorated_function

class Query(BaseModel):
    text: str

class Document(BaseModel):
    file_paths: Optional[List[str]] = None
    directory_path: Optional[str] = None

@app.route("/query", methods=["POST"])
def process_query():
    try:
        data = request.get_json()
        query = Query(**data)
        result = rag.process_query(query.text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@app.route("/admin/add-documents", methods=["POST"])
@verify_admin_key
def add_documents():
    try:
        data = request.get_json()
        documents = Document(**data)
        rag.add_media_documents(
            file_paths=documents.file_paths,
            directory_path=documents.directory_path
        )
        return jsonify({"message": "Documents added successfully"})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    try:
        # Check Redis connection
        redis_status = "healthy" if rag.redis_client.ping() else "unhealthy"

        # Get system metrics
        total_docs = len(rag.redis_client.keys("doc:*"))
        total_queries = len(rag.redis_client.keys("query:*"))

        return jsonify({
            "status": "healthy",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "services": {
                "redis": redis_status,
                "rag_pipeline": "healthy"
            },
            "metrics": {
                "total_documents": total_docs,
                "total_queries": total_queries
            }
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
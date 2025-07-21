from typing import List, Dict, Optional
import PyPDF2
from PIL import Image
import pytesseract
from docx import Document
from pathlib import Path
import logging

class MediaLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_pdf(self, file_path: str) -> str:
        """Extract text from PDF files."""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + '\n'
                return text
        except Exception as e:
            self.logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return ""

    def load_image(self, file_path: str) -> str:
        """Extract text from images using OCR."""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            self.logger.error(f"Error processing image {file_path}: {str(e)}")
            return ""

    def load_docx(self, file_path: str) -> str:
        """Extract text from Word documents."""
        try:
            doc = Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            self.logger.error(f"Error processing Word document {file_path}: {str(e)}")
            return ""

    def load_file(self, file_path: str) -> Optional[str]:
        """Load any supported file type and return its text content."""
        file_path = Path(file_path)
        
        handlers = {
            '.pdf': self.load_pdf,
            '.png': self.load_image,
            '.jpg': self.load_image,
            '.jpeg': self.load_image,
            '.docx': self.load_docx,
        }
        
        handler = handlers.get(file_path.suffix.lower())
        if handler:
            return handler(str(file_path))
        else:
            self.logger.warning(f"Unsupported file type: {file_path.suffix}")
            return None

    def load_directory(self, dir_path: str) -> List[Dict[str, str]]:
        """Load all supported files from a directory."""
        dir_path = Path(dir_path)
        documents = []
        
        for file_path in dir_path.glob('**/*'):
            if file_path.is_file():
                content = self.load_file(str(file_path))
                if content:
                    documents.append({
                        'content': content,
                        'source': str(file_path),
                        'type': file_path.suffix[1:]  # Remove the dot from extension
                    })
        
        return documents
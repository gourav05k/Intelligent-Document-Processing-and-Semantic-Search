# Intelligent Document Processing & Conversational AI

## Overview
End-to-end system for processing financial documents and enabling natural language queries using AI/ML techniques.

## Features
- **PDF Processing**: Both machine-readable and OCR for scanned documents
- **Structured Data Extraction**: Extract unit, rent, lease, and tenant information
- **Hybrid Storage**: PostgreSQL for structured data + Qdrant for semantic search
- **Conversational AI**: Natural language queries powered by LangChain and OpenAI
- **FastAPI Backend**: RESTful API with automatic documentation

## Architecture
- **Document Processing**: PyMuPDF, pdfplumber, Tesseract OCR
- **AI/ML**: LangChain, OpenAI GPT-4, text-embedding-ada-002
- **Databases**: PostgreSQL + Qdrant vector database
- **Web Framework**: FastAPI with async support
- **Deployment**: Docker Compose for easy setup

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd intelligent-document-processing-and-semantic-search
python setup_project.py  # Creates project structure
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Copy environment template
cp .env.example .env
# Edit .env with your API keys and database credentials
```

### 4. Start Databases
```bash
# Start PostgreSQL and Qdrant
docker-compose up -d

# Verify databases are running
docker-compose ps
```

### 5. Run Application
```bash
# Start FastAPI server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Access API documentation
# http://localhost:8000/docs
```

## Usage

### Upload Documents
```bash
curl -X POST "http://localhost:8000/upload-document/" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@financial_document.pdf" \
     -F "property_name=Downtown Complex"
```

### Query Data
```bash
curl -X POST "http://localhost:8000/query/" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the total rent for all units?"}'
```

### Example Queries
- "What is the total square feet for the property?"
- "How many units are occupied vs vacant?"
- "Find lease agreements with pet policies"
- "What's the average rent for 2-bedroom units?"

## Project Structure
```
intelligent-document-processing-and-semantic-search/
├── README.md
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── setup_project.py
├── src/
│   ├── __init__.py
│   ├── document_parser.py      # Document processing pipeline
│   ├── storage_manager.py      # Database operations
│   ├── query_interface.py      # Conversational AI interface
│   ├── main.py                # FastAPI application
│   ├── models/                # Data models
│   │   ├── __init__.py
│   │   ├── schemas.py         # Pydantic models
│   │   └── database.py        # SQLAlchemy models
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── ocr_utils.py       # OCR helper functions
│   │   ├── text_processing.py # Text processing utilities
│   │   └── vector_utils.py    # Vector operations
│   └── config/
│       ├── __init__.py
│       └── settings.py        # Configuration management
├── data/                      # Sample documents
├── tests/                     # Test files
├── docs/                      # Documentation
├── migrations/                # Database migrations
└── scripts/                   # Utility scripts
```

## Development

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Tesseract OCR (for scanned documents)

### Installation Notes
- **Tesseract OCR**: 
  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
  - macOS: `brew install tesseract`
  - Ubuntu: `sudo apt-get install tesseract-ocr`

### Testing
```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## API Documentation
Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License
MIT License - see LICENSE file for details

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
import tempfile
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import uvicorn
import asyncio
from datetime import datetime

from src.document_parser import DocumentParser
from src.storage_manager import StorageManager
from src.query_interface import QueryInterface
from src.models.schemas import QueryRequest, QueryResponse, DocumentUploadResponse, HealthResponse, ExampleQueries
from src.config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Document Processing & Conversational AI",
    description="End-to-end system for processing financial documents and enabling natural language queries using AI/ML techniques",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global app components
document_parser = None
storage_manager = None
query_interface = None

# Initialize components
@app.on_event("startup")
async def startup_event():
    """Initialize application components on startup"""
    global document_parser, storage_manager, query_interface
    
    try:
        logger.info("Initializing application components...")
        
        # Initialize components
        document_parser = DocumentParser()
        storage_manager = StorageManager()
        query_interface = QueryInterface()
        
        # Health check
        health = storage_manager.health_check()
        if not all(health.values()):
            logger.warning(f"Database health check issues: {health}")
        else:
            logger.info("All databases are healthy")
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

# Dependency functions
def get_document_parser():
    if document_parser is None:
        raise HTTPException(status_code=500, detail="Document parser not initialized")
    return document_parser

def get_storage_manager():
    if storage_manager is None:
        raise HTTPException(status_code=500, detail="Storage manager not initialized")
    return storage_manager

def get_query_interface():
    if query_interface is None:
        raise HTTPException(status_code=500, detail="Query interface not initialized")
    return query_interface

# API Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with system information and navigation"""
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Financial Document Processor</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                .feature { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .api-link { display: inline-block; background: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 5px; }
                .api-link:hover { background: #2980b9; }
                .status { color: #27ae60; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏢 Intelligent Document Processing & Conversational AI</h1>
                <p><span class="status">✅ System Online</span> - Ready to process financial documents and answer queries</p>
                
                <div class="feature">
                    <h3>📄 Document Processing</h3>
                    <p>Upload financial PDFs (machine-readable or scanned) to extract unit data, rent information, and lease details.</p>
                </div>
                
                <div class="feature">
                    <h3>💬 Conversational Queries</h3>
                    <p>Ask natural language questions about your data: "What's the total rent?", "How many units are occupied?", "Find pet policies"</p>
                </div>
                
                <div class="feature">
                    <h3>🔍 Hybrid Search</h3>
                    <p>Combines structured database queries with semantic document search for comprehensive answers.</p>
                </div>
                
                <h3>🚀 API Documentation</h3>
                <a href="/docs" class="api-link">📚 Interactive API Docs (Swagger)</a>
                <a href="/redoc" class="api-link">📖 Alternative Docs (ReDoc)</a>
                <a href="/health" class="api-link">🏥 System Health</a>
                <a href="/example-queries" class="api-link">💡 Example Queries</a>
                
                <h3>📊 Quick Stats</h3>
                <a href="/statistics" class="api-link">📈 View Statistics</a>
                <a href="/documents" class="api-link">📋 List Documents</a>
            </div>
        </body>
    </html>
    """

@app.post("/upload-document/", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    property_name: str = "Default Property",
    parser: DocumentParser = Depends(get_document_parser),
    storage: StorageManager = Depends(get_storage_manager)
):
    """Upload and process a financial document (PDF)"""
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Check file size
    if hasattr(file, 'size') and file.size and file.size > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File size exceeds {settings.max_file_size_mb}MB limit")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Validate PDF
        if not parser.validate_pdf_file(temp_path):
            os.unlink(temp_path)
            raise HTTPException(status_code=400, detail="Invalid PDF file")
        
        # Process document in background
        background_tasks.add_task(
            process_document_background,
            temp_path, file.filename, property_name, parser, storage
        )
        
        logger.info(f"Document upload initiated: {file.filename}")
        
        return DocumentUploadResponse(
            message="Document uploaded successfully and processing started",
            filename=file.filename,
            property_name=property_name,
            status="processing"
        )
    
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing document upload: {str(e)}")

async def process_document_background(
    temp_path: str,
    filename: str,
    property_name: str,
    parser: DocumentParser,
    storage: StorageManager
):
    """Background task to process uploaded document"""
    try:
        logger.info(f"Starting background processing for: {filename}")
        
        # Process document
        text, structured_data = parser.process_document(temp_path)
        
        if not structured_data:
            logger.warning(f"No structured data extracted from {filename}")
            return
        
        # Create property
        property_obj = storage.create_property(
            property_name=property_name,
            total_units=len(structured_data)
        )
        
        # Create document record
        doc_record = parser.create_document_record(filename, text)
        doc_obj = storage.create_document(doc_record)
        
        # Create unit records
        unit_records = parser.create_unit_records(structured_data, property_obj.id)
        unit_objects = storage.create_units(unit_records)
        
        # Create lease records
        unit_mapping = {unit.unit_number: unit.id for unit in unit_objects}
        lease_records = parser.create_lease_records(structured_data, unit_mapping)
        if lease_records:
            storage.create_leases(lease_records)
        
        # Store embeddings for semantic search
        metadata = {
            "filename": filename,
            "property_id": property_obj.id,
            "property_name": property_name,
            "document_type": "financial_pdf",
            "total_units": len(structured_data)
        }
        
        storage.store_document_embeddings(
            document_id=doc_obj.id,
            text=text,
            metadata=metadata
        )
        
        logger.info(f"Successfully processed document: {filename} - {len(structured_data)} units, {len(lease_records)} leases")
        
    except Exception as e:
        logger.error(f"Error processing document {filename}: {e}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/query/", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    query_interface: QueryInterface = Depends(get_query_interface)
):
    """Process a natural language query about the documents"""
    try:
        logger.info(f"Processing query: {request.query}")
        response = query_interface.process_query(request)
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/statistics/", response_model=Dict[str, Any])
async def get_statistics(
    storage: StorageManager = Depends(get_storage_manager)
):
    """Get comprehensive system statistics"""
    try:
        occupancy_stats = storage.get_occupancy_stats()
        total_rent = storage.get_total_rent()
        total_sqft = storage.get_total_square_feet()
        avg_rent = storage.get_average_rent()
        
        # Calculate additional metrics
        occupancy_rate = (occupancy_stats["occupied"] / max(occupancy_stats["total"], 1)) * 100
        rent_per_sqft = (total_rent / max(total_sqft, 1)) if total_sqft > 0 else 0
        
        return {
            "occupancy": occupancy_stats,
            "financial_metrics": {
                "total_rent": total_rent,
                "average_rent": avg_rent,
                "total_square_feet": total_sqft,
                "rent_per_sqft": rent_per_sqft
            },
            "calculated_metrics": {
                "occupancy_rate": round(occupancy_rate, 1),
                "vacant_rate": round(100 - occupancy_rate, 1),
                "revenue_potential": total_rent * 12  # Annual revenue
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")

@app.get("/health/", response_model=HealthResponse)
async def health_check(
    storage: StorageManager = Depends(get_storage_manager)
):
    """Comprehensive health check endpoint"""
    try:
        # Check database health
        db_health = storage.health_check()
        
        # Overall system health
        system_healthy = all(db_health.values())
        
        if system_healthy:
            return HealthResponse(
                status="healthy",
                message="All systems operational"
            )
        else:
            return HealthResponse(
                status="degraded", 
                message=f"Database issues detected: {db_health}"
            )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}"
        )

@app.get("/documents/", response_model=List[Dict[str, Any]])
async def list_documents(
    storage: StorageManager = Depends(get_storage_manager)
):
    """List all processed documents with metadata"""
    try:
        with storage.get_db_session() as db:
            from src.models.database import Document, Property
            
            # Get documents with property information
            documents = db.query(Document).all()
            properties = db.query(Property).all()
            
            # Create property lookup
            property_lookup = {prop.id: prop.property_name for prop in properties}
            
            result = []
            for doc in documents:
                # Try to find associated property (basic heuristic)
                associated_property = "Unknown"
                for prop_id, prop_name in property_lookup.items():
                    # This is a simple association - in production you'd have a proper relationship
                    associated_property = prop_name
                    break
                
                result.append({
                    "id": doc.id,
                    "filename": doc.filename,
                    "document_type": doc.document_type,
                    "processed_at": doc.processed_at.isoformat(),
                    "associated_property": associated_property,
                    "content_length": len(doc.content_text) if doc.content_text else 0
                })
            
            return result
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@app.get("/example-queries/", response_model=ExampleQueries)
async def get_example_queries(
    query_interface: QueryInterface = Depends(get_query_interface)
):
    """Get example queries tailored to the financial document format"""
    try:
        examples = query_interface.get_example_queries()
        return ExampleQueries(**examples)
    
    except Exception as e:
        logger.error(f"Error getting example queries: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving examples: {str(e)}")

@app.get("/metadata-schema/")
async def get_metadata_schema(
    storage: StorageManager = Depends(get_storage_manager)
):
    """Get the current metadata schema for vector search"""
    try:
        schema = storage.get_metadata_schema()
        return {
            "description": "Metadata schema for vector search and filtering",
            "schemas": schema,
            "usage": "Use these fields for filtering in semantic search queries"
        }
    
    except Exception as e:
        logger.error(f"Error getting metadata schema: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving schema: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": [
                "/docs - API documentation",
                "/health - System health check", 
                "/upload-document/ - Upload PDF documents",
                "/query/ - Process natural language queries",
                "/statistics/ - Get system statistics"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now().isoformat()
        }
    )

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.embeddings import OpenAIEmbeddings
from typing import List, Dict, Optional, Any
import uuid
import logging
from datetime import datetime

from src.config.settings import settings
from src.models.database import Base, Property, Unit, Lease, Document
from src.models.schemas import *

logger = logging.getLogger(__name__)


class StorageManager:
    def __init__(self):
        # PostgreSQL setup
        self.engine = create_engine(settings.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Qdrant setup
        self.qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key
        )
        
        # Embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
        
        # Metadata schema definitions
        self.core_metadata_schema = {
            "document_id": "integer",
            "document_type": "keyword",
            "property_name": "keyword", 
            "processed_date": "keyword",
            "file_name": "keyword",
            "page_number": "integer"
        }
        
        # Financial document specific metadata (based on your PDF samples)
        self.financial_metadata_schema = {
            "unit_number": "keyword",      # 01-101, 01-102
            "unit_type": "keyword",        # MBL2AC60, 2BR, 3BR
            "tenant_name": "keyword",      # Simon Marie, Pottinger Margaret
            "rent_amount": "float",        # 1511.00, 1306.00
            "occupancy_status": "keyword", # occupied, vacant
            "lease_start": "keyword",      # date as string
            "lease_end": "keyword",        # date as string
            "property_section": "keyword", # Building A, Building B
            "content_type": "keyword"      # unit_listing, lease_terms, policies
        }
        
        # Initialize databases
        self._init_databases()
    
    def _init_databases(self):
        """Initialize database schemas and collections"""
        try:
            # Create PostgreSQL tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("PostgreSQL tables created/verified")
            
            # Create Qdrant collection with hybrid metadata schema
            collection_name = "financial_documents"
            
            try:
                self.qdrant_client.get_collection(collection_name)
                logger.info(f"Qdrant collection '{collection_name}' already exists")
            except:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection '{collection_name}' with hybrid metadata schema")
        
        except Exception as e:
            logger.error(f"Error initializing databases: {e}")
            raise
    
    def get_db_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    # PostgreSQL Operations
    def create_property(self, property_name: str, total_units: int = None) -> Property:
        """Create a new property"""
        try:
            with self.get_db_session() as db:
                property_obj = Property(
                    property_name=property_name,
                    total_units=total_units
                )
                db.add(property_obj)
                db.commit()
                db.refresh(property_obj)
                logger.info(f"Created property: {property_name} with {total_units} units")
                return property_obj
        except Exception as e:
            logger.error(f"Error creating property: {e}")
            raise
    
    def create_units(self, units: List[UnitCreate]) -> List[Unit]:
        """Create multiple units"""
        try:
            with self.get_db_session() as db:
                unit_objects = []
                for unit_data in units:
                    unit_obj = Unit(**unit_data.dict())
                    db.add(unit_obj)
                    unit_objects.append(unit_obj)
                
                db.commit()
                for unit_obj in unit_objects:
                    db.refresh(unit_obj)
                
                logger.info(f"Created {len(unit_objects)} units")
                return unit_objects
        except Exception as e:
            logger.error(f"Error creating units: {e}")
            raise
    
    def create_leases(self, leases: List[LeaseCreate]) -> List[Lease]:
        """Create multiple leases"""
        try:
            with self.get_db_session() as db:
                lease_objects = []
                for lease_data in leases:
                    lease_obj = Lease(**lease_data.dict())
                    db.add(lease_obj)
                    lease_objects.append(lease_obj)
                
                db.commit()
                for lease_obj in lease_objects:
                    db.refresh(lease_obj)
                
                logger.info(f"Created {len(lease_objects)} leases")
                return lease_objects
        except Exception as e:
            logger.error(f"Error creating leases: {e}")
            raise
    
    def create_document(self, document: DocumentCreate) -> Document:
        """Create document record"""
        try:
            with self.get_db_session() as db:
                doc_obj = Document(**document.dict())
                db.add(doc_obj)
                db.commit()
                db.refresh(doc_obj)
                logger.info(f"Created document record: {document.filename}")
                return doc_obj
        except Exception as e:
            logger.error(f"Error creating document: {e}")
            raise
    
    # Query Operations with Filters
    def get_total_rent(self, filters: Dict[str, Any] = None) -> float:
        """Get total rent for all units with optional filters"""
        try:
            with self.get_db_session() as db:
                query = "SELECT SUM(rent_amount) FROM units WHERE rent_amount IS NOT NULL"
                params = {}
                
                if filters:
                    if filters.get('unit_type'):
                        query += " AND unit_type = :unit_type"
                        params['unit_type'] = filters['unit_type']
                    if filters.get('property_id'):
                        query += " AND property_id = :property_id"
                        params['property_id'] = filters['property_id']
                    if filters.get('status'):
                        query += " AND status = :status"
                        params['status'] = filters['status']
                
                result = db.execute(text(query), params)
                total = float(result.scalar() or 0)
                logger.info(f"Total rent calculated: ${total:,.2f} with filters: {filters}")
                return total
        except Exception as e:
            logger.error(f"Error calculating total rent: {e}")
            return 0.0
    
    def get_total_square_feet(self, filters: Dict[str, Any] = None) -> int:
        """Get total square feet for all units with optional filters"""
        try:
            with self.get_db_session() as db:
                query = "SELECT SUM(area_sqft) FROM units WHERE area_sqft IS NOT NULL"
                params = {}
                
                if filters:
                    if filters.get('unit_type'):
                        query += " AND unit_type = :unit_type"
                        params['unit_type'] = filters['unit_type']
                    if filters.get('property_id'):
                        query += " AND property_id = :property_id"
                        params['property_id'] = filters['property_id']
                
                result = db.execute(text(query), params)
                total = int(result.scalar() or 0)
                logger.info(f"Total square feet calculated: {total:,} with filters: {filters}")
                return total
        except Exception as e:
            logger.error(f"Error calculating total square feet: {e}")
            return 0
    
    def get_occupancy_stats(self, filters: Dict[str, Any] = None) -> Dict[str, int]:
        """Get occupancy statistics with optional filters"""
        try:
            with self.get_db_session() as db:
                base_query = "SELECT COUNT(*) FROM units"
                params = {}
                
                # Build WHERE clause for filters
                where_conditions = []
                if filters:
                    if filters.get('unit_type'):
                        where_conditions.append("unit_type = :unit_type")
                        params['unit_type'] = filters['unit_type']
                    if filters.get('property_id'):
                        where_conditions.append("property_id = :property_id")
                        params['property_id'] = filters['property_id']
                
                where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
                
                # Get occupied count
                occupied_query = base_query + where_clause + (" AND " if where_conditions else " WHERE ") + "status = 'occupied'"
                occupied = int(db.execute(text(occupied_query), params).scalar() or 0)
                
                # Get vacant count
                vacant_query = base_query + where_clause + (" AND " if where_conditions else " WHERE ") + "status = 'vacant'"
                vacant = int(db.execute(text(vacant_query), params).scalar() or 0)
                
                # Get total count
                total_query = base_query + where_clause
                total = int(db.execute(text(total_query), params).scalar() or 0)
                
                stats = {
                    "occupied": occupied,
                    "vacant": vacant,
                    "total": total
                }
                
                logger.info(f"Occupancy stats: {stats} with filters: {filters}")
                return stats
        except Exception as e:
            logger.error(f"Error calculating occupancy stats: {e}")
            return {"occupied": 0, "vacant": 0, "total": 0}
    
    def get_average_rent(self, filters: Dict[str, Any] = None) -> float:
        """Get average rent with optional filters"""
        try:
            with self.get_db_session() as db:
                query = "SELECT AVG(rent_amount) FROM units WHERE rent_amount IS NOT NULL"
                params = {}
                
                if filters:
                    if filters.get('unit_type'):
                        query += " AND unit_type = :unit_type"
                        params['unit_type'] = filters['unit_type']
                    if filters.get('property_id'):
                        query += " AND property_id = :property_id"
                        params['property_id'] = filters['property_id']
                
                result = db.execute(text(query), params)
                avg = float(result.scalar() or 0)
                logger.info(f"Average rent calculated: ${avg:,.2f} with filters: {filters}")
                return avg
        except Exception as e:
            logger.error(f"Error calculating average rent: {e}")
            return 0.0
    
    # Vector Operations with Dynamic Metadata
    def store_document_embeddings(self, document_id: int, text: str, metadata: Dict[str, Any]):
        """Store document embeddings in Qdrant with dynamic metadata"""
        try:
            # Split text into chunks
            chunks = self._split_text(text)
            
            points = []
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embeddings.embed_query(chunk)
                
                # Build comprehensive metadata combining core + financial + custom
                point_metadata = {
                    # Core metadata (always present)
                    "document_id": document_id,
                    "chunk_index": i,
                    "content": chunk,
                    "processed_date": datetime.now().isoformat(),
                    
                    # Add provided metadata (can include financial fields)
                    **metadata
                }
                
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=point_metadata
                )
                points.append(point)
            
            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name="financial_documents",
                points=points
            )
            
            logger.info(f"Stored {len(points)} embeddings for document {document_id} with metadata keys: {list(metadata.keys())}")
        
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise
    
    def search_similar_documents(self, query: str, filters: Dict[str, Any] = None, limit: int = 5) -> List[Dict]:
        """Search for similar documents using vector similarity with optional metadata filters"""
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            # Build Qdrant filter from provided filters
            qdrant_filter = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if key in {**self.core_metadata_schema, **self.financial_metadata_schema}:
                        filter_conditions.append({
                            "key": key,
                            "match": {"value": value}
                        })
                
                if filter_conditions:
                    qdrant_filter = {"must": filter_conditions}
            
            search_results = self.qdrant_client.search(
                collection_name="financial_documents",
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True
            )
            
            results = []
            for result in search_results:
                results.append({
                    "content": result.payload.get("content", ""),
                    "score": result.score,
                    "document_id": result.payload.get("document_id"),
                    "metadata": result.payload,
                    "unit_number": result.payload.get("unit_number"),
                    "tenant_name": result.payload.get("tenant_name"),
                    "rent_amount": result.payload.get("rent_amount")
                })
            
            logger.info(f"Found {len(results)} similar documents for query: '{query}' with filters: {filters}")
            return results
        
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def _split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.7:  # If we find a period in the last 30%
                    end = start + last_period + 1
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def get_metadata_schema(self) -> Dict[str, Dict[str, str]]:
        """Get the current metadata schema for reference"""
        return {
            "core_metadata": self.core_metadata_schema,
            "financial_metadata": self.financial_metadata_schema
        }
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of both databases"""
        health = {"postgresql": False, "qdrant": False}
        
        try:
            # Test PostgreSQL
            with self.get_db_session() as db:
                db.execute(text("SELECT 1"))
            health["postgresql"] = True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
        
        try:
            # Test Qdrant
            self.qdrant_client.get_collections()
            health["qdrant"] = True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
        
        return health

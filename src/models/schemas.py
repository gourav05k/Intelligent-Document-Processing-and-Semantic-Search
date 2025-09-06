from pydantic import BaseModel, Field
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from decimal import Decimal


class PropertyBase(BaseModel):
    property_name: str = Field(..., description="Name of the property")
    total_units: Optional[int] = Field(None, description="Total number of units")


class PropertyCreate(PropertyBase):
    pass


class Property(PropertyBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class UnitBase(BaseModel):
    unit_number: str = Field(..., description="Unit identifier")
    unit_type: Optional[str] = Field(None, description="Type of unit (e.g., 1BR, 2BR)")
    area_sqft: Optional[int] = Field(None, description="Area in square feet")
    rent_amount: Optional[Decimal] = Field(None, description="Monthly rent amount")
    status: str = Field(default="vacant", description="Unit status")


class UnitCreate(UnitBase):
    property_id: int


class Unit(UnitBase):
    id: int
    property_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class LeaseBase(BaseModel):
    tenant_name: str = Field(..., description="Tenant full name")
    lease_start: Optional[date] = Field(None, description="Lease start date")
    lease_end: Optional[date] = Field(None, description="Lease end date")
    move_in_date: Optional[date] = Field(None, description="Move in date")
    move_out_date: Optional[date] = Field(None, description="Move out date")
    total_amount: Optional[Decimal] = Field(None, description="Total lease amount")
    status: str = Field(default="active", description="Lease status")


class LeaseCreate(LeaseBase):
    unit_id: int


class Lease(LeaseBase):
    id: int
    unit_id: int
    
    class Config:
        from_attributes = True


class DocumentBase(BaseModel):
    filename: str = Field(..., description="Document filename")
    document_type: str = Field(..., description="Type of document")
    content_text: Optional[str] = Field(None, description="Extracted text content")


class DocumentCreate(DocumentBase):
    pass


class Document(DocumentBase):
    id: int
    processed_at: datetime
    
    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    include_context: bool = Field(default=True, description="Include document context")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional filters")
    limit: int = Field(default=10, description="Maximum number of results")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(default=[], description="Source references")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    query_type: str = Field(..., description="Type of query processed")
    filters_applied: Optional[Dict[str, Any]] = Field(default=None, description="Applied filters")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    execution_time: Optional[float] = Field(default=None, description="Query execution time in seconds")


class PropertyStats(BaseModel):
    total_units: int
    occupied_units: int
    vacant_units: int
    total_rent: float
    total_square_feet: int
    occupancy_rate: float


class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    property_name: str
    status: str


class HealthResponse(BaseModel):
    status: str
    message: str


class ExampleQueries(BaseModel):
    structured_queries: List[str]
    semantic_queries: List[str]
    hybrid_queries: List[str]

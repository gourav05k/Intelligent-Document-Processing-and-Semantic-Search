from sqlalchemy import Column, Integer, String, Text, DECIMAL, Date, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Property(Base):
    __tablename__ = "properties"
    
    id = Column(Integer, primary_key=True, index=True)
    property_name = Column(String(255), nullable=False)
    total_units = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    units = relationship("Unit", back_populates="property")


class Unit(Base):
    __tablename__ = "units"
    
    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, ForeignKey("properties.id"))
    unit_number = Column(String(20), nullable=False)
    unit_type = Column(String(50))
    area_sqft = Column(Integer)
    rent_amount = Column(DECIMAL(10, 2))
    status = Column(String(20), default="vacant")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    property = relationship("Property", back_populates="units")
    leases = relationship("Lease", back_populates="unit")


class Lease(Base):
    __tablename__ = "leases"
    
    id = Column(Integer, primary_key=True, index=True)
    unit_id = Column(Integer, ForeignKey("units.id"))
    tenant_name = Column(String(255), nullable=False)
    lease_start = Column(Date)
    lease_end = Column(Date)
    move_in_date = Column(Date)
    move_out_date = Column(Date)
    total_amount = Column(DECIMAL(10, 2))
    status = Column(String(20), default="active")
    
    # Relationships
    unit = relationship("Unit", back_populates="leases")


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    document_type = Column(String(50))
    content_text = Column(Text)
    processed_at = Column(DateTime(timezone=True), server_default=func.now())

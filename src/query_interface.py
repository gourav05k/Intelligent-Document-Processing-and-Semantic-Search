from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseOutputParser
from typing import Dict, List, Optional, Any
import logging
import json
import re
import time

from src.storage_manager import StorageManager
from src.models.schemas import QueryRequest, QueryResponse
from src.config.settings import settings

logger = logging.getLogger(__name__)


class QueryInterface:
    def __init__(self):
        self.storage_manager = StorageManager()
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        
        self.system_prompt = self._create_system_prompt()
        logger.info("QueryInterface initialized with LangChain and OpenAI")
    
    def _create_system_prompt(self) -> str:
        """Create enhanced system prompt with examples based on your PDF format"""
        return """You are a helpful assistant for a financial document processing system specializing in rental property management.
        You help users query information about rental properties, units, leases, and tenants.
        
        You have access to:
        1. Structured data about properties, units, and leases in a PostgreSQL database
        2. Document content and semantic search capabilities through a vector database
        
        DOCUMENT FORMAT CONTEXT:
        The system processes financial documents with the following structure:
        - Unit numbers: 01-101, 01-102, 01-103, etc.
        - Unit types: MBL2AC60, MBL3AC60 (which correspond to 2BR, 3BR apartments)
        - Rent amounts: $1,511.00, $1,306.00, etc.
        - Tenant names: Simon Marie, Pottinger Margaret, etc.
        - Occupancy status: Occupied, Vacant
        - Lease dates and terms
        
        EXAMPLE INTERACTIONS:
        
        User: "What is the total rent for 2-bedroom units?"
        Assistant: "The total rent for 2BR units is $4,250.00 across 3 units. This includes Unit 01-101 ($1,500), Unit 01-203 ($1,400), and Unit 02-105 ($1,350)."
        
        User: "How many units are occupied in the property?"
        Assistant: "Currently, 8 out of 12 units are occupied (66.7% occupancy rate). 4 units are vacant and available for rent."
        
        User: "Find lease agreements with pet policies"
        Assistant: "I found 3 documents mentioning pet policies: [Document references with specific clauses about pet deposits, restrictions, and fees]"
        
        User: "What's the average rent per square foot?"
        Assistant: "The average rent per square foot is $2.15. This is calculated from 15 units with a total of 18,500 sq ft and $39,775 in monthly rent."
        
        User: "Show me information about unit 01-101"
        Assistant: "Unit 01-101 is a 2BR apartment (MBL2AC60) with $1,511 monthly rent. It is currently occupied by Simon Marie with lease running from 9/1/2024 to 8/31/2025."
        
        RESPONSE GUIDELINES:
        - Be precise and factual with numbers
        - Include specific unit numbers and amounts when relevant
        - Calculate percentages and ratios when helpful
        - Cite document sources for policy-related queries
        - If information is incomplete, clearly state what's missing
        - For ambiguous queries, ask for clarification with specific options
        - Always format currency as $X,XXX.XX
        - Use clear, professional language
        
        AVAILABLE DATA FIELDS:
        - Unit numbers (01-101, 01-102, etc.)
        - Unit types (MBL2AC60=2BR, MBL3AC60=3BR, etc.)
        - Square footage and rent amounts
        - Tenant names and lease dates
        - Occupancy status (occupied/vacant)
        - Document content and policies
        """
    
    def extract_query_metadata(self, query: str) -> Dict[str, Any]:
        """Extract metadata and filters from user query"""
        query_lower = query.lower()
        metadata = {
            "fields": [],
            "filters": {},
            "aggregations": [],
            "entities": []
        }
        
        # Field extraction patterns based on your PDF format
        field_patterns = {
            "unit_number": r"unit\s*(?:number|#)?\s*(\d{2}-\d{3}|\d{1,3}[A-Z]?\d{0,3})",
            "unit_type": r"(\d+)\s*(?:bedroom|br|bed)|mbl(\d+)ac\d+",
            "tenant_name": r"tenant\s+([a-z\s]+)",
            "rent": r"rent|rental|payment|cost",
            "area": r"square\s*feet|sqft|area|size",
            "lease_dates": r"lease\s*(?:start|end|term|period|date)",
            "occupancy": r"occupied|vacant|empty|available"
        }
        
        # Extract specific fields mentioned
        for field, pattern in field_patterns.items():
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches or field.split('_')[0] in query_lower:
                metadata["fields"].append(field)
                if matches:
                    # Handle different match types
                    if field == "unit_type" and matches:
                        # Convert bedroom numbers to unit types
                        for match in matches:
                            if isinstance(match, tuple):
                                # From "2 bedroom" pattern
                                if match[0]:
                                    metadata["filters"]["unit_type"] = f"{match[0]}BR"
                                # From "MBL2AC60" pattern  
                                elif match[1]:
                                    metadata["filters"]["unit_type"] = f"{match[1]}BR"
                            else:
                                metadata["filters"]["unit_type"] = f"{match}BR"
                    else:
                        metadata["filters"][field] = matches[0] if len(matches) == 1 else matches
        
        # Extract aggregation operations
        aggregation_patterns = {
            "sum": r"total|sum",
            "count": r"how many|count|number of",
            "avg": r"average|mean",
            "max": r"maximum|highest|max",
            "min": r"minimum|lowest|min"
        }
        
        for agg_type, pattern in aggregation_patterns.items():
            if re.search(pattern, query_lower):
                metadata["aggregations"].append(agg_type)
        
        # Extract specific entities (unit numbers, tenant names, etc.)
        unit_matches = re.findall(r"(\d{2}-\d{3})", query_lower)
        if unit_matches:
            metadata["entities"].extend([f"unit_{unit}" for unit in unit_matches])
            metadata["filters"]["unit_number"] = unit_matches[0] if len(unit_matches) == 1 else unit_matches
        
        return metadata
    
    def classify_query_type(self, query: str) -> str:
        """Enhanced query classification with metadata awareness"""
        metadata = self.extract_query_metadata(query)
        query_lower = query.lower()
        
        # If specific aggregations or structured fields are mentioned
        if metadata["aggregations"] or any(field in ["rent", "area", "occupancy"] for field in metadata["fields"]):
            return "structured"
        
        # If document-related or semantic terms
        semantic_keywords = [
            'find documents', 'show me', 'lease agreement', 'policy', 'terms',
            'similar to', 'about', 'regarding', 'maintenance', 'pet', 'contract',
            'rules', 'policies', 'conditions', 'clauses'
        ]
        
        if any(keyword in query_lower for keyword in semantic_keywords):
            return "semantic"
        
        # Specific unit queries (hybrid - need both structured data and context)
        if metadata.get("entities") or "unit" in query_lower:
            return "hybrid"
        
        # Default to structured for financial queries
        financial_terms = ['rent', 'cost', 'price', 'occupied', 'vacant', 'tenant', 'lease']
        if any(term in query_lower for term in financial_terms):
            return "structured"
        
        # Default to semantic for everything else
        return "semantic"
    
    def process_structured_query(self, query: str) -> Dict[str, Any]:
        """Enhanced structured query processing with metadata filters"""
        start_time = time.time()
        metadata = self.extract_query_metadata(query)
        query_lower = query.lower()
        
        # Build filters from extracted metadata
        filters = {}
        if "unit_number" in metadata["filters"]:
            filters["unit_number"] = metadata["filters"]["unit_number"]
        if "unit_type" in metadata["filters"]:
            filters["unit_type"] = metadata["filters"]["unit_type"]
        
        try:
            # Process different query types with filters
            if 'total rent' in query_lower or 'sum' in metadata["aggregations"]:
                total_rent = self.storage_manager.get_total_rent(filters=filters)
                filter_desc = self._build_filter_description(filters)
                return {
                    "type": "structured",
                    "result": f"The total rent{filter_desc} is ${total_rent:,.2f}",
                    "value": total_rent,
                    "filters_applied": filters,
                    "execution_time": time.time() - start_time
                }
            
            elif 'total square feet' in query_lower or 'total area' in query_lower:
                total_sqft = self.storage_manager.get_total_square_feet(filters=filters)
                filter_desc = self._build_filter_description(filters)
                return {
                    "type": "structured", 
                    "result": f"The total square footage{filter_desc} is {total_sqft:,} sq ft",
                    "value": total_sqft,
                    "filters_applied": filters,
                    "execution_time": time.time() - start_time
                }
            
            elif any(keyword in query_lower for keyword in ['occupied', 'vacant', 'how many units', 'occupancy']):
                stats = self.storage_manager.get_occupancy_stats(filters=filters)
                filter_desc = self._build_filter_description(filters)
                occupancy_rate = (stats['occupied'] / max(stats['total'], 1)) * 100
                return {
                    "type": "structured",
                    "result": f"Out of {stats['total']} total units{filter_desc}: {stats['occupied']} are occupied ({occupancy_rate:.1f}%) and {stats['vacant']} are vacant",
                    "value": stats,
                    "filters_applied": filters,
                    "execution_time": time.time() - start_time
                }
            
            elif 'average rent' in query_lower or 'avg' in metadata["aggregations"]:
                avg_rent = self.storage_manager.get_average_rent(filters=filters)
                filter_desc = self._build_filter_description(filters)
                return {
                    "type": "structured",
                    "result": f"The average rent{filter_desc} is ${avg_rent:,.2f}",
                    "value": avg_rent,
                    "filters_applied": filters,
                    "execution_time": time.time() - start_time
                }
            
            else:
                return {
                    "type": "structured",
                    "result": "I couldn't find specific structured data for that query. Could you please be more specific about what financial information you're looking for?",
                    "value": None,
                    "metadata": metadata,
                    "execution_time": time.time() - start_time
                }
        
        except Exception as e:
            logger.error(f"Error in structured query processing: {e}")
            return {
                "type": "structured",
                "result": "I encountered an error processing your query. Please try rephrasing your question.",
                "value": None,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _build_filter_description(self, filters: Dict[str, Any]) -> str:
        """Build human-readable description of applied filters"""
        if not filters:
            return ""
        
        descriptions = []
        if "unit_number" in filters:
            descriptions.append(f"for unit {filters['unit_number']}")
        if "unit_type" in filters:
            descriptions.append(f"for {filters['unit_type']} units")
        
        return f" {' and '.join(descriptions)}" if descriptions else ""
    
    def process_semantic_query(self, query: str) -> Dict[str, Any]:
        """Process queries that require semantic search"""
        start_time = time.time()
        
        try:
            # Extract any filters from the query
            metadata = self.extract_query_metadata(query)
            filters = {}
            if "unit_number" in metadata.get("filters", {}):
                filters["unit_number"] = metadata["filters"]["unit_number"]
            
            search_results = self.storage_manager.search_similar_documents(query, filters=filters, limit=5)
            
            if not search_results:
                return {
                    "type": "semantic",
                    "result": "No relevant documents found for your query. Please try rephrasing or asking about specific units or lease terms.",
                    "sources": [],
                    "execution_time": time.time() - start_time
                }
            
            # Extract relevant content
            relevant_content = []
            sources = []
            
            for result in search_results:
                if result['score'] > 0.7:  # Only include high-confidence results
                    relevant_content.append(result['content'])
                    source_info = f"Document {result['document_id']} (confidence: {result['score']:.2f})"
                    if result.get('unit_number'):
                        source_info += f" - Unit {result['unit_number']}"
                    sources.append(source_info)
            
            if not relevant_content:
                return {
                    "type": "semantic",
                    "result": "I found some potentially relevant documents, but they don't seem closely related to your query. Could you be more specific?",
                    "sources": [f"Document {r['document_id']} (low confidence: {r['score']:.2f})" for r in search_results[:3]],
                    "execution_time": time.time() - start_time
                }
            
            # Use LLM to synthesize answer from relevant content
            context = "\n\n".join(relevant_content[:3])  # Limit context length
            
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                HumanMessagePromptTemplate.from_template(
                    "Based on the following document content, answer the user's question: {query}\n\n"
                    "Document content:\n{context}\n\n"
                    "Provide a clear, concise answer based on the available information. "
                    "If the information is incomplete, mention what additional details might be helpful."
                )
            ])
            
            response = self.llm(prompt.format_messages(query=query, context=context))
            
            return {
                "type": "semantic",
                "result": response.content,
                "sources": sources,
                "execution_time": time.time() - start_time
            }
        
        except Exception as e:
            logger.error(f"Error in semantic query processing: {e}")
            return {
                "type": "semantic",
                "result": "I encountered an error searching through the documents. Please try rephrasing your question.",
                "sources": [],
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def process_hybrid_query(self, query: str) -> Dict[str, Any]:
        """Process queries that require both structured and semantic search"""
        start_time = time.time()
        
        try:
            # Get both types of results
            structured_result = self.process_structured_query(query)
            semantic_result = self.process_semantic_query(query)
            
            # Combine results using LLM
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                HumanMessagePromptTemplate.from_template(
                    "User query: {query}\n\n"
                    "Structured data result: {structured}\n\n"
                    "Document search result: {semantic}\n\n"
                    "Provide a comprehensive answer that combines both sources of information. "
                    "Prioritize the structured data for factual numbers, and use document content for additional context."
                )
            ])
            
            response = self.llm(prompt.format_messages(
                query=query,
                structured=structured_result.get('result', 'No structured data found'),
                semantic=semantic_result.get('result', 'No document content found')
            ))
            
            # Combine sources
            all_sources = semantic_result.get('sources', [])
            if structured_result.get('value') is not None:
                all_sources.append("Database query results")
            
            return {
                "type": "hybrid",
                "result": response.content,
                "sources": all_sources,
                "structured_data": structured_result.get('value'),
                "filters_applied": structured_result.get('filters_applied', {}),
                "execution_time": time.time() - start_time
            }
        
        except Exception as e:
            logger.error(f"Error in hybrid query processing: {e}")
            return {
                "type": "hybrid",
                "result": "I encountered an error processing your query. Please try breaking it down into simpler questions.",
                "sources": [],
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def process_query(self, request: QueryRequest) -> QueryResponse:
        """Main query processing method"""
        try:
            start_time = time.time()
            query_type = self.classify_query_type(request.query)
            logger.info(f"Processing {query_type} query: {request.query}")
            
            if query_type == "structured":
                result = self.process_structured_query(request.query)
            elif query_type == "semantic":
                result = self.process_semantic_query(request.query)
            else:
                result = self.process_hybrid_query(request.query)
            
            # Calculate confidence based on result quality
            confidence = 0.9 if result.get('value') is not None else 0.7
            if result.get('error'):
                confidence = 0.3
            
            total_time = time.time() - start_time
            
            return QueryResponse(
                answer=result['result'],
                sources=result.get('sources', []),
                confidence=confidence,
                query_type=query_type,
                filters_applied=result.get('filters_applied'),
                metadata=result.get('metadata'),
                execution_time=total_time
            )
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return QueryResponse(
                answer="I encountered an error while processing your query. Please try again or rephrase your question.",
                sources=[],
                confidence=0.0,
                query_type="error",
                execution_time=time.time() - start_time if 'start_time' in locals() else 0.0
            )
    
    def get_example_queries(self) -> Dict[str, List[str]]:
        """Get example queries based on your PDF format"""
        return {
            "structured_queries": [
                "What is the total rent for the property?",
                "What is the total square feet for the property?", 
                "How many units are occupied vs vacant?",
                "What's the average rent for 2-bedroom units?",
                "Show me occupancy statistics",
                "What's the total rent for MBL2AC60 units?",
                "How many vacant units do we have?"
            ],
            "semantic_queries": [
                "Find lease agreements with pet policies",
                "Show me maintenance-related documents",
                "What are the parking policies?",
                "Find documents about lease termination",
                "Show me lease terms and conditions",
                "Find information about security deposits"
            ],
            "hybrid_queries": [
                "Tell me about unit 01-101",
                "Show me lease terms for occupied units", 
                "Find high-rent units with specific amenities",
                "What's the rent for units with Simon as tenant?",
                "Show me details for MBL3AC60 units"
            ]
        }

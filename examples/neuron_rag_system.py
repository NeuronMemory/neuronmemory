"""
NeuronMemory-Enhanced RAG System

This application demonstrates a next-generation RAG system that combines:
- Traditional document retrieval from PDF sources
- NeuronMemory API for persistent memory and learning
- Personalized responses based on user interaction history
- Knowledge gap identification and adaptive responses
- Multi-session continuity and relationship building

Features:
- PDF document processing and chunking
- Semantic search with vector embeddings
- Memory-enhanced context generation
- User preference learning
- Session continuity across interactions
- Analytics and insights generation
"""

import asyncio
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

# Document processing
try:
    import PyMuPDF as fitz  # pymupdf for PDF processing
except ImportError:
    print("Please install PyMuPDF: pip install PyMuPDF")
    fitz = None

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# NeuronMemory integration
from neuron_memory import NeuronMemoryAPI

# OpenAI for embeddings and generation
import openai
from openai import OpenAI

# Text processing
import re
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    
    # Ensure NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
except ImportError:
    print("Please install NLTK: pip install nltk")
    nltk = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuronRAGSystem:
    """Advanced RAG system with NeuronMemory integration for persistent learning and memory"""
    
    def __init__(self, docs_directory: str = "docs_data", collection_name: str = "expat_guide"):
        self.docs_directory = Path(docs_directory)
        self.collection_name = collection_name
        
        # Initialize NeuronMemory
        self.memory = NeuronMemoryAPI()
        self.current_session = None
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize OpenAI client
        self.openai_client = OpenAI()
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.sentence_transformer_ef
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=self.sentence_transformer_ef
            )
            logger.info(f"Created new collection: {collection_name}")
        
        # Document processing settings
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # RAG settings
        self.top_k_retrieval = 5
        self.relevance_threshold = 0.7
        
    async def start_rag_session(self, user_id: str, session_context: str = "expat_guide_consultation") -> str:
        """Start a new RAG session with memory integration"""
        self.current_session = f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
        
        await self.memory.start_session(
            session_id=self.current_session,
            user_id=user_id,
            task=f"RAG consultation about expat life - {session_context}",
            domain="expat_guidance"
        )
        
        # Get user's previous interests and knowledge gaps
        user_profile = await self._get_user_rag_profile(user_id)
        
        # Create personalized greeting
        greeting = await self._create_personalized_greeting(user_profile, user_id)
        
        # Log session start
        await self._log_rag_event(
            "session_start", greeting, user_id, "greeting",
            {"session_context": session_context, "user_profile": user_profile}
        )
        
        logger.info(f"Started RAG session: {self.current_session}")
        return greeting
    
    def process_pdf_documents(self) -> Dict[str, Any]:
        """Process all PDF documents in the docs directory"""
        processing_results = {
            "processed_files": [],
            "total_chunks": 0,
            "processing_errors": []
        }
        
        if not fitz:
            processing_results["processing_errors"].append("PyMuPDF not installed")
            return processing_results
        
        pdf_files = list(self.docs_directory.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.docs_directory}")
            return processing_results
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing PDF: {pdf_file.name}")
                
                # Extract text from PDF
                text_content = self._extract_pdf_text(pdf_file)
                
                # Create chunks
                chunks = self._create_text_chunks(text_content)
                
                # Generate embeddings and store
                doc_chunks_stored = self._store_document_chunks(pdf_file.name, chunks)
                
                processing_results["processed_files"].append({
                    "filename": pdf_file.name,
                    "chunks_created": len(chunks),
                    "chunks_stored": doc_chunks_stored
                })
                processing_results["total_chunks"] += doc_chunks_stored
                
                logger.info(f"Successfully processed {pdf_file.name}: {doc_chunks_stored} chunks stored")
                
            except Exception as e:
                error_msg = f"Error processing {pdf_file.name}: {str(e)}"
                logger.error(error_msg)
                processing_results["processing_errors"].append(error_msg)
        
        return processing_results
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text content from PDF file"""
        text_content = ""
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Clean and process text
                text = self._clean_text(text)
                text_content += f"\n\n--- Page {page_num + 1} ---\n\n{text}"
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            raise
        
        return text_content
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:()\-\'""]', '', text)
        
        # Normalize quotes
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r'['']', "'", text)
        
        return text.strip()
    
    def _create_text_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create overlapping chunks from text content"""
        chunks = []
        
        if nltk:
            # Split into sentences for more natural chunk boundaries
            sentences = sent_tokenize(text)
        else:
            # Fallback sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip(),
                    "length": current_length,
                    "sentence_count": len(sent_tokenize(current_chunk)) if nltk else current_chunk.count('.') + 1
                })
                
                # Start new chunk with overlap
                if nltk:
                    overlap_sentences = sent_tokenize(current_chunk)[-3:]  # Keep last 3 sentences for overlap
                else:
                    overlap_sentences = current_chunk.split('.')[-3:]
                current_chunk = " ".join(overlap_sentences) + " " + sentence if overlap_sentences else sentence
                current_length = len(current_chunk)
                chunk_id += 1
            else:
                current_chunk += " " + sentence
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "chunk_id": chunk_id,
                "text": current_chunk.strip(),
                "length": current_length,
                "sentence_count": len(sent_tokenize(current_chunk)) if nltk else current_chunk.count('.') + 1
            })
        
        return chunks
    
    def _store_document_chunks(self, filename: str, chunks: List[Dict[str, Any]]) -> int:
        """Store document chunks in ChromaDB with metadata"""
        stored_count = 0
        
        for chunk in chunks:
            try:
                # Create unique ID for chunk
                chunk_text = chunk["text"]
                chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
                chunk_id = f"{filename}_{chunk['chunk_id']}_{chunk_hash[:8]}"
                
                # Prepare metadata
                metadata = {
                    "source_file": filename,
                    "chunk_id": chunk["chunk_id"],
                    "length": chunk["length"],
                    "sentence_count": chunk["sentence_count"],
                    "processed_date": datetime.now().isoformat()
                }
                
                # Check if chunk already exists
                try:
                    existing = self.collection.get(ids=[chunk_id])
                    if existing["ids"]:
                        logger.debug(f"Chunk {chunk_id} already exists, skipping")
                        continue
                except:
                    pass  # Chunk doesn't exist, proceed with storage
                
                # Store in ChromaDB
                self.collection.add(
                    documents=[chunk_text],
                    metadatas=[metadata],
                    ids=[chunk_id]
                )
                
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Error storing chunk {chunk.get('chunk_id', 'unknown')}: {str(e)}")
        
        return stored_count
    
    async def query_documents(self, query: str, user_id: str, max_results: int = None) -> Dict[str, Any]:
        """Query documents with memory-enhanced context"""
        if max_results is None:
            max_results = self.top_k_retrieval
        
        logger.info(f"Processing query: {query}")
        
        # Get user's memory context for enhanced retrieval
        memory_context = await self.memory.get_context_for_llm(
            query=query,
            user_id=user_id,
            session_id=self.current_session,
            max_context_length=1000
        )
        
        # Retrieve relevant document chunks
        retrieval_results = self._retrieve_relevant_chunks(query, max_results)
        
        # Generate memory-enhanced response
        response = await self._generate_enhanced_response(
            query, retrieval_results, memory_context, user_id
        )
        
        # Learn from this interaction
        await self._learn_from_query(query, response, retrieval_results, user_id)
        
        # Log the interaction
        await self._log_rag_event(
            query, response["answer"], user_id, "query_response",
            {
                "retrieved_chunks": len(retrieval_results["chunks"]),
                "memory_context_used": bool(memory_context),
                "relevance_scores": retrieval_results["scores"]
            }
        )
        
        return {
            "query": query,
            "answer": response["answer"],
            "sources": retrieval_results["sources"],
            "relevance_scores": retrieval_results["scores"],
            "memory_enhanced": bool(memory_context),
            "learning_applied": response["learning_applied"],
            "suggested_followups": response.get("suggested_followups", [])
        }
    
    def _retrieve_relevant_chunks(self, query: str, max_results: int) -> Dict[str, Any]:
        """Retrieve relevant document chunks using semantic search"""
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results
            )
            
            chunks = []
            sources = []
            scores = []
            
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    similarity_score = 1 / (1 + distance)
                    
                    if similarity_score >= self.relevance_threshold:
                        chunks.append(doc)
                        sources.append({
                            "source_file": metadata.get("source_file", "unknown"),
                            "chunk_id": metadata.get("chunk_id", i),
                            "similarity_score": similarity_score
                        })
                        scores.append(similarity_score)
            
            return {
                "chunks": chunks,
                "sources": sources,
                "scores": scores,
                "total_found": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return {"chunks": [], "sources": [], "scores": [], "total_found": 0}
    
    async def _generate_enhanced_response(
        self, 
        query: str, 
        retrieval_results: Dict[str, Any], 
        memory_context: str, 
        user_id: str
    ) -> Dict[str, Any]:
        """Generate response using retrieved documents and memory context"""
        
        # Get user preferences and patterns
        user_profile = await self._get_user_rag_profile(user_id)
        
        # Build context for LLM
        context_parts = []
        
        # Add memory context if available
        if memory_context:
            context_parts.append(f"Previous conversation context:\n{memory_context}\n")
        
        # Add user preferences
        if user_profile.get("interests"):
            context_parts.append(f"User's known interests: {', '.join(user_profile['interests'])}\n")
        
        if user_profile.get("communication_style"):
            context_parts.append(f"Preferred communication style: {user_profile['communication_style']}\n")
        
        # Add retrieved document chunks
        if retrieval_results["chunks"]:
            context_parts.append("Relevant information from documents:\n")
            for i, (chunk, source) in enumerate(zip(retrieval_results["chunks"], retrieval_results["sources"])):
                context_parts.append(f"Source {i+1} (from {source['source_file']}):\n{chunk}\n")
        
        context = "\n".join(context_parts)
        
        # Create the prompt
        prompt = f"""You are an expert assistant for expat life guidance. Use the provided information to answer the user's question comprehensively and helpfully.

{context}

User Question: {query}

Instructions:
1. Provide a detailed, helpful answer based on the retrieved information
2. If the user has previous context, acknowledge it and build upon it
3. Cite specific sources when referencing information from documents
4. If information is not available in the documents, clearly state this
5. Suggest 2-3 relevant follow-up questions the user might be interested in
6. Adapt your communication style to the user's preferences if known

Answer:"""

        try:
            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert expat life advisor with access to comprehensive guidance materials."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # Extract suggested follow-ups using simple parsing
            suggested_followups = self._extract_followup_questions(answer)
            
            return {
                "answer": answer,
                "learning_applied": bool(memory_context),
                "suggested_followups": suggested_followups
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            
            # Fallback response
            if retrieval_results["chunks"]:
                fallback_answer = f"Based on the available information:\n\n{retrieval_results['chunks'][0][:500]}..."
            else:
                fallback_answer = "I couldn't find specific information about your question in the available documents. Could you please rephrase your question or ask about a different topic?"
            
            return {
                "answer": fallback_answer,
                "learning_applied": False,
                "suggested_followups": []
            }
    
    def _extract_followup_questions(self, response_text: str) -> List[str]:
        """Extract suggested follow-up questions from response"""
        followups = []
        
        # Look for common patterns of follow-up questions
        patterns = [
            r"follow.{0,20}questions?:?\s*(.+?)(?:\n\n|\Z)",
            r"you might also ask:?\s*(.+?)(?:\n\n|\Z)",
            r"related questions?:?\s*(.+?)(?:\n\n|\Z)",
            r"consider asking:?\s*(.+?)(?:\n\n|\Z)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Split by common delimiters and clean up
                questions = re.split(r'[•\-\d+\.\)\n]', match)
                for q in questions:
                    q = q.strip()
                    if q and len(q) > 10 and '?' in q:
                        followups.append(q)
        
        return followups[:3]  # Return top 3 follow-ups
    
    async def _get_user_rag_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user's RAG profile from memory"""
        try:
            # Get user's interests and preferences
            interest_memories = await self.memory.search_memories(
                query=f"user {user_id} expat interests topics preferences",
                user_id=user_id,
                memory_types=["social", "semantic"],
                limit=15
            )
            
            # Get communication style preferences
            style_memories = await self.memory.search_memories(
                query=f"user {user_id} communication style helpful detailed brief",
                user_id=user_id,
                memory_types=["procedural", "social"],
                limit=10
            )
            
            # Get previous topics and knowledge gaps
            topic_memories = await self.memory.search_memories(
                query=f"user {user_id} asked about visa work housing",
                user_id=user_id,
                memory_types=["episodic"],
                limit=20
            )
            
            profile = {
                "total_interactions": len(topic_memories),
                "interests": [],
                "communication_style": "detailed",
                "frequent_topics": [],
                "knowledge_level": "beginner"
            }
            
            # Extract interests and topics
            all_memories = interest_memories + style_memories + topic_memories
            for memory in all_memories:
                content = memory["content"].lower()
                
                # Extract interests
                expat_topics = ["visa", "housing", "work", "tax", "healthcare", "culture", "language", "banking", "immigration"]
                for topic in expat_topics:
                    if topic in content and topic not in profile["interests"]:
                        profile["interests"].append(topic)
                
                # Determine communication style
                if "detailed" in content or "comprehensive" in content:
                    profile["communication_style"] = "detailed"
                elif "brief" in content or "concise" in content:
                    profile["communication_style"] = "brief"
                elif "examples" in content:
                    profile["communication_style"] = "example-based"
            
            # Determine knowledge level based on interaction patterns
            if profile["total_interactions"] > 15:
                profile["knowledge_level"] = "experienced"
            elif profile["total_interactions"] > 5:
                profile["knowledge_level"] = "intermediate"
            
            return profile
            
        except Exception as e:
            logger.error(f"Error getting user RAG profile: {str(e)}")
            return {
                "total_interactions": 0,
                "interests": [],
                "communication_style": "detailed",
                "frequent_topics": [],
                "knowledge_level": "beginner"
            }
    
    async def _create_personalized_greeting(self, user_profile: Dict[str, Any], user_id: str) -> str:
        """Create personalized greeting based on user profile"""
        
        interaction_count = user_profile.get("total_interactions", 0)
        interests = user_profile.get("interests", [])
        knowledge_level = user_profile.get("knowledge_level", "beginner")
        
        if interaction_count == 0:
            greeting = """Hello! I'm your expat life guidance assistant. I have access to comprehensive information about living abroad, including topics like visas, housing, work permits, taxes, healthcare, and cultural adaptation.

What would you like to know about expat life today? I can help with specific questions or general guidance on any aspect of living abroad."""
        
        elif interaction_count < 5:
            greeting = f"""Welcome back! I see you've been exploring expat life information with me before. 

Based on our previous conversations, I'm here to help you with any questions about living abroad. What's on your mind today?"""
        
        else:
            topics_mentioned = f" particularly around {', '.join(interests[:3])}" if interests else ""
            greeting = f"""Hello again! Great to see you back. I remember our previous conversations about expat life{topics_mentioned}.

As an {knowledge_level}-level expat researcher, what would you like to dive into today? I'm here to provide detailed guidance tailored to your experience level."""
        
        return greeting
    
    async def _learn_from_query(
        self, 
        query: str, 
        response: Dict[str, Any], 
        retrieval_results: Dict[str, Any], 
        user_id: str
    ):
        """Learn from user query and response for future personalization"""
        
        # Store the query and successful retrieval patterns
        await self.memory.store_memory(
            content=f"User asked: '{query}'. Successfully retrieved {len(retrieval_results['chunks'])} relevant chunks and provided comprehensive answer.",
            memory_type="episodic",
            importance=7,
            user_id=user_id,
            session_id=self.current_session,
            metadata={
                "query_type": "expat_guidance",
                "retrieval_success": len(retrieval_results['chunks']) > 0,
                "response_quality": "comprehensive" if len(response["answer"]) > 200 else "brief",
                "topics_covered": self._extract_topics_from_query(query)
            }
        )
        
        # Store user interest patterns
        topics = self._extract_topics_from_query(query)
        if topics:
            await self.memory.store_memory(
                content=f"User expressed interest in: {', '.join(topics)}",
                memory_type="social",
                importance=6,
                user_id=user_id,
                session_id=self.current_session,
                metadata={
                    "interest_type": "expat_topics",
                    "topics": topics
                }
            )
        
        # Store effective retrieval patterns for system improvement
        if retrieval_results["chunks"]:
            await self.memory.store_memory(
                content=f"Query pattern '{query[:50]}...' successfully matched with document sources: {[s['source_file'] for s in retrieval_results['sources']]}",
                memory_type="procedural",
                importance=5,
                user_id=user_id,
                session_id=self.current_session,
                metadata={
                    "retrieval_pattern": "successful",
                    "source_files": [s['source_file'] for s in retrieval_results['sources']],
                    "avg_relevance": sum(retrieval_results['scores']) / len(retrieval_results['scores']) if retrieval_results['scores'] else 0
                }
            )
    
    def _extract_topics_from_query(self, query: str) -> List[str]:
        """Extract relevant expat topics from user query"""
        topics = []
        query_lower = query.lower()
        
        topic_keywords = {
            "visa": ["visa", "permit", "immigration", "entry"],
            "work": ["work", "job", "employment", "career", "salary"],
            "housing": ["housing", "apartment", "rent", "accommodation", "home"],
            "tax": ["tax", "taxes", "taxation", "income", "filing"],
            "healthcare": ["healthcare", "health", "medical", "insurance", "doctor"],
            "banking": ["bank", "banking", "account", "money", "finance", "payment"],
            "culture": ["culture", "cultural", "customs", "traditions", "social"],
            "language": ["language", "speak", "communication", "translate"],
            "education": ["school", "education", "university", "children", "kids"],
            "legal": ["legal", "law", "rights", "lawyer", "documentation"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    async def _log_rag_event(
        self, 
        user_input: str, 
        ai_response: str, 
        user_id: str, 
        event_type: str,
        metadata: Dict
    ):
        """Log RAG events for analytics and improvement"""
        
        # Create structured log entry
        log_content = f"RAG {event_type}: User query processed, retrieved relevant information, enhanced with memory context"
        
        await self.memory.store_memory(
            content=log_content,
            memory_type="episodic",
            importance=6,
            user_id=user_id,
            session_id=self.current_session,
            metadata={
                "event_type": f"rag_{event_type}",
                "query_length": len(user_input),
                "response_length": len(ai_response),
                "retrieval_success": metadata.get("retrieved_chunks", 0) > 0,
                "memory_enhanced": metadata.get("memory_context_used", False),
                "session_type": "expat_guidance_rag"
            }
        )
    
    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics about user's RAG usage patterns"""
        
        # Get all RAG interactions
        rag_memories = await self.memory.search_memories(
            query=f"user {user_id} RAG query expat guidance",
            user_id=user_id,
            memory_types=["episodic"],
            limit=50
        )
        
        # Get interest patterns
        interest_memories = await self.memory.search_memories(
            query=f"user {user_id} interest expat topics",
            user_id=user_id,
            memory_types=["social"],
            limit=30
        )
        
        analytics = {
            "total_queries": len(rag_memories),
            "top_interests": [],
            "query_patterns": [],
            "knowledge_progression": "beginner",
            "preferred_topics": {},
            "response_satisfaction": "unknown"
        }
        
        # Analyze interests
        topic_counts = {}
        for memory in interest_memories:
            topics = memory.get("metadata", {}).get("topics", [])
            for topic in topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        analytics["top_interests"] = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Determine knowledge progression
        if analytics["total_queries"] > 20:
            analytics["knowledge_progression"] = "advanced"
        elif analytics["total_queries"] > 8:
            analytics["knowledge_progression"] = "intermediate"
        
        return analytics
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        try:
            # Get collection info
            collection_count = self.collection.count()
            
            # Get sample documents for analysis
            sample_results = self.collection.get(limit=min(100, collection_count))
            
            stats = {
                "total_chunks": collection_count,
                "source_files": set(),
                "avg_chunk_length": 0,
                "processing_date": None
            }
            
            if sample_results["metadatas"]:
                # Analyze metadata
                total_length = 0
                for metadata in sample_results["metadatas"]:
                    if "source_file" in metadata:
                        stats["source_files"].add(metadata["source_file"])
                    if "length" in metadata:
                        total_length += metadata["length"]
                    if "processed_date" in metadata and not stats["processing_date"]:
                        stats["processing_date"] = metadata["processed_date"]
                
                stats["avg_chunk_length"] = total_length / len(sample_results["metadatas"]) if sample_results["metadatas"] else 0
                stats["source_files"] = list(stats["source_files"])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
    
    async def end_rag_session(self, user_id: str, session_summary: str = None):
        """End RAG session with summary and learning consolidation"""
        
        if session_summary:
            await self.memory.store_memory(
                content=f"RAG session ended. Summary: {session_summary}",
                memory_type="episodic",
                importance=7,
                user_id=user_id,
                session_id=self.current_session,
                metadata={
                    "event_type": "session_end",
                    "has_summary": True,
                    "session_type": "expat_guidance_rag"
                }
            )
        
        await self.memory.end_session(self.current_session)
        
        logger.info(f"Ended RAG session: {self.current_session}")
        self.current_session = None


async def demo_neuron_rag_system():
    """Demonstrate the NeuronMemory-enhanced RAG system"""
    
    print("=" * 70)
    print("🔍 NEURONMEMORY-ENHANCED RAG SYSTEM DEMO")
    print("=" * 70)
    print("Advanced RAG with Memory, Learning, and Personalization")
    print()
    
    # Initialize the system
    rag_system = NeuronRAGSystem()
    
    # Demo user
    user_id = "demo_user_expat"
    
    # Step 1: Process PDF documents
    print("📄 STEP 1: Processing PDF Documents")
    print("-" * 40)
    
    processing_results = rag_system.process_pdf_documents()
    print(f"✅ Document Processing Results:")
    print(f"   • Files processed: {len(processing_results['processed_files'])}")
    print(f"   • Total chunks created: {processing_results['total_chunks']}")
    
    for file_result in processing_results['processed_files']:
        print(f"   • {file_result['filename']}: {file_result['chunks_stored']} chunks")
    
    if processing_results['processing_errors']:
        print(f"   ⚠️ Errors: {len(processing_results['processing_errors'])}")
    print()
    
    # Step 2: Start RAG session
    print("🚀 STEP 2: Starting Personalized RAG Session")
    print("-" * 40)
    
    greeting = await rag_system.start_rag_session(user_id, "first_time_expat_guidance")
    print(f"System: {greeting}")
    print()
    
    # Step 3: Query examples with learning progression
    print("💬 STEP 3: Interactive Query Examples")
    print("-" * 40)
    
    # Query 1: Visa information
    print("Query 1: Visa and Immigration")
    query1 = "What do I need to know about getting a work visa for a new country?"
    
    response1 = await rag_system.query_documents(query1, user_id)
    print(f"User: {query1}")
    print(f"System: {response1['answer'][:300]}...")
    print(f"📊 Retrieved {len(response1['sources'])} relevant sources")
    print(f"🧠 Memory enhanced: {response1['memory_enhanced']}")
    print()
    
    # Query 2: Housing with memory context
    print("Query 2: Housing (Building on Previous Context)")
    query2 = "Now that I understand the visa process, what should I know about finding housing?"
    
    response2 = await rag_system.query_documents(query2, user_id)
    print(f"User: {query2}")
    print(f"System: {response2['answer'][:300]}...")
    print(f"📊 Retrieved {len(response2['sources'])} relevant sources")
    print(f"🧠 Memory enhanced: {response2['memory_enhanced']}")
    if response2.get('suggested_followups'):
        print(f"💡 Suggested follow-ups: {response2['suggested_followups'][:2]}")
    print()
    
    # Query 3: Tax information with personalization
    print("Query 3: Tax Information (Personalized Response)")
    query3 = "I'm particularly interested in tax implications. Can you give me detailed information?"
    
    response3 = await rag_system.query_documents(query3, user_id)
    print(f"User: {query3}")
    print(f"System: {response3['answer'][:300]}...")
    print(f"📊 Retrieved {len(response3['sources'])} relevant sources")
    print(f"🧠 Memory enhanced: {response3['memory_enhanced']}")
    print()
    
    # Step 4: Analytics and insights
    print("📈 STEP 4: User Analytics and Learning Insights")
    print("-" * 40)
    
    analytics = await rag_system.get_user_analytics(user_id)
    print(f"👤 User Analytics:")
    print(f"   • Total queries: {analytics['total_queries']}")
    print(f"   • Knowledge level: {analytics['knowledge_progression']}")
    if analytics['top_interests']:
        print(f"   • Top interests: {[f'{topic} ({count})' for topic, count in analytics['top_interests'][:3]]}")
    print()
    
    # Step 5: Collection statistics
    print("📚 STEP 5: Document Collection Statistics")
    print("-" * 40)
    
    collection_stats = rag_system.get_collection_stats()
    print(f"📊 Collection Statistics:")
    print(f"   • Total document chunks: {collection_stats.get('total_chunks', 'N/A')}")
    print(f"   • Source files: {collection_stats.get('source_files', [])}")
    print(f"   • Average chunk length: {collection_stats.get('avg_chunk_length', 0):.0f} characters")
    print()
    
    # Step 6: End session
    await rag_system.end_rag_session(
        user_id, 
        "User explored visa, housing, and tax topics. Showed preference for detailed information. Good engagement with follow-up suggestions."
    )
    
    print("=" * 70)
    print("🔍 NEURONMEMORY RAG DEMO COMPLETE")
    print("=" * 70)
    print("Key Features Demonstrated:")
    print("• PDF document processing and intelligent chunking")
    print("• Semantic search with relevance scoring")
    print("• Memory-enhanced context for personalized responses")
    print("• Learning from user interactions and preferences")
    print("• Cross-session continuity and relationship building")
    print("• Analytics and insights for continuous improvement")
    print("• Intelligent follow-up question suggestions")
    print()
    print("🚀 INTEGRATION READY:")
    print("This system can be integrated into:")
    print("• Web applications with REST API")
    print("• Chat interfaces and messaging platforms")
    print("• Enterprise knowledge management systems")
    print("• Mobile applications for expat guidance")
    print("• Voice assistants and conversational AI")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_neuron_rag_system()) 
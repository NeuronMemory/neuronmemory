"""
RAG Application with NeuronMemory Integration
Enhanced Retrieval-Augmented Generation with persistent memory and learning
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# NeuronMemory imports
from neuron_memory import NeuronMemoryAPI

# For this example, we'll simulate a document store and LLM
# In a real application, you'd use your actual vector database and LLM

@dataclass
class Document:
    """Document structure for our knowledge base"""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]

class SimpleDocumentStore:
    """Simulated document store - replace with your actual vector DB"""
    
    def __init__(self):
        # Sample documents for demonstration
        self.documents = [
            Document(
                id="doc1",
                title="Python Best Practices",
                content="Python best practices include using virtual environments, following PEP 8 style guide, writing docstrings, using type hints, and implementing proper error handling.",
                metadata={"category": "programming", "language": "python"}
            ),
            Document(
                id="doc2", 
                title="Machine Learning Fundamentals",
                content="Machine learning involves training algorithms to make predictions or decisions based on data. Key concepts include supervised learning, unsupervised learning, and reinforcement learning.",
                metadata={"category": "AI", "difficulty": "beginner"}
            ),
            Document(
                id="doc3",
                title="Database Design Principles", 
                content="Good database design follows normalization principles, uses appropriate indexes, maintains data integrity, and considers performance optimization.",
                metadata={"category": "databases", "difficulty": "intermediate"}
            )
        ]
    
    async def search(self, query: str, limit: int = 5) -> List[Document]:
        """Simple keyword search - replace with your vector search"""
        results = []
        query_lower = query.lower()
        
        for doc in self.documents:
            # Simple relevance scoring based on keyword matches
            score = 0
            for word in query_lower.split():
                if word in doc.content.lower():
                    score += 1
                if word in doc.title.lower():
                    score += 2
            
            if score > 0:
                results.append((doc, score))
        
        # Sort by relevance score
        results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in results[:limit]]

class IntelligentRAGSystem:
    """
    Enhanced RAG system with NeuronMemory integration
    """
    
    def __init__(self):
        self.memory = NeuronMemoryAPI()
        self.document_store = SimpleDocumentStore()
        self.conversation_history = []
        
    async def start_conversation(self, user_id: str, topic: Optional[str] = None) -> str:
        """Start a new conversation session with memory context"""
        session_id = f"rag_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
        
        await self.memory.start_session(
            session_id=session_id,
            user_id=user_id,
            task="RAG conversation",
            domain=topic or "general"
        )
        
        print(f"🚀 Started RAG conversation session: {session_id}")
        return session_id
    
    async def query(
        self, 
        user_question: str, 
        user_id: str, 
        session_id: str,
        store_interaction: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query with memory-enhanced RAG
        """
        print(f"\n📝 User Question: {user_question}")
        
        # Step 1: Get memory context about user and previous conversations
        memory_context = await self._get_memory_context(user_question, user_id, session_id)
        
        # Step 2: Search documents with query expansion based on memory
        expanded_query = await self._expand_query_with_memory(user_question, memory_context)
        documents = await self.document_store.search(expanded_query, limit=3)
        
        # Step 3: Get user preferences and conversation style from memory
        user_preferences = await self._get_user_preferences(user_id)
        
        # Step 4: Generate context-aware response
        response = await self._generate_response(
            user_question=user_question,
            documents=documents,
            memory_context=memory_context,
            user_preferences=user_preferences,
            session_id=session_id
        )
        
        # Step 5: Store the interaction in memory for future reference
        if store_interaction:
            await self._store_interaction(
                user_question=user_question,
                response=response,
                documents=documents,
                user_id=user_id,
                session_id=session_id
            )
        
        return {
            "response": response,
            "sources": [{"title": doc.title, "id": doc.id} for doc in documents],
            "memory_context_used": bool(memory_context),
            "query_expanded": expanded_query != user_question
        }
    
    async def _get_memory_context(self, query: str, user_id: str, session_id: str) -> str:
        """Get relevant memory context for the current query"""
        try:
            memory_context = await self.memory.get_context_for_llm(
                query=f"user preferences conversation history {query}",
                user_id=user_id,
                session_id=session_id,
                max_context_length=1000
            )
            
            if memory_context:
                print(f"🧠 Using memory context: {len(memory_context)} characters")
            
            return memory_context
        except Exception as e:
            print(f"⚠️ Error getting memory context: {e}")
            return ""
    
    async def _expand_query_with_memory(self, original_query: str, memory_context: str) -> str:
        """Expand the query based on memory context"""
        if not memory_context:
            return original_query
        
        # Simple query expansion - in a real app, use LLM for this
        if "python" in memory_context.lower() and "best practices" in original_query.lower():
            return f"{original_query} python programming code quality"
        elif "beginner" in memory_context.lower():
            return f"{original_query} introduction basics fundamentals"
        elif "advanced" in memory_context.lower():
            return f"{original_query} advanced expert deep dive"
        
        return original_query
    
    async def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences from social memory"""
        try:
            # Search for social memories about this user
            user_memories = await self.memory.search_memories(
                query=f"user preferences learning style communication {user_id}",
                user_id=user_id,
                memory_types=["social", "episodic"],
                limit=5
            )
            
            preferences = {
                "explanation_style": "detailed",  # default
                "technical_level": "intermediate",  # default
                "preferred_examples": True  # default
            }
            
            # Extract preferences from memories
            for memory in user_memories:
                content = memory["content"].lower()
                if "beginner" in content or "simple" in content:
                    preferences["technical_level"] = "beginner"
                elif "advanced" in content or "expert" in content:
                    preferences["technical_level"] = "advanced"
                
                if "brief" in content or "concise" in content:
                    preferences["explanation_style"] = "brief"
                elif "detailed" in content or "thorough" in content:
                    preferences["explanation_style"] = "detailed"
            
            return preferences
            
        except Exception as e:
            print(f"⚠️ Error getting user preferences: {e}")
            return {"explanation_style": "detailed", "technical_level": "intermediate"}
    
    async def _generate_response(
        self,
        user_question: str,
        documents: List[Document],
        memory_context: str,
        user_preferences: Dict[str, Any],
        session_id: str
    ) -> str:
        """Generate a response using retrieved documents and memory context"""
        
        # Build the context for LLM
        document_context = "\n\n".join([
            f"**{doc.title}**\n{doc.content}" for doc in documents
        ])
        
        # Simulate LLM response generation
        # In a real application, send this to your LLM (OpenAI, Anthropic, etc.)
        
        response_parts = []
        
        # Add personalized greeting if we have memory context
        if memory_context and "conversation" in memory_context.lower():
            response_parts.append("Based on our previous conversations,")
        
        # Adapt response style based on user preferences
        style = user_preferences.get("explanation_style", "detailed")
        level = user_preferences.get("technical_level", "intermediate")
        
        if level == "beginner":
            response_parts.append("let me explain this in simple terms:")
        elif level == "advanced":
            response_parts.append("here's a comprehensive technical explanation:")
        else:
            response_parts.append("here's what you need to know:")
        
        # Add main content based on documents
        if documents:
            main_content = f"\n\n{documents[0].content}"
            if style == "brief":
                # Simulate brief response
                main_content = main_content[:200] + "..."
            response_parts.append(main_content)
            
            # Add examples if user prefers them
            if user_preferences.get("preferred_examples", True) and len(documents) > 1:
                response_parts.append(f"\n\nAdditional context: {documents[1].content[:150]}...")
        else:
            response_parts.append("\n\nI don't have specific information about that in my knowledge base, but I'll remember your question for future reference.")
        
        return " ".join(response_parts)
    
    async def _store_interaction(
        self,
        user_question: str,
        response: str,
        documents: List[Document],
        user_id: str,
        session_id: str
    ):
        """Store the interaction in memory for future reference"""
        try:
            # Store the conversation as episodic memory
            conversation_memory = f"User asked: '{user_question}'. I provided information about {', '.join([doc.title for doc in documents])}."
            await self.memory.create_episodic_memory(
                content=conversation_memory,
                participants=[user_id, "rag_assistant"],
                location="rag_conversation",
                user_id=user_id,
                session_id=session_id
            )
            
            # Store topic interest as semantic memory
            if documents:
                topic_interest = f"User is interested in {documents[0].metadata.get('category', 'general')} topics"
                await self.memory.create_memory(
                    content=topic_interest,
                    memory_type="semantic",
                    user_id=user_id,
                    session_id=session_id
                )
            
            # Store user interaction pattern as social memory
            interaction_pattern = f"User asks detailed questions about technical topics and prefers comprehensive answers"
            await self.memory.create_social_memory(
                content=interaction_pattern,
                person_id=user_id,
                relationship_type="user",
                user_id=user_id,
                session_id=session_id
            )
            
            print(f"💾 Stored interaction in memory")
            
        except Exception as e:
            print(f"⚠️ Error storing interaction: {e}")
    
    async def get_conversation_summary(self, session_id: str, user_id: str) -> str:
        """Get a summary of the conversation from memory"""
        try:
            memories = await self.memory.search_memories(
                query="conversation summary topics discussed",
                user_id=user_id,
                session_id=session_id,
                memory_types=["episodic"],
                limit=10
            )
            
            topics = set()
            for memory in memories:
                content = memory["content"]
                # Extract topics (simplified)
                if "information about" in content:
                    topic = content.split("information about")[1].split(".")[0].strip()
                    topics.add(topic)
            
            if topics:
                return f"In this conversation, we discussed: {', '.join(topics)}"
            else:
                return "This was our first conversation."
                
        except Exception as e:
            return f"Error getting summary: {e}"
    
    async def end_conversation(self, session_id: str):
        """End the conversation session"""
        try:
            await self.memory.end_session(session_id)
            print(f"🏁 Ended conversation session: {session_id}")
        except Exception as e:
            print(f"⚠️ Error ending session: {e}")

async def demo_intelligent_rag():
    """Demonstrate the intelligent RAG system with memory"""
    
    print("="*60)
    print("🤖 Intelligent RAG System with NeuronMemory")
    print("="*60)
    
    rag = IntelligentRAGSystem()
    user_id = "demo_user_123"
    
    # Start conversation
    session_id = await rag.start_conversation(user_id, topic="software_development")
    
    try:
        # Simulate a multi-turn conversation
        questions = [
            "What are Python best practices?",
            "Tell me more about error handling in Python", 
            "How do I design a good database?",
            "Can you explain machine learning basics?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*40}")
            print(f"Question {i}/{len(questions)}")
            print(f"{'='*40}")
            
            result = await rag.query(
                user_question=question,
                user_id=user_id,
                session_id=session_id
            )
            
            print(f"🤖 Response: {result['response'][:200]}...")
            print(f"📚 Sources: {', '.join([s['title'] for s in result['sources']])}")
            print(f"🧠 Memory used: {result['memory_context_used']}")
            print(f"🔍 Query expanded: {result['query_expanded']}")
            
            # Simulate user feedback to improve future responses
            if i == 2:  # After second question, store preference
                await rag.memory.create_social_memory(
                    content="User prefers detailed technical explanations with examples",
                    person_id=user_id,
                    relationship_type="user",
                    user_id=user_id,
                    session_id=session_id
                )
                print("💡 Stored user preference for detailed explanations")
        
        # Show conversation summary
        print(f"\n{'='*40}")
        print("📋 Conversation Summary")
        print(f"{'='*40}")
        summary = await rag.get_conversation_summary(session_id, user_id)
        print(f"Summary: {summary}")
        
        # Show memory statistics
        stats = await rag.memory.get_statistics()
        print(f"\n📊 Memory Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    finally:
        # End conversation
        await rag.end_conversation(session_id)

if __name__ == "__main__":
    asyncio.run(demo_intelligent_rag()) 
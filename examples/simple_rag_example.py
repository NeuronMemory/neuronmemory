"""
Simple RAG + NeuronMemory Integration Example
A working demonstration of memory-enhanced retrieval
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any
from neuron_memory import NeuronMemoryAPI

class SimpleRAGWithMemory:
    """Simple RAG system enhanced with NeuronMemory"""
    
    def __init__(self):
        self.memory = NeuronMemoryAPI()
        
        # Simple document store (replace with your vector DB)
        self.documents = [
            {
                "id": "doc1",
                "title": "Python Best Practices",
                "content": "Use virtual environments, follow PEP 8, write tests, use type hints.",
                "category": "programming"
            },
            {
                "id": "doc2", 
                "title": "Machine Learning Basics",
                "content": "ML involves training algorithms on data to make predictions. Key types: supervised, unsupervised, reinforcement learning.",
                "category": "AI"
            },
            {
                "id": "doc3",
                "title": "Database Optimization",
                "content": "Use indexes, normalize data, optimize queries, monitor performance.",
                "category": "databases"
            }
        ]
    
    async def start_session(self, user_id: str) -> str:
        """Start a new conversation session"""
        session_id = f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
        
        await self.memory.start_session(
            session_id=session_id,
            user_id=user_id,
            task="Q&A with documents",
            domain="technical_help"
        )
        
        print(f"🚀 Started session: {session_id}")
        return session_id
    
    async def query(self, question: str, user_id: str, session_id: str) -> Dict[str, Any]:
        """Process a query with memory enhancement"""
        
        print(f"\n❓ Question: {question}")
        
        # 1. Get memory context about user and previous conversations
        memory_context = await self.memory.get_context_for_llm(
            query=question,
            user_id=user_id,
            session_id=session_id,
            max_context_length=500
        )
        
        # 2. Search documents (simple keyword matching)
        relevant_docs = self._search_documents(question)
        
        # 3. Get user preferences from memory
        user_prefs = await self._get_user_preferences(user_id)
        
        # 4. Generate response
        response = await self._generate_response(
            question, relevant_docs, memory_context, user_prefs
        )
        
        # 5. Store this interaction for learning
        await self._store_interaction(question, response, relevant_docs, user_id, session_id)
        
        return {
            "response": response,
            "sources": [doc["title"] for doc in relevant_docs],
            "memory_used": bool(memory_context)
        }
    
    def _search_documents(self, query: str) -> List[Dict]:
        """Simple document search (replace with vector search)"""
        query_words = query.lower().split()
        scored_docs = []
        
        for doc in self.documents:
            score = 0
            content_lower = doc["content"].lower()
            title_lower = doc["title"].lower()
            
            for word in query_words:
                if word in content_lower:
                    score += 1
                if word in title_lower:
                    score += 2
            
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by relevance and return top 2
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:2]]
    
    async def _get_user_preferences(self, user_id: str) -> Dict[str, str]:
        """Extract user preferences from memory"""
        try:
            prefs_memories = await self.memory.search_memories(
                query=f"user {user_id} preferences style level",
                user_id=user_id,
                memory_types=["social"],
                limit=5
            )
            
            # Default preferences
            prefs = {"level": "intermediate", "style": "detailed"}
            
            # Extract from memories
            for memory in prefs_memories:
                content = memory["content"].lower()
                if "beginner" in content:
                    prefs["level"] = "beginner"
                elif "advanced" in content:
                    prefs["level"] = "advanced"
                if "brief" in content:
                    prefs["style"] = "brief"
            
            return prefs
            
        except Exception as e:
            print(f"⚠️ Error getting preferences: {e}")
            return {"level": "intermediate", "style": "detailed"}
    
    async def _generate_response(
        self, 
        question: str, 
        docs: List[Dict], 
        memory_context: str, 
        user_prefs: Dict[str, str]
    ) -> str:
        """Generate response based on docs and memory"""
        
        response_parts = []
        
        # Add memory-based greeting
        if memory_context and "conversation" in memory_context:
            response_parts.append("Based on our previous discussions,")
        
        # Adapt to user level
        if user_prefs["level"] == "beginner":
            response_parts.append("let me explain this simply:")
        elif user_prefs["level"] == "advanced":
            response_parts.append("here's a detailed technical explanation:")
        else:
            response_parts.append("here's what you need to know:")
        
        # Add document content
        if docs:
            main_content = docs[0]["content"]
            if user_prefs["style"] == "brief":
                main_content = main_content[:100] + "..."
            response_parts.append(f"\n\n{main_content}")
            
            if len(docs) > 1:
                response_parts.append(f"\n\nAlso relevant: {docs[1]['content'][:80]}...")
        else:
            response_parts.append("\n\nI don't have specific information about that, but I'll remember your question.")
        
        return " ".join(response_parts)
    
    async def _store_interaction(
        self, 
        question: str, 
        response: str, 
        docs: List[Dict], 
        user_id: str, 
        session_id: str
    ):
        """Store interaction for future learning"""
        try:
            # Store conversation
            conversation = f"User asked: '{question}'. Discussed: {', '.join([doc['title'] for doc in docs])}"
            await self.memory.create_episodic_memory(
                content=conversation,
                participants=[user_id, "assistant"],
                user_id=user_id,
                session_id=session_id
            )
            
            # Store topic interest
            if docs:
                topic = f"User interested in {docs[0]['category']} topics"
                await self.memory.create_memory(
                    content=topic,
                    memory_type="semantic",
                    user_id=user_id,
                    session_id=session_id
                )
            
            print("💾 Stored interaction in memory")
            
        except Exception as e:
            print(f"⚠️ Error storing interaction: {e}")
    
    async def end_session(self, session_id: str):
        """End the session"""
        await self.memory.end_session(session_id)
        print(f"🏁 Ended session: {session_id}")

async def demo():
    """Demonstrate the memory-enhanced RAG system"""
    
    print("="*50)
    print("🤖 Simple RAG + NeuronMemory Demo")
    print("="*50)
    
    rag = SimpleRAGWithMemory()
    user_id = "demo_user"
    
    # Start session
    session_id = await rag.start_session(user_id)
    
    try:
        # Demo conversation
        questions = [
            "What are Python best practices?",
            "Tell me about machine learning",
            "How do I optimize databases?",
            "Can you give me more Python tips?"  # This will use memory from first question
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*30}")
            print(f"Question {i}")
            print(f"{'='*30}")
            
            result = await rag.query(question, user_id, session_id)
            
            print(f"🤖 {result['response'][:150]}...")
            print(f"📚 Sources: {', '.join(result['sources'])}")
            print(f"🧠 Memory used: {result['memory_used']}")
            
            # Simulate learning user preferences
            if i == 2:
                await rag.memory.create_social_memory(
                    content="User prefers detailed technical explanations",
                    person_id=user_id,
                    relationship_type="user", 
                    user_id=user_id,
                    session_id=session_id
                )
                print("💡 Learned: User likes detailed explanations")
        
        # Show what was learned
        print(f"\n{'='*30}")
        print("📊 What I learned about you:")
        print(f"{'='*30}")
        
        user_memories = await rag.memory.search_memories(
            query=f"user {user_id} preferences interests",
            user_id=user_id,
            limit=5
        )
        
        for memory in user_memories:
            print(f"- {memory['content']}")
    
    finally:
        await rag.end_session(session_id)

if __name__ == "__main__":
    asyncio.run(demo()) 
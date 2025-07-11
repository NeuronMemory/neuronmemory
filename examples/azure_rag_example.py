"""
Advanced RAG + NeuronMemory Integration with Azure OpenAI
Complete working example using Azure OpenAI for response generation
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any
from neuron_memory import NeuronMemoryAPI
from neuron_memory.llm.azure_openai_client import AzureOpenAIClient

class AzureRAGWithMemory:
    """Advanced RAG system using Azure OpenAI and NeuronMemory"""
    
    def __init__(self):
        self.memory = NeuronMemoryAPI()
        self.llm = AzureOpenAIClient()  # Uses your Azure OpenAI config
        
        # Sample document store (replace with your vector DB)
        self.documents = [
            {
                "id": "doc1",
                "title": "Python Best Practices",
                "content": "Python best practices include: 1) Use virtual environments to isolate dependencies. 2) Follow PEP 8 style guide for code formatting. 3) Write comprehensive docstrings and type hints. 4) Implement proper error handling with try-except blocks. 5) Use list comprehensions and generator expressions for efficiency. 6) Write unit tests for all functions. 7) Use version control with meaningful commit messages.",
                "category": "programming",
                "tags": ["python", "coding", "best-practices"]
            },
            {
                "id": "doc2", 
                "title": "Machine Learning Fundamentals",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. Key types include: Supervised Learning (classification and regression), Unsupervised Learning (clustering and dimensionality reduction), and Reinforcement Learning (learning through trial and error). Common algorithms include linear regression, decision trees, neural networks, and support vector machines.",
                "category": "AI",
                "tags": ["machine-learning", "AI", "algorithms"]
            },
            {
                "id": "doc3",
                "title": "Database Optimization Techniques", 
                "content": "Database optimization involves several strategies: 1) Create appropriate indexes on frequently queried columns. 2) Normalize your database design to reduce redundancy. 3) Optimize SQL queries by avoiding SELECT *, using JOINs efficiently, and limiting result sets. 4) Monitor query performance and identify slow queries. 5) Consider database partitioning for large tables. 6) Use connection pooling to manage database connections efficiently.",
                "category": "databases",
                "tags": ["database", "optimization", "performance"]
            },
            {
                "id": "doc4",
                "title": "RESTful API Design Principles",
                "content": "RESTful API design follows these principles: 1) Use HTTP methods correctly (GET, POST, PUT, DELETE). 2) Design intuitive URL structures with nouns, not verbs. 3) Return appropriate HTTP status codes. 4) Implement proper error handling and meaningful error messages. 5) Use JSON for data exchange. 6) Implement authentication and authorization. 7) Version your APIs properly. 8) Document your APIs thoroughly.",
                "category": "web-development",
                "tags": ["API", "REST", "web-development"]
            }
        ]
    
    async def start_session(self, user_id: str, topic: str = None) -> str:
        """Start a new conversation session"""
        session_id = f"azure_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
        
        await self.memory.start_session(
            session_id=session_id,
            user_id=user_id,
            task="Technical Q&A with Azure OpenAI",
            domain=topic or "general_technical"
        )
        
        print(f"🚀 Started Azure RAG session: {session_id}")
        return session_id
    
    async def query(self, question: str, user_id: str, session_id: str) -> Dict[str, Any]:
        """Process a query with Azure OpenAI and memory enhancement"""
        
        print(f"\n❓ Question: {question}")
        
        # 1. Get memory context about user and previous conversations
        memory_context = await self.memory.get_context_for_llm(
            query=question,
            user_id=user_id,
            session_id=session_id,
            max_context_length=800
        )
        
        # 2. Expand query based on memory context
        expanded_query = await self._expand_query_with_memory(question, memory_context)
        
        # 3. Search documents
        relevant_docs = self._search_documents(expanded_query)
        
        # 4. Get user preferences from memory
        user_prefs = await self._get_user_preferences(user_id)
        
        # 5. Generate response using Azure OpenAI
        response = await self._generate_azure_response(
            question, relevant_docs, memory_context, user_prefs
        )
        
        # 6. Store interaction for learning
        await self._store_interaction(question, response, relevant_docs, user_id, session_id)
        
        return {
            "response": response,
            "sources": [{"title": doc["title"], "category": doc["category"]} for doc in relevant_docs],
            "memory_used": bool(memory_context),
            "query_expanded": expanded_query != question,
            "expanded_query": expanded_query if expanded_query != question else None
        }
    
    async def _expand_query_with_memory(self, original_query: str, memory_context: str) -> str:
        """Use Azure OpenAI to expand query based on memory context"""
        if not memory_context:
            return original_query
        
        try:
            prompt = f"""Based on this conversation history and user context:
{memory_context}

Expand this search query to be more specific and relevant based on the user's interests and previous discussions:
"{original_query}"

Return only the expanded query, no explanation:"""
            
            messages = [{"role": "user", "content": prompt}]
            expanded = await self.llm.chat_completion(messages, temperature=0.3, max_tokens=100)
            
            # Clean up the response
            expanded = expanded.strip().strip('"\'')
            return expanded if expanded else original_query
            
        except Exception as e:
            print(f"⚠️ Error expanding query: {e}")
            return original_query
    
    def _search_documents(self, query: str) -> List[Dict]:
        """Enhanced document search with tag matching"""
        query_words = query.lower().split()
        scored_docs = []
        
        for doc in self.documents:
            score = 0
            content_lower = doc["content"].lower()
            title_lower = doc["title"].lower()
            category_lower = doc["category"].lower()
            tags_lower = [tag.lower() for tag in doc["tags"]]
            
            for word in query_words:
                # Content matching
                if word in content_lower:
                    score += 1
                # Title matching (higher weight)
                if word in title_lower:
                    score += 3
                # Category matching
                if word in category_lower:
                    score += 2
                # Tag matching (high weight)
                for tag in tags_lower:
                    if word in tag or tag in word:
                        score += 4
            
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by relevance and return top 3
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:3]]
    
    async def _get_user_preferences(self, user_id: str) -> Dict[str, str]:
        """Extract user preferences from memory"""
        try:
            prefs_memories = await self.memory.search_memories(
                query=f"user {user_id} preferences style level technical",
                user_id=user_id,
                memory_types=["social"],
                limit=10
            )
            
            # Default preferences
            prefs = {
                "level": "intermediate", 
                "style": "detailed",
                "format": "explanatory"
            }
            
            # Extract preferences from memories
            for memory in prefs_memories:
                content = memory["content"].lower()
                if "beginner" in content or "simple" in content:
                    prefs["level"] = "beginner"
                elif "advanced" in content or "expert" in content:
                    prefs["level"] = "advanced"
                    
                if "brief" in content or "concise" in content:
                    prefs["style"] = "brief"
                elif "detailed" in content or "comprehensive" in content:
                    prefs["style"] = "detailed"
                    
                if "examples" in content:
                    prefs["format"] = "with_examples"
            
            return prefs
            
        except Exception as e:
            print(f"⚠️ Error getting user preferences: {e}")
            return {"level": "intermediate", "style": "detailed", "format": "explanatory"}
    
    async def _generate_azure_response(
        self, 
        question: str, 
        docs: List[Dict], 
        memory_context: str, 
        user_prefs: Dict[str, str]
    ) -> str:
        """Generate response using Azure OpenAI with context and preferences"""
        
        # Build document context
        doc_context = ""
        if docs:
            doc_context = "\n\n".join([
                f"**{doc['title']}** ({doc['category']}):\n{doc['content']}" 
                for doc in docs
            ])
        
        # Build the prompt based on user preferences
        level_instruction = {
            "beginner": "Explain concepts in simple terms, avoid technical jargon, and provide analogies when helpful.",
            "intermediate": "Use standard technical language with clear explanations of complex concepts.",
            "advanced": "Feel free to use advanced terminology and dive deep into technical details."
        }[user_prefs["level"]]
        
        style_instruction = {
            "brief": "Keep the response concise and to the point.",
            "detailed": "Provide a comprehensive explanation with context and background."
        }[user_prefs["style"]]
        
        format_instruction = {
            "with_examples": "Include practical examples and code snippets where appropriate.",
            "explanatory": "Focus on clear explanations and conceptual understanding."
        }[user_prefs["format"]]
        
        # Memory context integration
        memory_instruction = ""
        if memory_context:
            memory_instruction = f"""
Previous conversation context:
{memory_context}

Reference our previous discussions when relevant and build upon what we've already covered."""
        
        prompt = f"""You are a helpful technical assistant. Answer the user's question based on the provided information and context.

{memory_instruction}

User Question: {question}

Available Information:
{doc_context if doc_context else "No specific documents found, provide general knowledge."}

Instructions:
- {level_instruction}
- {style_instruction}
- {format_instruction}
- Be conversational and helpful
- If referencing previous conversations, acknowledge the continuity naturally
- If the question is outside the provided information, say so and provide general guidance

Response:"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.chat_completion(
                messages, 
                temperature=0.7, 
                max_tokens=800
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"⚠️ Error generating Azure OpenAI response: {e}")
            # Fallback response
            if docs:
                return f"Based on the available information about {docs[0]['title']}: {docs[0]['content'][:200]}..."
            else:
                return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
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
            # Store conversation as episodic memory
            doc_titles = [doc["title"] for doc in docs] if docs else ["general knowledge"]
            conversation = f"User asked: '{question}'. I provided information about: {', '.join(doc_titles)}."
            
            await self.memory.create_episodic_memory(
                content=conversation,
                participants=[user_id, "azure_assistant"],
                location="azure_rag_conversation",
                user_id=user_id,
                session_id=session_id
            )
            
            # Store topic interests
            for doc in docs:
                topic_interest = f"User showed interest in {doc['category']} - specifically {doc['title']}"
                await self.memory.create_memory(
                    content=topic_interest,
                    memory_type="semantic",
                    user_id=user_id,
                    session_id=session_id
                )
            
            print("💾 Stored interaction in memory")
            
        except Exception as e:
            print(f"⚠️ Error storing interaction: {e}")
    
    async def get_personalized_suggestions(self, user_id: str) -> List[str]:
        """Generate personalized topic suggestions based on user's interests"""
        try:
            interests = await self.memory.search_memories(
                query=f"user {user_id} interested topics categories",
                user_id=user_id,
                memory_types=["semantic"],
                limit=10
            )
            
            if not interests:
                return [
                    "What are Python best practices?",
                    "How does machine learning work?", 
                    "What are database optimization techniques?"
                ]
            
            # Extract topics from interests
            topics = set()
            for interest in interests:
                content = interest["content"].lower()
                if "programming" in content or "python" in content:
                    topics.add("advanced Python techniques")
                if "ai" in content or "machine learning" in content:
                    topics.add("machine learning algorithms")
                if "database" in content:
                    topics.add("database design patterns")
                if "web" in content or "api" in content:
                    topics.add("API design best practices")
            
            suggestions = [f"Tell me more about {topic}" for topic in list(topics)[:3]]
            return suggestions if suggestions else [
                "What programming concepts should I learn next?",
                "How can I improve my technical skills?",
                "What are current best practices in software development?"
            ]
            
        except Exception as e:
            print(f"⚠️ Error generating suggestions: {e}")
            return ["How can I help you today?"]
    
    async def end_session(self, session_id: str):
        """End the session"""
        await self.memory.end_session(session_id)
        print(f"🏁 Ended Azure RAG session: {session_id}")

async def demo_azure_rag():
    """Demonstrate the Azure OpenAI + RAG system with memory"""
    
    print("="*60)
    print("🤖 Azure OpenAI RAG + NeuronMemory Demo")
    print("="*60)
    
    rag = AzureRAGWithMemory()
    user_id = "azure_demo_user"
    
    # Start session
    session_id = await rag.start_session(user_id, topic="software_development")
    
    try:
        # Demo conversation showing memory and personalization
        questions = [
            "What are Python best practices?",
            "Tell me about machine learning basics", 
            "How do I optimize database performance?",
            "Can you give me more advanced Python techniques?"  # This will reference the first question
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*40}")
            print(f"Question {i}/{len(questions)}")
            print(f"{'='*40}")
            
            result = await rag.query(question, user_id, session_id)
            
            print(f"🤖 Response: {result['response'][:200]}...")
            print(f"📚 Sources: {', '.join([s['title'] for s in result['sources']])}")
            print(f"🧠 Memory used: {result['memory_used']}")
            print(f"🔍 Query expanded: {result['query_expanded']}")
            if result['expanded_query']:
                print(f"   Expanded to: {result['expanded_query']}")
            
            # Learn user preferences after second question
            if i == 2:
                await rag.memory.create_social_memory(
                    content="User prefers detailed technical explanations with practical examples",
                    person_id=user_id,
                    relationship_type="user", 
                    user_id=user_id,
                    session_id=session_id
                )
                print("💡 Learned: User likes detailed explanations with examples")
        
        # Show personalized suggestions
        print(f"\n{'='*40}")
        print("🎯 Personalized Suggestions")
        print(f"{'='*40}")
        
        suggestions = await rag.get_personalized_suggestions(user_id)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
        
        # Show memory statistics
        print(f"\n{'='*40}")
        print("📊 Memory System Statistics")
        print(f"{'='*40}")
        
        stats = await rag.memory.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    finally:
        await rag.end_session(session_id)

if __name__ == "__main__":
    print("Make sure your Azure OpenAI credentials are configured!")
    print("This demo uses your Azure OpenAI API for response generation.")
    print("\nStarting demo in 3 seconds...")
    
    import time
    time.sleep(3)
    
    asyncio.run(demo_azure_rag()) 
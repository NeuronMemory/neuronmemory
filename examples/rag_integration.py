"""
RAG Integration Example

This example demonstrates how to integrate NeuronMemory with RAG (Retrieval-Augmented Generation) systems:
- Memory-enhanced query expansion
- Context-aware document retrieval  
- User preference learning from interactions
- Conversation history for better responses
- Personalized search result ranking

Shows intermediate-level integration patterns for production RAG systems.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from neuron_memory import NeuronMemoryAPI

class RAGWithMemoryIntegration:
    """RAG system enhanced with NeuronMemory for personalization and context"""
    
    def __init__(self):
        self.memory = NeuronMemoryAPI()
        self.current_session = None
        
        # Sample document store (replace with your vector database)
        self.documents = [
            {
                "id": "doc1",
                "title": "Python Best Practices",
                "content": "Python best practices include using virtual environments, following PEP 8 style guide, writing comprehensive docstrings, implementing proper error handling, using list comprehensions for efficiency, writing unit tests, and using meaningful variable names.",
                "category": "programming",
                "tags": ["python", "coding", "best-practices", "development"],
                "difficulty": "intermediate"
            },
            {
                "id": "doc2", 
                "title": "Machine Learning Fundamentals",
                "content": "Machine learning is a subset of AI that enables computers to learn from data. Key types include supervised learning (classification/regression), unsupervised learning (clustering), and reinforcement learning. Common algorithms include linear regression, decision trees, and neural networks.",
                "category": "AI",
                "tags": ["machine-learning", "AI", "algorithms", "data-science"],
                "difficulty": "beginner"
            },
            {
                "id": "doc3",
                "title": "Advanced Database Optimization",
                "content": "Database optimization involves indexing strategies, query optimization, normalization, partitioning, connection pooling, and monitoring performance. Key techniques include avoiding SELECT *, using JOINs efficiently, and implementing proper caching strategies.",
                "category": "databases",
                "tags": ["database", "optimization", "performance", "SQL"],
                "difficulty": "advanced"
            },
            {
                "id": "doc4",
                "title": "RESTful API Design",
                "content": "RESTful API design follows HTTP methods (GET, POST, PUT, DELETE), uses resource-based URLs, returns appropriate status codes, implements proper error handling, uses JSON for data exchange, and includes authentication/authorization.",
                "category": "web-development",
                "tags": ["API", "REST", "web-development", "backend"],
                "difficulty": "intermediate"
            },
            {
                "id": "doc5",
                "title": "Microservices Architecture",
                "content": "Microservices architecture breaks applications into small, independent services that communicate via APIs. Benefits include scalability, technology diversity, and fault isolation. Challenges include complexity, data consistency, and network latency.",
                "category": "architecture",
                "tags": ["microservices", "architecture", "scalability", "distributed-systems"],
                "difficulty": "advanced"
            }
        ]
    
    async def start_rag_session(self, user_id: str, domain: str = "general") -> str:
        """Start a new RAG session with memory tracking"""
        self.current_session = f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
        
        await self.memory.start_session(
            session_id=self.current_session,
            user_id=user_id,
            task=f"RAG-enhanced Q&A session - {domain}",
            domain=domain
        )
        
        # Get user profile for personalized greeting
        user_profile = await self._get_user_profile(user_id)
        greeting = await self._create_personalized_greeting(user_profile)
        
        print(f"🚀 Started RAG session: {self.current_session}")
        return greeting
    
    async def query_with_memory(self, question: str, user_id: str) -> Dict[str, Any]:
        """Process query with memory-enhanced RAG"""
        
        print(f"\n❓ User Question: {question}")
        
        # Step 1: Get memory context about user and previous conversations
        memory_context = await self.memory.get_context_for_llm(
            query=question,
            user_id=user_id,
            session_id=self.current_session,
            max_context_length=1000
        )
        
        # Step 2: Expand query based on user's previous interests and context
        expanded_query = await self._expand_query_with_memory(question, memory_context, user_id)
        
        # Step 3: Get user preferences for search personalization
        user_preferences = await self._get_user_preferences(user_id)
        
        # Step 4: Search documents with personalized ranking
        relevant_docs = await self._search_documents_personalized(
            expanded_query, user_preferences, user_id
        )
        
        # Step 5: Generate memory-informed response
        response = await self._generate_memory_informed_response(
            question, relevant_docs, memory_context, user_preferences
        )
        
        # Step 6: Learn from interaction for future personalization
        await self._learn_from_interaction(question, response, relevant_docs, user_id)
        
        return {
            "response": response,
            "sources": [{"title": doc["title"], "category": doc["category"]} for doc in relevant_docs],
            "original_query": question,
            "expanded_query": expanded_query if expanded_query != question else None,
            "memory_context_used": bool(memory_context),
            "personalization_applied": bool(user_preferences),
            "context_length": len(memory_context) if memory_context else 0
        }
    
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile from memory"""
        try:
            profile_memories = await self.memory.search_memories(
                query=f"user {user_id} profile preferences experience level",
                user_id=user_id,
                memory_types=["social", "semantic"],
                limit=10
            )
            
            profile = {
                "experience_level": "intermediate",
                "preferred_topics": [],
                "communication_style": "detailed",
                "learning_pace": "normal"
            }
            
            for memory in profile_memories:
                content = memory["content"].lower()
                if "beginner" in content:
                    profile["experience_level"] = "beginner"
                elif "advanced" in content or "expert" in content:
                    profile["experience_level"] = "advanced"
                    
                if "brief" in content or "concise" in content:
                    profile["communication_style"] = "brief"
                elif "detailed" in content:
                    profile["communication_style"] = "detailed"
            
            return profile
            
        except Exception as e:
            print(f"⚠️ Error getting user profile: {e}")
            return {"experience_level": "intermediate", "preferred_topics": [], "communication_style": "detailed"}
    
    async def _create_personalized_greeting(self, user_profile: Dict) -> str:
        """Create personalized greeting based on user profile"""
        
        level = user_profile.get("experience_level", "intermediate")
        style = user_profile.get("communication_style", "detailed")
        
        greetings = {
            ("beginner", "detailed"): "Hello! I'm here to help you learn with clear, step-by-step explanations. What would you like to explore?",
            ("beginner", "brief"): "Hi! I'll keep my answers simple and concise. What can I help you with?",
            ("intermediate", "detailed"): "Hello! I can provide comprehensive answers with good depth. What technical topic interests you?",
            ("intermediate", "brief"): "Hi! I'll give you focused, practical answers. What's your question?",
            ("advanced", "detailed"): "Hello! I can dive deep into complex topics and advanced concepts. What challenging question do you have?",
            ("advanced", "brief"): "Hi! I'll give you precise, technical answers. What do you need to know?"
        }
        
        return greetings.get((level, style), "Hello! How can I help you today?")
    
    async def _expand_query_with_memory(self, original_query: str, memory_context: str, user_id: str) -> str:
        """Expand query based on memory context and user history"""
        
        if not memory_context:
            return original_query
        
        # Simple query expansion based on memory context
        # In production, you'd use an LLM for this
        
        expanded_terms = []
        original_lower = original_query.lower()
        
        # Add related terms from memory context
        if "python" in original_lower and "programming" in memory_context.lower():
            expanded_terms.append("programming")
        
        if "database" in original_lower and "optimization" in memory_context.lower():
            expanded_terms.append("performance optimization")
        
        if "api" in original_lower and "REST" in memory_context:
            expanded_terms.append("RESTful design")
        
        # Get user's previously discussed topics
        try:
            topic_memories = await self.memory.search_memories(
                query=f"user {user_id} discussed topics interests",
                user_id=user_id,
                memory_types=["semantic"],
                limit=5
            )
            
            for memory in topic_memories:
                content = memory["content"].lower()
                if any(term in content for term in ["machine learning", "AI"]) and "learn" in original_lower:
                    expanded_terms.append("machine learning")
                if "microservices" in content and "architecture" in original_lower:
                    expanded_terms.append("microservices")
        
        except Exception:
            pass
        
        if expanded_terms:
            return f"{original_query} {' '.join(expanded_terms)}"
        
        return original_query
    
    async def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences for search personalization"""
        try:
            pref_memories = await self.memory.search_memories(
                query=f"user {user_id} prefers likes interests difficulty",
                user_id=user_id,
                memory_types=["social", "semantic"],
                limit=10
            )
            
            preferences = {
                "preferred_difficulty": "intermediate",
                "favorite_categories": [],
                "learning_style": "practical",
                "topic_interests": []
            }
            
            for memory in pref_memories:
                content = memory["content"].lower()
                
                # Extract difficulty preferences
                if "beginner" in content or "simple" in content:
                    preferences["preferred_difficulty"] = "beginner"
                elif "advanced" in content or "complex" in content:
                    preferences["preferred_difficulty"] = "advanced"
                
                # Extract category interests
                categories = ["programming", "AI", "databases", "web-development", "architecture"]
                for category in categories:
                    if category.replace("-", " ") in content or category in content:
                        if category not in preferences["favorite_categories"]:
                            preferences["favorite_categories"].append(category)
                
                # Extract learning style
                if "example" in content or "practical" in content:
                    preferences["learning_style"] = "practical"
                elif "theory" in content or "conceptual" in content:
                    preferences["learning_style"] = "theoretical"
            
            return preferences
            
        except Exception as e:
            print(f"⚠️ Error getting user preferences: {e}")
            return {"preferred_difficulty": "intermediate", "favorite_categories": [], "learning_style": "practical"}
    
    async def _search_documents_personalized(
        self, 
        query: str, 
        user_preferences: Dict, 
        user_id: str
    ) -> List[Dict]:
        """Search documents with personalized ranking"""
        
        query_words = query.lower().split()
        scored_docs = []
        
        for doc in self.documents:
            score = 0
            content_lower = doc["content"].lower()
            title_lower = doc["title"].lower()
            
            # Base relevance scoring
            for word in query_words:
                if word in content_lower:
                    score += 1
                if word in title_lower:
                    score += 3
                if word in doc["category"].lower():
                    score += 2
                for tag in doc["tags"]:
                    if word in tag.lower():
                        score += 2
            
            # Personalization boosts
            if score > 0:  # Only boost if document is already relevant
                
                # Difficulty preference boost
                if doc["difficulty"] == user_preferences.get("preferred_difficulty", "intermediate"):
                    score += 2
                
                # Category preference boost
                if doc["category"] in user_preferences.get("favorite_categories", []):
                    score += 3
                
                # Learning style boost
                learning_style = user_preferences.get("learning_style", "practical")
                if learning_style == "practical" and any(word in content_lower for word in ["example", "step", "how"]):
                    score += 1
                elif learning_style == "theoretical" and any(word in content_lower for word in ["concept", "theory", "principle"]):
                    score += 1
            
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by relevance and return top results
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:3]]
    
    async def _generate_memory_informed_response(
        self, 
        question: str, 
        docs: List[Dict], 
        memory_context: str, 
        user_preferences: Dict
    ) -> str:
        """Generate response informed by memory and user preferences"""
        
        # This is simplified - in production use an LLM with the context
        
        if not docs:
            return "I couldn't find specific information about that topic. Could you provide more details or try a related question?"
        
        primary_doc = docs[0]
        response = f"Based on the information about {primary_doc['title']}:\n\n"
        response += primary_doc['content'][:200] + "..."
        
        # Add personalization based on user preferences
        difficulty = user_preferences.get("preferred_difficulty", "intermediate")
        learning_style = user_preferences.get("learning_style", "practical")
        
        if difficulty == "beginner":
            response += "\n\nLet me break this down in simpler terms if you'd like more explanation."
        elif difficulty == "advanced":
            response += "\n\nI can provide more advanced details or dive deeper into specific aspects if you're interested."
        
        if learning_style == "practical":
            response += "\n\nWould you like some practical examples or step-by-step guidance?"
        elif learning_style == "theoretical":
            response += "\n\nI can explain the underlying concepts and principles in more detail if you'd like."
        
        # Reference memory context if relevant
        if memory_context and any(topic in memory_context.lower() for topic in ["discussed", "mentioned", "talked"]):
            response += "\n\nBuilding on what we've discussed before, this connects to your previous interests."
        
        # Suggest related topics
        if len(docs) > 1:
            related_topics = [doc["title"] for doc in docs[1:]]
            response += f"\n\nRelated topics you might find interesting: {', '.join(related_topics)}"
        
        return response
    
    async def _learn_from_interaction(
        self, 
        question: str, 
        response: str, 
        docs: List[Dict], 
        user_id: str
    ):
        """Learn from user interaction for future personalization"""
        try:
            # Store the conversation as episodic memory
            conversation = f"User asked: '{question}'. Provided information about: {', '.join([doc['title'] for doc in docs])}"
            
            await self.memory.create_episodic_memory(
                content=conversation,
                participants=[user_id, "rag_assistant"],
                location="rag_chat_session",
                emotional_state="helpful",
                user_id=user_id,
                session_id=self.current_session
            )
            
            # Store topic interests
            for doc in docs:
                interest_memory = f"User showed interest in {doc['category']} topic: {doc['title']}"
                await self.memory.create_semantic_memory(
                    content=interest_memory,
                    domain="user_interests",
                    confidence=0.8,
                    user_id=user_id,
                    session_id=self.current_session
                )
            
            # Store difficulty level signals
            question_lower = question.lower()
            if any(word in question_lower for word in ["simple", "basic", "beginner"]):
                await self.memory.create_social_memory(
                    content="User prefers beginner-level explanations",
                    person_id=user_id,
                    relationship_type="user",
                    user_id=user_id,
                    session_id=self.current_session
                )
            elif any(word in question_lower for word in ["advanced", "complex", "detailed"]):
                await self.memory.create_social_memory(
                    content="User prefers advanced technical details",
                    person_id=user_id,
                    relationship_type="user",
                    user_id=user_id,
                    session_id=self.current_session
                )
            
            print("💾 Learned from interaction and updated user model")
            
        except Exception as e:
            print(f"⚠️ Error learning from interaction: {e}")
    
    async def get_personalized_suggestions(self, user_id: str) -> List[str]:
        """Generate personalized topic suggestions"""
        try:
            interests = await self.memory.search_memories(
                query=f"user {user_id} interested topics categories",
                user_id=user_id,
                memory_types=["semantic"],
                limit=10
            )
            
            if not interests:
                return [
                    "What programming concepts would you like to learn?",
                    "Are you interested in machine learning basics?",
                    "Would you like to know about database optimization?"
                ]
            
            suggestions = []
            seen_categories = set()
            
            for interest in interests:
                content = interest["content"].lower()
                if "programming" in content and "programming" not in seen_categories:
                    suggestions.append("What advanced Python techniques interest you?")
                    seen_categories.add("programming")
                elif "ai" in content or "machine learning" in content and "ai" not in seen_categories:
                    suggestions.append("Would you like to learn about neural networks?")
                    seen_categories.add("ai")
                elif "database" in content and "database" not in seen_categories:
                    suggestions.append("Interested in database scaling strategies?")
                    seen_categories.add("database")
            
            return suggestions[:3] if suggestions else [
                "What technical topic can I help you explore today?"
            ]
            
        except Exception as e:
            print(f"⚠️ Error generating suggestions: {e}")
            return ["How can I help you learn something new today?"]
    
    async def end_rag_session(self, user_id: str):
        """End the RAG session"""
        if self.current_session:
            # Store session summary
            await self.memory.create_episodic_memory(
                content="RAG session completed successfully with memory-enhanced personalization",
                participants=[user_id, "rag_assistant"],
                location="rag_chat_session",
                emotional_state="completed",
                user_id=user_id,
                session_id=self.current_session
            )
            
            await self.memory.end_session(self.current_session)
            print(f"🏁 Ended RAG session: {self.current_session}")
            self.current_session = None

async def demo_rag_integration():
    """Demonstrate RAG integration with NeuronMemory"""
    
    print("="*70)
    print("🔍 RAG + NeuronMemory Integration Demo")
    print("="*70)
    
    rag = RAGWithMemoryIntegration()
    user_id = "rag_demo_user"
    
    try:
        # Start session
        greeting = await rag.start_rag_session(user_id, "software_development")
        print(f"Assistant: {greeting}")
        
        # Demo conversation showing learning and personalization
        questions = [
            "What are Python best practices?",
            "Tell me about machine learning basics",
            "I need help with advanced database optimization",
            "Can you suggest more Python topics?"  # This should be personalized
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*50}")
            print(f"Query {i}/{len(questions)}")
            print(f"{'='*50}")
            
            result = await rag.query_with_memory(question, user_id)
            
            print(f"🤖 Response: {result['response'][:300]}...")
            print(f"📚 Sources: {', '.join([s['title'] for s in result['sources']])}")
            print(f"🔍 Query expanded: {result['expanded_query'] is not None}")
            if result['expanded_query']:
                print(f"   From: '{result['original_query']}'")
                print(f"   To: '{result['expanded_query']}'")
            print(f"🧠 Memory used: {result['memory_context_used']}")
            print(f"🎯 Personalization: {result['personalization_applied']}")
            
            # Add some learning signals after the second query
            if i == 2:
                await rag.memory.create_social_memory(
                    content="User prefers advanced technical explanations with detailed examples",
                    person_id=user_id,
                    relationship_type="user",
                    user_id=user_id,
                    session_id=rag.current_session
                )
                print("💡 System learned: User prefers advanced explanations")
        
        # Show personalized suggestions
        print(f"\n{'='*50}")
        print("🎯 Personalized Suggestions")
        print(f"{'='*50}")
        
        suggestions = await rag.get_personalized_suggestions(user_id)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
        
        # Show memory insights
        print(f"\n{'='*50}")
        print("📊 Memory-Based Insights")
        print(f"{'='*50}")
        
        stats = await rag.memory.get_statistics()
        print(f"Total memories created: {stats.get('total_memories', 'N/A')}")
        print(f"Session memories: {stats.get('session_memories', 'N/A')}")
        print(f"User profile enriched through interactions")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await rag.end_rag_session(user_id)
    
    print("\n" + "="*70)
    print("✅ RAG Integration Demo Complete!")
    print("="*70)
    print("Key features demonstrated:")
    print("• Memory-enhanced query expansion")
    print("• Personalized document ranking")
    print("• User preference learning")
    print("• Context-aware responses")
    print("• Conversation continuity")
    print("• Adaptive difficulty matching")

if __name__ == "__main__":
    asyncio.run(demo_rag_integration())

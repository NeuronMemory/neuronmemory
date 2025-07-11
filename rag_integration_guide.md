# 🤖 NeuronMemory + RAG Integration Guide

## Overview
This guide shows how to enhance your RAG (Retrieval-Augmented Generation) application with NeuronMemory to create an intelligent system that remembers conversations, learns user preferences, and provides contextually aware responses.

## 🚀 Key Benefits

### Traditional RAG vs Memory-Enhanced RAG

**Traditional RAG:**
- Static document retrieval
- No conversation memory
- Same response for same query
- No user personalization

**Memory-Enhanced RAG:**
- ✅ Remembers past conversations
- ✅ Learns user preferences
- ✅ Contextual query expansion
- ✅ Personalized responses
- ✅ Improves over time

## 📋 Implementation Steps

### 1. Basic Integration Pattern

```python
from neuron_memory import NeuronMemoryAPI

class IntelligentRAG:
    def __init__(self):
        self.memory = NeuronMemoryAPI()
        self.vector_db = YourVectorDatabase()  # Pinecone, Weaviate, etc.
        self.llm = YourLLMClient()  # OpenAI, Anthropic, etc.
    
    async def query(self, user_question: str, user_id: str, session_id: str):
        # 1. Get memory context
        memory_context = await self.memory.get_context_for_llm(
            query=user_question,
            user_id=user_id,
            session_id=session_id,
            max_context_length=1000
        )
        
        # 2. Expand query based on memory
        expanded_query = await self.expand_query_with_memory(
            user_question, 
            memory_context
        )
        
        # 3. Retrieve documents
        documents = await self.vector_db.search(expanded_query, top_k=5)
        
        # 4. Get user preferences
        user_prefs = await self.get_user_preferences(user_id)
        
        # 5. Generate response with context
        response = await self.generate_response(
            question=user_question,
            documents=documents,
            memory_context=memory_context,
            user_preferences=user_prefs
        )
        
        # 6. Store interaction for learning
        await self.store_interaction(
            user_question, response, documents, user_id, session_id
        )
        
        return response
```

### 2. Memory-Enhanced Query Expansion

```python
async def expand_query_with_memory(self, query: str, memory_context: str) -> str:
    """Expand query based on conversation history and user preferences"""
    
    if not memory_context:
        return query
    
    # Use LLM to expand query based on memory
    expansion_prompt = f"""
    Based on this conversation history and user context:
    {memory_context}
    
    Expand this query to be more specific and relevant:
    "{query}"
    
    Return only the expanded query:
    """
    
    expanded = await self.llm.generate(expansion_prompt)
    return expanded.strip()
```

### 3. User Preference Learning

```python
async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
    """Extract user preferences from social memory"""
    
    user_memories = await self.memory.search_memories(
        query=f"user preferences communication style {user_id}",
        user_id=user_id,
        memory_types=["social"],
        limit=10
    )
    
    preferences = {
        "detail_level": "medium",  # low, medium, high
        "technical_level": "intermediate",  # beginner, intermediate, advanced
        "response_style": "conversational",  # formal, conversational, brief
        "preferred_topics": [],
        "learning_style": "examples"  # theory, examples, practice
    }
    
    # Analyze memories to extract preferences
    for memory in user_memories:
        content = memory["content"].lower()
        
        if "detailed explanation" in content:
            preferences["detail_level"] = "high"
        elif "brief" in content or "quick" in content:
            preferences["detail_level"] = "low"
            
        if "beginner" in content:
            preferences["technical_level"] = "beginner"
        elif "advanced" in content or "expert" in content:
            preferences["technical_level"] = "advanced"
    
    return preferences
```

### 4. Context-Aware Response Generation

```python
async def generate_response(
    self,
    question: str,
    documents: List[Document],
    memory_context: str,
    user_preferences: Dict[str, Any]
) -> str:
    """Generate personalized response using retrieved docs and memory"""
    
    # Build context for LLM
    doc_context = "\n\n".join([doc.content for doc in documents])
    
    # Adapt prompt based on user preferences
    detail_instruction = {
        "low": "Keep the response brief and to the point.",
        "medium": "Provide a balanced explanation with key details.",
        "high": "Give a comprehensive, detailed explanation."
    }[user_preferences["detail_level"]]
    
    level_instruction = {
        "beginner": "Explain in simple terms, avoid jargon.",
        "intermediate": "Use standard technical language.",
        "advanced": "Feel free to use advanced concepts and terminology."
    }[user_preferences["technical_level"]]
    
    prompt = f"""
    {memory_context}
    
    User Question: {question}
    
    Retrieved Information:
    {doc_context}
    
    Instructions:
    - {detail_instruction}
    - {level_instruction}
    - Be conversational and helpful
    - Reference previous conversation context when relevant
    
    Response:
    """
    
    return await self.llm.generate(prompt)
```

### 5. Interaction Storage for Learning

```python
async def store_interaction(
    self,
    question: str,
    response: str,
    documents: List[Document],
    user_id: str,
    session_id: str
):
    """Store interaction for future learning"""
    
    # Store conversation as episodic memory
    conversation = f"User asked: '{question}'. I provided information about {', '.join([doc.title for doc in documents])}."
    await self.memory.create_episodic_memory(
        content=conversation,
        participants=[user_id, "assistant"],
        user_id=user_id,
        session_id=session_id
    )
    
    # Store topic interests
    for doc in documents:
        topic_interest = f"User showed interest in {doc.category} - {doc.title}"
        await self.memory.create_memory(
            content=topic_interest,
            memory_type="semantic",
            user_id=user_id,
            session_id=session_id
        )
    
    # Learn about user's question patterns
    question_pattern = f"User asks {len(question.split())} word questions about technical topics"
    await self.memory.create_social_memory(
        content=question_pattern,
        person_id=user_id,
        relationship_type="user",
        user_id=user_id,
        session_id=session_id
    )
```

## 🛠️ Integration with Popular Vector Databases

### Pinecone Integration

```python
import pinecone
from neuron_memory import NeuronMemoryAPI

class PineconeRAGWithMemory:
    def __init__(self, pinecone_api_key: str, index_name: str):
        pinecone.init(api_key=pinecone_api_key)
        self.index = pinecone.Index(index_name)
        self.memory = NeuronMemoryAPI()
    
    async def search_with_memory(
        self, 
        query: str, 
        user_id: str, 
        session_id: str
    ):
        # Get memory context
        memory_context = await self.memory.get_context_for_llm(
            query=query, user_id=user_id, session_id=session_id
        )
        
        # Expand query embedding with memory
        query_embedding = await self.get_enhanced_embedding(query, memory_context)
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )
        
        return results
```

### Weaviate Integration

```python
import weaviate
from neuron_memory import NeuronMemoryAPI

class WeaviateRAGWithMemory:
    def __init__(self, weaviate_url: str):
        self.client = weaviate.Client(weaviate_url)
        self.memory = NeuronMemoryAPI()
    
    async def hybrid_search_with_memory(
        self,
        query: str,
        user_id: str,
        session_id: str
    ):
        # Get user context from memory
        user_context = await self.memory.search_memories(
            query=f"user preferences topics {query}",
            user_id=user_id,
            limit=5
        )
        
        # Build enhanced query
        context_terms = [mem["content"] for mem in user_context]
        enhanced_query = f"{query} {' '.join(context_terms)}"
        
        # Hybrid search in Weaviate
        result = (
            self.client.query
            .get("Document", ["title", "content", "category"])
            .with_hybrid(query=enhanced_query)
            .with_limit(10)
            .do()
        )
        
        return result
```

## 📊 Advanced Features

### 1. Conversation Continuity

```python
async def continue_conversation(self, user_id: str, session_id: str):
    """Continue previous conversation with context"""
    
    recent_memories = await self.memory.search_memories(
        query="recent conversation topics",
        user_id=user_id,
        session_id=session_id,
        memory_types=["episodic"],
        limit=5
    )
    
    if recent_memories:
        last_topic = recent_memories[0]["content"]
        greeting = f"Continuing our discussion about {last_topic}..."
        return greeting
    
    return "Hello! How can I help you today?"
```

### 2. Proactive Suggestions

```python
async def get_proactive_suggestions(self, user_id: str) -> List[str]:
    """Suggest questions based on user's interests"""
    
    interests = await self.memory.search_memories(
        query="user interests topics categories",
        user_id=user_id,
        memory_types=["semantic", "social"],
        limit=10
    )
    
    suggestions = []
    for interest in interests:
        # Generate follow-up questions
        suggestion = f"Would you like to learn more about {interest['content']}?"
        suggestions.append(suggestion)
    
    return suggestions[:3]
```

### 3. Learning from Feedback

```python
async def process_user_feedback(
    self,
    response_id: str,
    feedback: str,
    user_id: str,
    session_id: str
):
    """Learn from user feedback to improve responses"""
    
    if "too technical" in feedback.lower():
        await self.memory.create_social_memory(
            content="User prefers simpler, less technical explanations",
            person_id=user_id,
            relationship_type="user",
            user_id=user_id,
            session_id=session_id
        )
    
    elif "more detail" in feedback.lower():
        await self.memory.create_social_memory(
            content="User wants more detailed and comprehensive responses",
            person_id=user_id,
            relationship_type="user",
            user_id=user_id,
            session_id=session_id
        )
    
    # Store feedback as learning experience
    await self.memory.create_episodic_memory(
        content=f"User provided feedback: '{feedback}' on response {response_id}",
        user_id=user_id,
        session_id=session_id
    )
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install neuron-memory
# Add your vector database client
pip install pinecone-client  # or weaviate-client, etc.
```

### 2. Basic Setup

```python
from neuron_memory import NeuronMemoryAPI

# Initialize
memory = NeuronMemoryAPI()

# Start a session
session_id = await memory.start_session(
    session_id="user_session_001",
    user_id="user_123",
    task="document_qa",
    domain="technology"
)

# Use with your RAG pipeline
result = await your_rag_query(
    question="How do I optimize database queries?",
    user_id="user_123",
    session_id=session_id
)
```

### 3. Run the Example

```bash
python rag_example.py
```

## 📈 Performance Tips

1. **Batch Memory Operations**: Store multiple memories in batches
2. **Cache User Preferences**: Cache frequently accessed user preferences
3. **Async Operations**: Use async/await for all memory operations
4. **Memory Cleanup**: Regularly clean up old, low-importance memories
5. **Context Window Management**: Limit memory context to stay within LLM token limits

## 🔒 Privacy Considerations

- Store only necessary information in memory
- Implement user data deletion capabilities
- Use encryption for sensitive data
- Provide transparency about what's remembered
- Allow users to view and edit their memory data

This integration pattern creates a truly intelligent RAG system that learns and adapts to users over time! 
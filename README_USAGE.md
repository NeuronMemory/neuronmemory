# NeuronMemory Usage Guide

This guide explains how to use the NeuronMemory system that has been built according to the comprehensive specifications.

## 🚀 Quick Start

### 1. Set Up Environment

First, create a `.env` file in the project root with your Azure OpenAI credentials:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# NeuronMemory Configuration (Optional - defaults provided)
NEURON_MEMORY_LOG_LEVEL=INFO
NEURON_MEMORY_MAX_MEMORY_SIZE=10000
CHROMA_DB_PATH=./data/chromadb
MEMORY_STORE_PATH=./data/memory_store
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Basic Usage

```python
import asyncio
from neuron_memory import NeuronMemoryAPI

async def basic_example():
    # Initialize the API
    api = NeuronMemoryAPI()
    
    # Start a session
    session_id = "my_session_001"
    user_id = "user_123"
    
    await api.start_session(
        session_id=session_id,
        user_id=user_id,
        task="Learning Python",
        domain="Programming"
    )
    
    # Create a memory
    memory_id = await api.create_memory(
        content="Python is great for data science and machine learning",
        memory_type="semantic",
        user_id=user_id,
        session_id=session_id
    )
    
    # Search for memories
    results = await api.search_memories(
        query="Python programming",
        user_id=user_id,
        session_id=session_id,
        limit=5
    )
    
    print(f"Found {len(results)} relevant memories")
    for result in results:
        print(f"- {result['content']} (relevance: {result['relevance_score']:.2f})")
    
    # End session
    await api.end_session(session_id)

# Run the example
asyncio.run(basic_example())
```

## 📚 Memory Types

NeuronMemory supports multiple types of memories, each optimized for different use cases:

### 1. Semantic Memory (Facts & Knowledge)

```python
memory_id = await api.create_memory(
    content="The capital of France is Paris",
    memory_type="semantic",
    user_id="user_123"
)
```

### 2. Episodic Memory (Personal Experiences)

```python
memory_id = await api.create_episodic_memory(
    content="Had a great team meeting to discuss the new project roadmap",
    participants=["Alice", "Bob", "Charlie"],
    location="Conference Room B",
    emotions={"valence": 0.8, "arousal": 0.6, "dominance": 0.7},
    user_id="user_123",
    session_id="session_001"
)
```

### 3. Social Memory (Relationships & People)

```python
memory_id = await api.create_social_memory(
    content="Alice prefers morning meetings and is expert in machine learning",
    person_id="alice_456",
    relationship_type="colleague",
    user_id="user_123"
)
```

### 4. Procedural Memory (How-to Knowledge)

```python
memory_id = await api.create_memory(
    content="To deploy the app: 1) Run tests, 2) Build Docker image, 3) Push to registry, 4) Update Kubernetes",
    memory_type="procedural",
    user_id="user_123"
)
```

### 5. Working Memory (Temporary Context)

```python
memory_id = await api.create_memory(
    content="Currently debugging the authentication issue in the login module",
    memory_type="working",
    task_context="Bug fixing session",
    user_id="user_123",
    session_id="session_001"
)
```

## 🔍 Advanced Search Features

### Search with Filters

```python
# Search specific memory types
results = await api.search_memories(
    query="team collaboration",
    memory_types=["episodic", "social"],
    user_id="user_123",
    limit=10
)

# Search with minimum relevance threshold
results = await api.search_memories(
    query="important project decisions",
    min_relevance=0.7,  # Only highly relevant results
    user_id="user_123"
)
```

### Context-Aware Search

```python
# Start session with specific context
await api.start_session(
    session_id="project_session",
    user_id="user_123",
    task="Planning new feature",
    domain="Product Development"
)

# Search will be influenced by session context
results = await api.search_memories(
    query="user feedback",
    session_id="project_session",
    user_id="user_123"
)
```

## 🤖 LLM Integration

### Getting Memory Context for LLM

```python
async def llm_with_memory(user_query: str, user_id: str, session_id: str):
    api = NeuronMemoryAPI()
    
    # Get relevant memory context
    memory_context = await api.get_context_for_llm(
        query=user_query,
        user_id=user_id,
        session_id=session_id,
        max_context_length=2000
    )
    
    # Combine with user query for LLM
    full_prompt = f"""
    Context from memory:
    {memory_context}
    
    User question: {user_query}
    
    Please provide a response considering the above context.
    """
    
    # Send to your LLM (OpenAI, Anthropic, etc.)
    # response = await your_llm_client.chat_completion(full_prompt)
    
    # Store the interaction as a new memory
    await api.create_memory(
        content=f"User asked: {user_query}. Provided relevant information based on memory context.",
        memory_type="episodic",
        user_id=user_id,
        session_id=session_id
    )
    
    return response
```

### Automatic Memory Creation from Conversations

```python
async def process_conversation(conversation: str, user_id: str, session_id: str):
    api = NeuronMemoryAPI()
    
    # Extract key information and create memories
    if "meeting" in conversation.lower():
        await api.create_episodic_memory(
            content=conversation,
            user_id=user_id,
            session_id=session_id
        )
    elif "how to" in conversation.lower():
        await api.create_memory(
            content=conversation,
            memory_type="procedural",
            user_id=user_id,
            session_id=session_id
        )
    else:
        await api.create_memory(
            content=conversation,
            memory_type="semantic",
            user_id=user_id,
            session_id=session_id
        )
```

## 📊 Monitoring & Analytics

### Get System Statistics

```python
stats = await api.get_statistics()
print(f"Total memories: {stats['total_memories']}")
print(f"Active sessions: {stats['active_sessions']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Average response time: {stats['average_response_time']:.3f}s")
```

### Memory Health Check

```python
if api.is_healthy():
    print("✅ Memory system is operational")
else:
    print("❌ Memory system has issues")
```

## 🛠️ Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_OPENAI_API_KEY` | Your Azure OpenAI API key | Required |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | Required |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | GPT model deployment name | gpt-4 |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment | text-embedding-ada-002 |
| `NEURON_MEMORY_LOG_LEVEL` | Logging level | INFO |
| `NEURON_MEMORY_MAX_MEMORY_SIZE` | Maximum number of memories | 10000 |
| `CHROMA_DB_PATH` | ChromaDB storage path | ./data/chromadb |
| `MEMORY_STORE_PATH` | Memory store path | ./data/memory_store |

### Advanced Configuration

```python
from neuron_memory.config import neuron_memory_config

# Access configuration
print(f"Max memory size: {neuron_memory_config.max_memory_size}")
print(f"Vector store: {neuron_memory_config.vector_store}")
print(f"Embedding model: {neuron_memory_config.embedding_model}")
```

## 🚀 Best Practices

### 1. Session Management

- Always start/end sessions for context-aware operations
- Use descriptive session IDs and context information
- Group related operations in the same session

### 2. Memory Types

- Use **semantic** for general facts and knowledge
- Use **episodic** for personal experiences and events
- Use **social** for information about people and relationships
- Use **procedural** for step-by-step instructions
- Use **working** for temporary, task-specific information

### 3. Search Optimization

- Be specific with search queries
- Use appropriate memory type filters
- Set reasonable relevance thresholds
- Leverage session context for better results

### 4. Performance

- Batch similar operations when possible
- Use session contexts to improve search relevance
- Monitor system statistics for optimization opportunities

## 🔧 Troubleshooting

### Common Issues

1. **Configuration Errors**
   ```
   ValueError: Invalid configuration. Please check your .env file.
   ```
   - Verify Azure OpenAI credentials in `.env`
   - Check endpoint URL format
   - Ensure all required variables are set

2. **Memory Store Issues**
   ```
   Error initializing ChromaDB
   ```
   - Check disk space for database storage
   - Verify write permissions for data directory
   - Ensure ChromaDB dependencies are installed

3. **Performance Issues**
   - Monitor memory usage with `get_statistics()`
   - Check cache hit rates
   - Consider adjusting `max_memory_size` configuration

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📖 Examples

Run the included demonstration script to see NeuronMemory in action:

```bash
python demo.py
```

This script demonstrates:
- Basic memory operations
- Different memory types
- Search functionality
- LLM integration patterns
- Session management

## 🤝 Integration Patterns

### With LangChain

```python
from langchain.memory import ConversationBufferMemory
from neuron_memory import NeuronMemoryAPI

class NeuronMemoryLangChain:
    def __init__(self):
        self.api = NeuronMemoryAPI()
        self.session_id = "langchain_session"
    
    async def add_message(self, message: str, user_id: str):
        await self.api.create_memory(
            content=message,
            memory_type="episodic",
            user_id=user_id,
            session_id=self.session_id
        )
    
    async def get_relevant_context(self, query: str, user_id: str):
        return await self.api.get_context_for_llm(
            query=query,
            user_id=user_id,
            session_id=self.session_id
        )
```

### With Custom AI Agents

```python
class IntelligentAgent:
    def __init__(self):
        self.memory = NeuronMemoryAPI()
        self.agent_id = "agent_001"
    
    async def process_input(self, user_input: str, user_id: str):
        # Get relevant memories
        context = await self.memory.get_context_for_llm(
            query=user_input,
            user_id=user_id
        )
        
        # Process with LLM
        response = await self.generate_response(user_input, context)
        
        # Store interaction
        await self.memory.create_memory(
            content=f"User: {user_input} | Agent: {response}",
            memory_type="episodic",
            user_id=user_id
        )
        
        return response
```

This completes the comprehensive NeuronMemory system implementation with full functionality for advanced memory management in AI applications! 
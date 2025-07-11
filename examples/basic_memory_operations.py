"""
Basic Memory Operations Example

This example demonstrates fundamental NeuronMemory operations:
- Creating different types of memories
- Storing and retrieving memories
- Updating existing memories
- Searching memories
- Basic session management

Perfect for beginners to understand NeuronMemory basics.
"""

import asyncio
from datetime import datetime
from neuron_memory import NeuronMemoryAPI

class BasicMemoryDemo:
    """Demonstrates basic NeuronMemory operations"""
    
    def __init__(self):
        self.memory = NeuronMemoryAPI()
        self.user_id = "basic_demo_user"
        self.session_id = None
    
    async def setup_session(self):
        """Initialize a memory session"""
        self.session_id = f"basic_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        await self.memory.start_session(
            session_id=self.session_id,
            user_id=self.user_id,
            task="Basic memory operations demonstration",
            domain="learning"
        )
        
        print(f"✅ Started session: {self.session_id}")
        return self.session_id
    
    async def create_memories_example(self):
        """Demonstrate creating different types of memories"""
        print("\n" + "="*50)
        print("📝 Creating Different Types of Memories")
        print("="*50)
        
        # 1. Create an episodic memory (personal experience)
        episodic_id = await self.memory.create_episodic_memory(
            content="Had a productive meeting about the new project roadmap. Discussed key milestones and resource allocation.",
            participants=[self.user_id, "team_lead", "product_manager"],
            location="conference_room_A",
            emotional_state="excited",
            user_id=self.user_id,
            session_id=self.session_id
        )
        print(f"📖 Created episodic memory: {episodic_id}")
        
        # 2. Create a semantic memory (factual knowledge)
        semantic_id = await self.memory.create_semantic_memory(
            content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms.",
            domain="programming",
            confidence=0.9,
            user_id=self.user_id,
            session_id=self.session_id
        )
        print(f"🧠 Created semantic memory: {semantic_id}")
        
        # 3. Create a procedural memory (how-to knowledge)
        procedural_id = await self.memory.create_procedural_memory(
            content="To deploy a Python application: 1) Set up virtual environment, 2) Install dependencies, 3) Configure environment variables, 4) Run deployment script",
            skill_domain="deployment",
            proficiency_level="intermediate",
            user_id=self.user_id,
            session_id=self.session_id
        )
        print(f"⚙️ Created procedural memory: {procedural_id}")
        
        # 4. Create a social memory (relationship/person info)
        social_id = await self.memory.create_social_memory(
            content="John is a senior developer who prefers detailed technical discussions and always provides thorough code reviews.",
            person_id="john_dev",
            relationship_type="colleague",
            user_id=self.user_id,
            session_id=self.session_id
        )
        print(f"👥 Created social memory: {social_id}")
        
        # 5. Create a working memory (temporary context)
        working_id = await self.memory.create_working_memory(
            content="Currently working on user authentication feature. Need to implement JWT tokens and password hashing.",
            context="current_task",
            priority=0.8,
            user_id=self.user_id,
            session_id=self.session_id
        )
        print(f"⚡ Created working memory: {working_id}")
        
        return {
            "episodic": episodic_id,
            "semantic": semantic_id,
            "procedural": procedural_id,
            "social": social_id,
            "working": working_id
        }
    
    async def retrieve_memories_example(self, memory_ids):
        """Demonstrate retrieving memories by ID"""
        print("\n" + "="*50)
        print("🔍 Retrieving Memories by ID")
        print("="*50)
        
        for memory_type, memory_id in memory_ids.items():
            try:
                memory = await self.memory.get_memory(memory_id)
                print(f"\n{memory_type.upper()} Memory:")
                print(f"   ID: {memory['id']}")
                print(f"   Content: {memory['content'][:80]}...")
                print(f"   Created: {memory['created_at']}")
                print(f"   Type: {memory['type']}")
            except Exception as e:
                print(f"❌ Error retrieving {memory_type} memory: {e}")
    
    async def search_memories_example(self):
        """Demonstrate searching memories"""
        print("\n" + "="*50)
        print("🔎 Searching Memories")
        print("="*50)
        
        # Search by content similarity
        search_queries = [
            "Python programming",
            "team meeting",
            "deployment process",
            "John developer"
        ]
        
        for query in search_queries:
            print(f"\n🔍 Searching for: '{query}'")
            results = await self.memory.search_memories(
                query=query,
                user_id=self.user_id,
                limit=3
            )
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"   {i}. [{result['type']}] {result['content'][:60]}...")
                    print(f"      Relevance: {result.get('relevance_score', 'N/A')}")
            else:
                print("   No results found")
    
    async def update_memory_example(self, memory_ids):
        """Demonstrate updating existing memories"""
        print("\n" + "="*50)
        print("✏️ Updating Memories")
        print("="*50)
        
        # Update the semantic memory with additional information
        semantic_id = memory_ids["semantic"]
        
        print(f"Updating semantic memory: {semantic_id}")
        await self.memory.update_memory(
            memory_id=semantic_id,
            updates={
                "content": "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including object-oriented, functional, and procedural programming. Created by Guido van Rossum in 1991.",
                "confidence": 0.95
            }
        )
        
        # Retrieve updated memory
        updated_memory = await self.memory.get_memory(semantic_id)
        print(f"✅ Updated content: {updated_memory['content'][:100]}...")
        print(f"✅ Updated confidence: {updated_memory.get('confidence', 'N/A')}")
    
    async def memory_statistics_example(self):
        """Demonstrate memory statistics and analytics"""
        print("\n" + "="*50)
        print("📊 Memory Statistics")
        print("="*50)
        
        stats = await self.memory.get_statistics()
        
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    async def context_retrieval_example(self):
        """Demonstrate context retrieval for LLM integration"""
        print("\n" + "="*50)
        print("🧠 Context Retrieval for LLM")
        print("="*50)
        
        # Get context for a hypothetical LLM query
        query = "How should I approach the new Python project?"
        context = await self.memory.get_context_for_llm(
            query=query,
            user_id=self.user_id,
            session_id=self.session_id,
            max_context_length=500
        )
        
        print(f"Query: {query}")
        print(f"Context length: {len(context)} characters")
        print(f"Context preview: {context[:200]}...")
    
    async def cleanup_session(self):
        """Clean up the session"""
        if self.session_id:
            await self.memory.end_session(self.session_id)
            print(f"\n🏁 Ended session: {self.session_id}")

async def run_basic_memory_demo():
    """Run the complete basic memory operations demonstration"""
    
    print("="*60)
    print("🚀 NeuronMemory Basic Operations Demo")
    print("="*60)
    print("This demo covers:")
    print("• Creating different types of memories")
    print("• Retrieving memories by ID")
    print("• Searching memories by content")
    print("• Updating existing memories")
    print("• Getting memory statistics")
    print("• Context retrieval for LLM integration")
    
    demo = BasicMemoryDemo()
    
    try:
        # Setup
        await demo.setup_session()
        
        # Create memories
        memory_ids = await demo.create_memories_example()
        
        # Retrieve memories
        await demo.retrieve_memories_example(memory_ids)
        
        # Search memories
        await demo.search_memories_example()
        
        # Update memories
        await demo.update_memory_example(memory_ids)
        
        # Show statistics
        await demo.memory_statistics_example()
        
        # Context retrieval
        await demo.context_retrieval_example()
        
        print("\n" + "="*60)
        print("✅ Basic Memory Operations Demo Complete!")
        print("="*60)
        print("Key takeaways:")
        print("• NeuronMemory supports 5 types of memories")
        print("• Each memory type has specific fields and use cases")
        print("• Memories can be searched by content similarity")
        print("• Context can be retrieved for LLM integration")
        print("• Sessions help organize related memories")
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await demo.cleanup_session()

if __name__ == "__main__":
    asyncio.run(run_basic_memory_demo()) 
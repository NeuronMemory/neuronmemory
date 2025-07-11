#!/usr/bin/env python3
"""
NeuronMemory Demonstration Script

This script demonstrates the key features of the NeuronMemory system including:
- Creating different types of memories
- Searching and retrieving memories
- Session management
- LLM context integration

Before running this script, make sure to:
1. Install dependencies: pip install -r requirements.txt
2. Set up your .env file with Azure OpenAI credentials
3. Configure the necessary environment variables
"""

import asyncio
import logging
import json
from datetime import datetime
from neuron_memory import NeuronMemoryAPI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_basic_operations():
    """Demonstrate basic memory operations"""
    print("Initializing NeuronMemory...")
    
    try:
        # Initialize the API
        api = NeuronMemoryAPI()
        
        if not api.is_healthy():
            print("NeuronMemory system is not healthy!")
            return
        
        print("NeuronMemory initialized successfully!")
        
        # Start a session
        session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        user_id = "demo_user"
        
        print(f"\nStarting session: {session_id}")
        await api.start_session(
            session_id=session_id,
            user_id=user_id,
            task="Learning about NeuronMemory capabilities",
            domain="AI and Machine Learning"
        )
        
        # Create different types of memories
        print("\nCreating memories...")
        
        # 1. Semantic memory (facts and knowledge)
        memory_id_1 = await api.create_memory(
            content="Python is a high-level programming language known for its simplicity and readability",
            memory_type="semantic",
            user_id=user_id,
            session_id=session_id
        )
        print(f"   Created semantic memory: {memory_id_1}")
        
        # 2. Episodic memory (personal experience)
        memory_id_2 = await api.create_episodic_memory(
            content="Had a productive meeting discussing the implementation of NeuronMemory with the development team",
            participants=["Alice", "Bob", "Charlie"],
            location="Conference Room A",
            emotions={"valence": 0.8, "arousal": 0.6, "dominance": 0.7},
            user_id=user_id,
            session_id=session_id
        )
        print(f"   Created episodic memory: {memory_id_2}")
        
        # 3. Social memory (about people and relationships)
        memory_id_3 = await api.create_social_memory(
            content="Alice is an expert in machine learning and prefers technical discussions in the morning",
            person_id="alice_123",
            relationship_type="colleague",
            user_id=user_id,
            session_id=session_id
        )
        print(f"   Created social memory: {memory_id_3}")
        
        # 4. Procedural memory (how to do things)
        memory_id_4 = await api.create_memory(
            content="To initialize NeuronMemory: 1) Set up environment variables, 2) Create API instance, 3) Start session",
            memory_type="procedural",
            user_id=user_id,
            session_id=session_id
        )
        print(f"   Created procedural memory: {memory_id_4}")
        
        # Search for memories
        print("\nSearching memories...")
        
        # Search by content similarity
        results = await api.search_memories(
            query="Python programming language",
            user_id=user_id,
            session_id=session_id,
            limit=3
        )
        
        print(f"   Found {len(results)} results for 'Python programming language':")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['content'][:60]}... (relevance: {result['relevance_score']:.2f})")
        
        # Search for social information
        social_results = await api.search_memories(
            query="Alice preferences and expertise",
            user_id=user_id,
            memory_types=["social", "episodic"],
            limit=2
        )
        
        print(f"\n   Found {len(social_results)} social results about Alice:")
        for i, result in enumerate(social_results, 1):
            print(f"   {i}. {result['content'][:60]}... (type: {result['memory_type']})")
        
        # Get LLM context
        print("\nGetting context for LLM...")
        context = await api.get_context_for_llm(
            query="How to work effectively with Alice on technical projects",
            user_id=user_id,
            session_id=session_id,
            max_context_length=500
        )
        
        print("   LLM Context:")
        print(f"   {context}")
        
        # Retrieve specific memory
        print("\nRetrieving specific memory...")
        memory_details = await api.get_memory(memory_id_2)
        if memory_details:
            print(f"   Memory: {memory_details['content']}")
            print(f"   Type: {memory_details['memory_type']}")
            print(f"   Importance: {memory_details['importance_score']:.2f}")
            print(f"   Created: {memory_details['created_at']}")
        
        # Get system statistics
        print("\nSystem Statistics:")
        stats = await api.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # End session
        print(f"\nEnding session: {session_id}")
        await api.end_session(session_id)
        
        print("\nDemonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        print(f"Error: {e}")

async def demonstrate_advanced_features():
    """Demonstrate advanced memory features"""
    print("\nAdvanced Features Demonstration")
    
    try:
        api = NeuronMemoryAPI()
        
        # Create a new session for advanced features
        session_id = f"advanced_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        user_id = "advanced_user"
        
        await api.start_session(
            session_id=session_id,
            user_id=user_id,
            task="Advanced memory system testing",
            domain="Cognitive Science"
        )
        
        # Create memories with different importance levels
        print("\nCreating memories with different importance levels...")
        
        high_importance = await api.create_memory(
            content="Critical security vulnerability discovered in production system - requires immediate attention",
            memory_type="semantic",
            importance=0.95,
            user_id=user_id,
            session_id=session_id
        )
        
        medium_importance = await api.create_memory(
            content="Team lunch scheduled for Friday at 12:30 PM in the cafeteria",
            memory_type="episodic",
            importance=0.4,
            user_id=user_id,
            session_id=session_id
        )
        
        # Search with different strategies
        print("\nTesting different search approaches...")
        
        # Search for critical information
        critical_results = await api.search_memories(
            query="security vulnerability production",
            user_id=user_id,
            session_id=session_id,
            min_relevance=0.5
        )
        
        print(f"   Critical search results: {len(critical_results)}")
        for result in critical_results:
            print(f"   - {result['content'][:50]}... (importance: {result['importance_score']:.2f})")
        
        # Search by memory type
        episodic_results = await api.search_memories(
            query="team activities and events",
            user_id=user_id,
            memory_types=["episodic"],
            limit=5
        )
        
        print(f"\n   Episodic memories found: {len(episodic_results)}")
        for result in episodic_results:
            print(f"   - {result['content'][:50]}...")
        
        await api.end_session(session_id)
        
    except Exception as e:
        logger.error(f"Error in advanced demonstration: {e}")
        print(f"Advanced demo error: {e}")

async def demonstrate_llm_integration():
    """Demonstrate how to integrate NeuronMemory with LLM workflows"""
    print("\nLLM Integration Demonstration")
    
    try:
        api = NeuronMemoryAPI()
        
        session_id = f"llm_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        user_id = "llm_user"
        
        await api.start_session(
            session_id=session_id,
            user_id=user_id,
            task="Customer support conversation",
            domain="Customer Service"
        )
        
        # Simulate storing conversation context
        print("\nStoring conversation context...")
        
        await api.create_memory(
            content="Customer John Smith called about billing issue with invoice #12345",
            memory_type="episodic",
            user_id=user_id,
            session_id=session_id
        )
        
        await api.create_social_memory(
            content="John Smith is a premium customer, prefers phone support, has been with us for 3 years",
            person_id="john_smith_456",
            relationship_type="customer",
            user_id=user_id,
            session_id=session_id
        )
        
        await api.create_memory(
            content="Billing issues should be escalated to the finance team if over $500",
            memory_type="procedural",
            user_id=user_id,
            session_id=session_id
        )
        
        # Simulate an LLM query with memory context
        print("\nGetting memory context for LLM response...")
        
        user_query = "The customer is asking about refund policy for their recent invoice"
        
        memory_context = await api.get_context_for_llm(
            query=f"customer service billing refund policy {user_query}",
            user_id=user_id,
            session_id=session_id,
            max_context_length=1000
        )
        
        print("   Memory context for LLM:")
        print(memory_context)
        
        # This is where you would send the context + query to your LLM
        print("\n   This context would be sent to LLM along with the user query:")
        print(f"   User Query: {user_query}")
        print(f"   Memory Context: {memory_context}")
        
        # Store the interaction as a new memory
        interaction_summary = f"Provided customer service for John Smith regarding refund policy inquiry"
        await api.create_memory(
            content=interaction_summary,
            memory_type="episodic",
            user_id=user_id,
            session_id=session_id
        )
        
        await api.end_session(session_id)
        
        print("\nLLM integration demonstration completed!")
        
    except Exception as e:
        logger.error(f"Error in LLM integration demo: {e}")
        print(f"LLM integration error: {e}")

async def main():
    """Main demonstration function"""
    print("=" * 60)
    print("NeuronMemory System Demonstration")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        await demonstrate_basic_operations()
        await demonstrate_advanced_features()
        await demonstrate_llm_integration()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in main demonstration: {e}")
        print(f"Main error: {e}")
        print("\nMake sure you have:")
        print("   1. Set up your .env file with Azure OpenAI credentials")
        print("   2. Installed all required dependencies")
        print("   3. Configured the AZURE_OPENAI_* environment variables")

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
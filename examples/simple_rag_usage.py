"""
Simple Usage Example for NeuronMemory RAG System

This script demonstrates how to use the RAG system with your PDF data.
"""

import asyncio
from examples.neuron_rag_system import NeuronRAGSystem

async def main():
    print("🚀 NeuronMemory RAG System - Simple Usage Example")
    print("=" * 60)
    
    # Initialize the RAG system
    rag = NeuronRAGSystem(docs_directory="docs_data")
    
    # Process PDF documents
    print("📄 Processing PDF documents...")
    results = rag.process_pdf_documents()
    
    if results["processing_errors"]:
        print("❌ Errors occurred:")
        for error in results["processing_errors"]:
            print(f"   {error}")
        return
    
    print(f"✅ Successfully processed {len(results['processed_files'])} files")
    print(f"   Total chunks created: {results['total_chunks']}")
    
    # Start a session
    user_id = "expat_user_123"
    greeting = await rag.start_rag_session(user_id)
    print(f"\n🤖 {greeting}")
    
    # Example queries
    queries = [
        "What are the main challenges of living abroad?",
        "How do I handle visa applications?",
        "What should I know about housing in a new country?",
        "How can I manage my finances as an expat?",
        "What are some tips for cultural adaptation?"
    ]
    
    print("\n💬 Query Examples:")
    print("-" * 40)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. User: {query}")
        
        response = await rag.query_documents(query, user_id)
        
        print(f"   🤖 Response: {response['answer'][:200]}...")
        print(f"   📊 Sources: {len(response['sources'])} documents")
        print(f"   🧠 Memory Enhanced: {response['memory_enhanced']}")
        
        if response.get('suggested_followups'):
            print(f"   💡 Follow-ups: {response['suggested_followups'][0]}")
    
    # Get analytics
    analytics = await rag.get_user_analytics(user_id)
    print(f"\n📈 Session Analytics:")
    print(f"   Total queries: {analytics['total_queries']}")
    print(f"   Knowledge level: {analytics['knowledge_progression']}")
    
    # End session
    await rag.end_rag_session(user_id, "Comprehensive expat guidance session completed")
    print("\n✅ Session ended successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 
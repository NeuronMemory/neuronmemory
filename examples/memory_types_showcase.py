"""
Memory Types Showcase Example

This example demonstrates all 5 types of memories in NeuronMemory:
1. Episodic Memory - Personal experiences and events
2. Semantic Memory - Factual knowledge and concepts  
3. Procedural Memory - Skills and procedures
4. Social Memory - Relationships and social interactions
5. Working Memory - Temporary, active information

Perfect for understanding the different memory types and their use cases.
"""

import asyncio
from datetime import datetime
from neuron_memory import NeuronMemoryAPI

class MemoryTypesShowcase:
    """Comprehensive demonstration of all NeuronMemory types"""
    
    def __init__(self):
        self.memory = NeuronMemoryAPI()
        self.user_id = "memory_showcase_user"
        self.session_id = None
    
    async def setup_demo(self):
        """Initialize the demo session"""
        self.session_id = f"memory_types_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        await self.memory.start_session(
            session_id=self.session_id,
            user_id=self.user_id,
            task="Memory types demonstration",
            domain="learning"
        )
        
        print(f"✅ Started memory types demo session: {self.session_id}")
    
    async def demonstrate_episodic_memory(self):
        """Demonstrate episodic memory - personal experiences"""
        print("\n" + "="*60)
        print("📖 EPISODIC MEMORY - Personal Experiences & Events")
        print("="*60)
        print("Episodic memory stores personal experiences with:")
        print("• Who was involved (participants)")
        print("• Where it happened (location)")
        print("• When it occurred (temporal context)")
        print("• Emotional context")
        
        # Example 1: Meeting experience
        print("\n📝 Example 1: Work Meeting")
        meeting_id = await self.memory.create_episodic_memory(
            content="Had a productive quarterly review meeting. Discussed project milestones and budget allocation.",
            participants=[self.user_id, "sarah_analyst", "mike_manager"],
            location="conference_room_B",
            emotional_state="satisfied",
            user_id=self.user_id,
            session_id=self.session_id
        )
        print(f"   Created meeting memory: {meeting_id}")
        
        # Example 2: Learning experience
        print("\n📝 Example 2: Learning Experience")
        learning_id = await self.memory.create_episodic_memory(
            content="Attended AI workshop on neural networks. Made connections with other developers.",
            participants=[self.user_id, "workshop_instructor"],
            location="tech_conference_hall",
            emotional_state="excited",
            user_id=self.user_id,
            session_id=self.session_id
        )
        print(f"   Created learning memory: {learning_id}")
        
        print("\n💡 Episodic Memory Use Cases:")
        print("   • Personal diary/journal entries")
        print("   • Meeting notes and outcomes")
        print("   • Learning experiences and insights")
        
        return [meeting_id, learning_id]
    
    async def demonstrate_semantic_memory(self):
        """Demonstrate semantic memory - factual knowledge"""
        print("\n" + "="*60)
        print("🧠 SEMANTIC MEMORY - Factual Knowledge & Concepts")
        print("="*60)
        print("Semantic memory stores factual knowledge including:")
        print("• Concepts and definitions")
        print("• Facts and data")
        print("• Domain-specific knowledge")
        
        # Example 1: Technical knowledge
        print("\n📝 Example 1: Technical Knowledge")
        tech_id = await self.memory.create_semantic_memory(
            content="Machine Learning is a subset of AI that enables computers to learn patterns from data without explicit programming.",
            domain="machine_learning",
            confidence=0.9,
            user_id=self.user_id,
            session_id=self.session_id
        )
        print(f"   Created technical knowledge: {tech_id}")
        
        # Example 2: Business knowledge
        print("\n📝 Example 2: Business Knowledge")
        business_id = await self.memory.create_semantic_memory(
            content="Agile methodology emphasizes iterative development and customer collaboration.",
            domain="business_methodology",
            confidence=0.85,
            user_id=self.user_id,
            session_id=self.session_id
        )
        print(f"   Created business knowledge: {business_id}")
        
        print("\n💡 Semantic Memory Use Cases:")
        print("   • Knowledge base articles")
        print("   • Technical documentation")
        print("   • Best practices and guidelines")
        
        return [tech_id, business_id]
    
    async def demonstrate_procedural_memory(self):
        """Demonstrate procedural memory - skills and procedures"""
        print("\n" + "="*60)
        print("⚙️ PROCEDURAL MEMORY - Skills & Procedures")
        print("="*60)
        print("Procedural memory stores how-to knowledge including:")
        print("• Step-by-step procedures")
        print("• Skills and techniques")
        print("• Workflows and processes")
        
        # Example 1: Development workflow
        print("\n📝 Example 1: Development Workflow")
        dev_id = await self.memory.create_procedural_memory(
            content="Code Review Process: 1) Create feature branch, 2) Implement changes with tests, 3) Create pull request, 4) Address feedback, 5) Merge after approval.",
            skill_domain="software_development",
            proficiency_level="advanced",
            user_id=self.user_id,
            session_id=self.session_id
        )
        print(f"   Created development workflow: {dev_id}")
        
        # Example 2: Debugging procedure
        print("\n📝 Example 2: Debugging Procedure")
        debug_id = await self.memory.create_procedural_memory(
            content="Debugging Approach: 1) Reproduce issue, 2) Read error messages, 3) Check recent changes, 4) Use logging, 5) Test fix.",
            skill_domain="problem_solving",
            proficiency_level="intermediate",
            user_id=self.user_id,
            session_id=self.session_id
        )
        print(f"   Created debugging procedure: {debug_id}")
        
        print("\n💡 Procedural Memory Use Cases:")
        print("   • Standard operating procedures")
        print("   • Troubleshooting guides")
        print("   • Training materials")
        
        return [dev_id, debug_id]
    
    async def demonstrate_social_memory(self):
        """Demonstrate social memory - relationships"""
        print("\n" + "="*60)
        print("👥 SOCIAL MEMORY - Relationships & Social Context")
        print("="*60)
        print("Social memory stores relationship information including:")
        print("• Individual personality traits")
        print("• Communication preferences")
        print("• Relationship history")
        
        # Example 1: Colleague profile
        print("\n📝 Example 1: Colleague Profile")
        colleague_id = await self.memory.create_social_memory(
            content="Sarah is a detail-oriented data analyst who prefers written communication and provides thorough documentation.",
            person_id="sarah_analyst",
            relationship_type="colleague",
            user_id=self.user_id,
            session_id=self.session_id
        )
        print(f"   Created colleague profile: {colleague_id}")
        
        # Example 2: Manager relationship
        print("\n📝 Example 2: Manager Relationship")
        manager_id = await self.memory.create_social_memory(
            content="Mike is a supportive manager who values innovation and prefers high-level updates.",
            person_id="mike_manager",
            relationship_type="manager",
            user_id=self.user_id,
            session_id=self.session_id
        )
        print(f"   Created manager relationship: {manager_id}")
        
        print("\n💡 Social Memory Use Cases:")
        print("   • CRM and customer profiles")
        print("   • Team collaboration optimization")
        print("   • Personalized communication")
        
        return [colleague_id, manager_id]
    
    async def demonstrate_working_memory(self):
        """Demonstrate working memory - temporary active information"""
        print("\n" + "="*60)
        print("⚡ WORKING MEMORY - Temporary Active Information")
        print("="*60)
        print("Working memory stores temporary information including:")
        print("• Current task context")
        print("• Active thoughts and ideas")
        print("• Immediate priorities")
        
        # Example 1: Current project
        print("\n📝 Example 1: Current Project Context")
        project_id = await self.memory.create_working_memory(
            content="Currently working on user authentication feature. Need JWT validation and password hashing.",
            context="current_project",
            priority=0.9,
            user_id=self.user_id,
            session_id=self.session_id
        )
        print(f"   Created project context: {project_id}")
        
        # Example 2: Today's tasks
        print("\n📝 Example 2: Today's Tasks")
        tasks_id = await self.memory.create_working_memory(
            content="Today's priorities: 1) Code review, 2) Test API endpoints, 3) Prepare presentation.",
            context="daily_tasks",
            priority=0.8,
            user_id=self.user_id,
            session_id=self.session_id
        )
        print(f"   Created daily tasks: {tasks_id}")
        
        print("\n💡 Working Memory Use Cases:")
        print("   • Task management")
        print("   • Active problem-solving context")
        print("   • Session-specific information")
        
        return [project_id, tasks_id]
    
    async def demonstrate_memory_interactions(self):
        """Show how different memory types interact"""
        print("\n" + "="*60)
        print("🔗 MEMORY INTERACTIONS & CONNECTIONS")
        print("="*60)
        
        # Search across all memory types
        print("\n🔍 Cross-Memory Search: 'Sarah'")
        results = await self.memory.search_memories(
            query="Sarah",
            user_id=self.user_id,
            limit=3
        )
        
        for i, result in enumerate(results, 1):
            print(f"   {i}. [{result['type'].upper()}] {result['content'][:60]}...")
        
        print("\n💡 Memory Interaction Patterns:")
        print("   • Episodic experiences inform semantic knowledge")
        print("   • Procedural skills apply to working memory tasks")
        print("   • Social context influences all memory types")
    
    async def cleanup_demo(self):
        """Clean up the demo session"""
        if self.session_id:
            await self.memory.end_session(self.session_id)
            print(f"\n🏁 Ended demo session: {self.session_id}")

async def run_memory_types_showcase():
    """Run the complete memory types showcase"""
    
    print("="*70)
    print("🧠 NeuronMemory Types Comprehensive Showcase")
    print("="*70)
    
    showcase = MemoryTypesShowcase()
    
    try:
        await showcase.setup_demo()
        
        # Demonstrate each memory type
        episodic_ids = await showcase.demonstrate_episodic_memory()
        semantic_ids = await showcase.demonstrate_semantic_memory()
        procedural_ids = await showcase.demonstrate_procedural_memory()
        social_ids = await showcase.demonstrate_social_memory()
        working_ids = await showcase.demonstrate_working_memory()
        
        # Show interactions
        await showcase.demonstrate_memory_interactions()
        
        print("\n" + "="*70)
        print("✅ Memory Types Showcase Complete!")
        print("="*70)
        print("Summary of what was created:")
        print(f"• {len(episodic_ids)} Episodic memories (experiences)")
        print(f"• {len(semantic_ids)} Semantic memories (knowledge)")
        print(f"• {len(procedural_ids)} Procedural memories (skills)")
        print(f"• {len(social_ids)} Social memories (relationships)")
        print(f"• {len(working_ids)} Working memories (temporary context)")
        
    except Exception as e:
        print(f"❌ Showcase failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await showcase.cleanup_demo()

if __name__ == "__main__":
    asyncio.run(run_memory_types_showcase())
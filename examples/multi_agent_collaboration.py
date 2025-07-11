"""
Multi-Agent Collaboration Example

This example demonstrates multiple AI agents sharing and collaborating through memory:
- Multiple agents with shared memory spaces
- Agent-specific memory isolation and privacy
- Cross-agent communication and knowledge sharing
- Collaborative learning and problem-solving
- Distributed memory management across agents

Shows advanced multi-agent memory patterns and collaborative intelligence.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import json
from neuron_memory import NeuronMemoryAPI

class CollaborativeAgent:
    """Base class for collaborative AI agents with shared memory"""
    
    def __init__(self, agent_id: str, agent_name: str, specialization: str):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.specialization = specialization
        self.memory = NeuronMemoryAPI()
        self.current_session = None
        
        # Agent capabilities and permissions
        self.capabilities = []
        self.memory_permissions = {
            "read_shared": True,
            "write_shared": True,
            "read_private": True,
            "write_private": True
        }
        
        # Collaboration history
        self.collaboration_history = []
        self.trusted_agents = set()
    
    async def initialize_agent(self, capabilities: List[str] = None):
        """Initialize agent with capabilities and memory setup"""
        self.capabilities = capabilities or []
        
        # Create agent profile in memory
        await self.memory.create_semantic_memory(
            content=f"Agent {self.agent_name} ({self.agent_id}) specializes in {self.specialization}. Capabilities: {', '.join(self.capabilities)}",
            domain="agent_profiles",
            confidence=1.0,
            user_id=self.agent_id
        )
        
        print(f"🤖 Agent {self.agent_name} initialized with specialization: {self.specialization}")
    
    async def start_collaboration_session(self, session_id: str, collaborating_agents: List[str], task: str):
        """Start a collaborative session with other agents"""
        self.current_session = session_id
        
        await self.memory.start_session(
            session_id=session_id,
            user_id=self.agent_id,
            task=f"Collaborative task: {task}",
            domain="multi_agent_collaboration"
        )
        
        # Log collaboration start
        await self.memory.create_episodic_memory(
            content=f"Started collaboration session {session_id} with agents: {', '.join(collaborating_agents)} for task: {task}",
            participants=[self.agent_id] + collaborating_agents,
            location="virtual_workspace",
            emotional_state="collaborative",
            user_id=self.agent_id,
            session_id=session_id
        )
        
        print(f"🤝 {self.agent_name} started collaboration session: {session_id}")
    
    async def share_knowledge(self, knowledge: str, target_agents: List[str] = None, knowledge_type: str = "general"):
        """Share knowledge with other agents"""
        
        # Store as shared knowledge
        shared_memory = await self.memory.create_semantic_memory(
            content=f"[SHARED by {self.agent_name}] {knowledge}",
            domain=f"shared_knowledge_{knowledge_type}",
            confidence=0.8,
            user_id="shared_workspace",  # Use shared workspace
            session_id=self.current_session
        )
        
        # Log the knowledge sharing
        await self.memory.create_episodic_memory(
            content=f"Agent {self.agent_name} shared knowledge: {knowledge[:100]}... with agents: {target_agents or 'all'}",
            participants=[self.agent_id] + (target_agents or []),
            location="shared_memory_space",
            emotional_state="collaborative",
            user_id=self.agent_id,
            session_id=self.current_session
        )
        
        print(f"📤 {self.agent_name} shared knowledge: {knowledge[:50]}...")
        return shared_memory
    
    async def request_assistance(self, problem: str, preferred_agents: List[str] = None) -> Dict[str, Any]:
        """Request assistance from other agents"""
        
        # Store assistance request
        request = await self.memory.create_episodic_memory(
            content=f"[ASSISTANCE REQUEST] {self.agent_name} needs help with: {problem}",
            participants=[self.agent_id] + (preferred_agents or []),
            location="collaboration_channel",
            emotional_state="seeking_help",
            user_id=self.agent_id,
            session_id=self.current_session
        )
        
        # Search for relevant knowledge from other agents
        relevant_knowledge = await self.memory.search_memories(
            query=problem,
            user_id="shared_workspace",
            memory_types=["semantic", "procedural"],
            limit=10
        )
        
        assistance_response = {
            "request_id": request["id"] if "id" in request else "temp_id",
            "problem": problem,
            "relevant_knowledge": relevant_knowledge,
            "responding_agents": [],
            "solutions": []
        }
        
        print(f"🆘 {self.agent_name} requested assistance: {problem[:50]}...")
        return assistance_response
    
    async def provide_assistance(self, request_id: str, problem: str, solution: str) -> Dict[str, Any]:
        """Provide assistance to another agent"""
        
        # Store the solution as shared knowledge
        solution_memory = await self.memory.create_procedural_memory(
            content=f"[SOLUTION by {self.agent_name}] For problem '{problem}': {solution}",
            skill_domain=self.specialization,
            proficiency_level="expert",
            user_id="shared_workspace",
            session_id=self.current_session
        )
        
        # Log the assistance provided
        await self.memory.create_episodic_memory(
            content=f"Agent {self.agent_name} provided solution for: {problem}",
            participants=[self.agent_id],
            location="collaboration_channel",
            emotional_state="helpful",
            user_id=self.agent_id,
            session_id=self.current_session
        )
        
        assistance_data = {
            "solution_id": solution_memory.get("id", "temp_id"),
            "providing_agent": self.agent_name,
            "solution": solution,
            "confidence": 0.8,
            "specialization_match": self.specialization in problem.lower()
        }
        
        print(f"💡 {self.agent_name} provided solution: {solution[:50]}...")
        return assistance_data
    
    async def learn_from_collaboration(self, collaboration_data: Dict[str, Any]):
        """Learn from collaborative experiences"""
        
        # Extract learning insights
        insights = []
        
        if collaboration_data.get("successful_collaborations", 0) > 0:
            insights.append("Successful collaborative problem-solving demonstrated")
        
        if collaboration_data.get("knowledge_shared", 0) > 0:
            insights.append(f"Contributed {collaboration_data['knowledge_shared']} knowledge items")
        
        if collaboration_data.get("assistance_provided", 0) > 0:
            insights.append("Effective assistance provider in team context")
        
        # Store collaborative learning
        if insights:
            await self.memory.create_semantic_memory(
                content=f"Collaborative learning insights: {'; '.join(insights)}",
                domain="collaborative_learning",
                confidence=0.7,
                user_id=self.agent_id,
                session_id=self.current_session
            )
        
        print(f"🧠 {self.agent_name} learned from collaboration: {len(insights)} insights")
    
    async def get_shared_knowledge(self, topic: str, limit: int = 10) -> List[Dict]:
        """Retrieve shared knowledge on a topic"""
        
        shared_knowledge = await self.memory.search_memories(
            query=topic,
            user_id="shared_workspace",
            memory_types=["semantic", "procedural"],
            limit=limit
        )
        
        # Filter by shared knowledge marker
        filtered_knowledge = [
            memory for memory in shared_knowledge 
            if "[SHARED" in memory.get("content", "")
        ]
        
        print(f"📚 {self.agent_name} retrieved {len(filtered_knowledge)} shared knowledge items on: {topic}")
        return filtered_knowledge
    
    async def end_collaboration_session(self, session_summary: str = None):
        """End collaborative session"""
        if self.current_session:
            summary = session_summary or f"Collaboration session completed by {self.agent_name}"
            
            await self.memory.create_episodic_memory(
                content=f"Session summary: {summary}",
                participants=[self.agent_id],
                location="virtual_workspace",
                emotional_state="accomplished",
                user_id=self.agent_id,
                session_id=self.current_session
            )
            
            await self.memory.end_session(self.current_session)
            print(f"🏁 {self.agent_name} ended collaboration session: {self.current_session}")
            self.current_session = None

class MultiAgentCollaborationSystem:
    """System for managing multi-agent collaboration"""
    
    def __init__(self):
        self.agents = {}
        self.active_sessions = {}
        self.collaboration_history = []
        self.memory = NeuronMemoryAPI()
    
    def register_agent(self, agent: CollaborativeAgent):
        """Register an agent in the collaboration system"""
        self.agents[agent.agent_id] = agent
        print(f"✅ Registered agent: {agent.agent_name} ({agent.agent_id})")
    
    async def orchestrate_collaboration(self, task: str, required_specializations: List[str] = None) -> Dict[str, Any]:
        """Orchestrate collaboration between agents for a specific task"""
        
        session_id = f"collab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Select agents based on specializations
        selected_agents = []
        if required_specializations:
            for spec in required_specializations:
                for agent in self.agents.values():
                    if spec.lower() in agent.specialization.lower() and agent not in selected_agents:
                        selected_agents.append(agent)
                        break
        else:
            selected_agents = list(self.agents.values())[:3]  # Default to first 3 agents
        
        if not selected_agents:
            print("❌ No suitable agents found for collaboration")
            return {"error": "No suitable agents available"}
        
        # Start collaboration session for all selected agents
        agent_ids = [agent.agent_id for agent in selected_agents]
        for agent in selected_agents:
            await agent.start_collaboration_session(session_id, agent_ids, task)
        
        self.active_sessions[session_id] = {
            "task": task,
            "agents": selected_agents,
            "start_time": datetime.now(),
            "status": "active"
        }
        
        collaboration_result = {
            "session_id": session_id,
            "task": task,
            "participating_agents": [agent.agent_name for agent in selected_agents],
            "status": "initiated"
        }
        
        print(f"🎯 Orchestrated collaboration for task: {task}")
        print(f"   Participating agents: {', '.join(collaboration_result['participating_agents'])}")
        
        return collaboration_result
    
    async def facilitate_knowledge_exchange(self, session_id: str, knowledge_requests: List[Dict]) -> Dict[str, Any]:
        """Facilitate knowledge exchange between agents"""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        agents = session["agents"]
        
        exchange_results = {
            "session_id": session_id,
            "exchanges": [],
            "total_knowledge_shared": 0,
            "collaboration_score": 0
        }
        
        # Process each knowledge request
        for request in knowledge_requests:
            requesting_agent_id = request.get("agent_id")
            topic = request.get("topic")
            
            requesting_agent = next((a for a in agents if a.agent_id == requesting_agent_id), None)
            if not requesting_agent:
                continue
            
            # Find agents that can provide knowledge on this topic
            relevant_knowledge = []
            providing_agents = []
            
            for agent in agents:
                if agent.agent_id != requesting_agent_id:
                    # Check if agent has knowledge on this topic
                    agent_knowledge = await agent.get_shared_knowledge(topic, limit=3)
                    if agent_knowledge:
                        relevant_knowledge.extend(agent_knowledge)
                        providing_agents.append(agent.agent_name)
            
            if relevant_knowledge:
                exchange_results["exchanges"].append({
                    "requesting_agent": requesting_agent.agent_name,
                    "topic": topic,
                    "providing_agents": providing_agents,
                    "knowledge_items": len(relevant_knowledge)
                })
                exchange_results["total_knowledge_shared"] += len(relevant_knowledge)
        
        # Calculate collaboration score
        if exchange_results["exchanges"]:
            exchange_results["collaboration_score"] = min(100, 
                (exchange_results["total_knowledge_shared"] / len(agents)) * 20
            )
        
        print(f"🔄 Facilitated knowledge exchange: {len(exchange_results['exchanges'])} exchanges")
        return exchange_results
    
    async def resolve_collaborative_problem(self, session_id: str, problem: str) -> Dict[str, Any]:
        """Coordinate agents to solve a problem collaboratively"""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        agents = session["agents"]
        
        # Step 1: All agents analyze the problem
        problem_analysis = {}
        for agent in agents:
            # Let each agent provide their perspective
            agent_analysis = await self._get_agent_perspective(agent, problem)
            problem_analysis[agent.agent_name] = agent_analysis
        
        # Step 2: Identify the best-suited agent(s) for the problem
        best_suited_agents = []
        for agent in agents:
            if any(keyword in problem.lower() for keyword in agent.specialization.lower().split()):
                best_suited_agents.append(agent)
        
        if not best_suited_agents:
            best_suited_agents = agents[:2]  # Default to first two agents
        
        # Step 3: Generate collaborative solution
        solutions = []
        for agent in best_suited_agents:
            solution = await agent.provide_assistance("collaborative_problem", problem, 
                f"As a {agent.specialization} specialist, I suggest addressing this through systematic analysis and applying domain-specific best practices."
            )
            solutions.append(solution)
        
        # Step 4: Synthesize solutions
        collaborative_solution = await self._synthesize_solutions(problem, solutions, agents)
        
        result = {
            "session_id": session_id,
            "problem": problem,
            "participating_agents": [agent.agent_name for agent in agents],
            "primary_solvers": [agent.agent_name for agent in best_suited_agents],
            "individual_solutions": solutions,
            "collaborative_solution": collaborative_solution
        }
        
        print(f"🎯 Resolved collaborative problem: {problem[:50]}...")
        return result
    
    async def _get_agent_perspective(self, agent: CollaborativeAgent, problem: str) -> Dict[str, Any]:
        """Get an agent's perspective on a problem"""
        
        # Simple analysis based on agent's specialization
        perspective = {
            "agent": agent.agent_name,
            "specialization": agent.specialization,
            "relevance_score": 0.5,
            "key_insights": [],
            "suggested_approach": ""
        }
        
        # Calculate relevance based on specialization match
        if any(keyword in problem.lower() for keyword in agent.specialization.lower().split()):
            perspective["relevance_score"] = 0.9
            perspective["key_insights"].append(f"High relevance to {agent.specialization}")
        
        # Generate approach based on specialization
        if "research" in agent.specialization.lower():
            perspective["suggested_approach"] = "Systematic research and analysis approach"
        elif "technical" in agent.specialization.lower():
            perspective["suggested_approach"] = "Technical implementation and testing approach"
        elif "creative" in agent.specialization.lower():
            perspective["suggested_approach"] = "Creative problem-solving and innovation approach"
        else:
            perspective["suggested_approach"] = "Collaborative and interdisciplinary approach"
        
        return perspective
    
    async def _synthesize_solutions(self, problem: str, solutions: List[Dict], agents: List[CollaborativeAgent]) -> str:
        """Synthesize individual solutions into a collaborative solution"""
        
        # Extract key elements from all solutions
        solution_elements = []
        for solution in solutions:
            if solution.get("solution"):
                # Extract key phrases (simplified)
                words = solution["solution"].split()
                if len(words) > 10:
                    key_phrase = " ".join(words[:10]) + "..."
                else:
                    key_phrase = solution["solution"]
                solution_elements.append(f"• {solution['providing_agent']}: {key_phrase}")
        
        # Create synthesized solution
        synthesized = f"Collaborative solution for: {problem}\n\n"
        synthesized += "Integrated approach combining multiple perspectives:\n"
        synthesized += "\n".join(solution_elements)
        synthesized += f"\n\nThis solution leverages the expertise of {len(agents)} specialized agents working together."
        
        return synthesized
    
    async def generate_collaboration_report(self, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive collaboration report"""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        agents = session["agents"]
        
        # Gather collaboration metrics
        collaboration_metrics = {
            "session_id": session_id,
            "task": session["task"],
            "duration": (datetime.now() - session["start_time"]).total_seconds() / 60,  # minutes
            "participating_agents": len(agents),
            "agent_details": [],
            "knowledge_sharing_events": 0,
            "assistance_requests": 0,
            "solutions_generated": 0,
            "collaboration_effectiveness": 0
        }
        
        # Analyze each agent's contribution
        for agent in agents:
            agent_memories = await agent.memory.search_memories(
                query=f"session {session_id}",
                user_id=agent.agent_id,
                limit=50
            )
            
            sharing_events = len([m for m in agent_memories if "shared knowledge" in m.get("content", "").lower()])
            assistance_events = len([m for m in agent_memories if "assistance" in m.get("content", "").lower()])
            
            collaboration_metrics["agent_details"].append({
                "agent_name": agent.agent_name,
                "specialization": agent.specialization,
                "total_contributions": len(agent_memories),
                "knowledge_sharing": sharing_events,
                "assistance_provided": assistance_events
            })
            
            collaboration_metrics["knowledge_sharing_events"] += sharing_events
            collaboration_metrics["assistance_requests"] += assistance_events
        
        # Calculate collaboration effectiveness
        total_interactions = (collaboration_metrics["knowledge_sharing_events"] + 
                            collaboration_metrics["assistance_requests"])
        if total_interactions > 0:
            collaboration_metrics["collaboration_effectiveness"] = min(100, 
                (total_interactions / len(agents)) * 25
            )
        
        print(f"📊 Generated collaboration report for session: {session_id}")
        return collaboration_metrics
    
    async def end_collaboration_session(self, session_id: str):
        """End a collaboration session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # End session for all participating agents
            for agent in session["agents"]:
                await agent.end_collaboration_session(f"Multi-agent collaboration on: {session['task']}")
            
            # Move to history
            session["end_time"] = datetime.now()
            session["status"] = "completed"
            self.collaboration_history.append(session)
            
            del self.active_sessions[session_id]
            print(f"🏁 Ended collaboration session: {session_id}")

async def demo_multi_agent_collaboration():
    """Demonstrate the multi-agent collaboration system"""
    
    print("="*70)
    print("🤖 Multi-Agent Collaboration Demo")
    print("="*70)
    
    # Initialize collaboration system
    collaboration_system = MultiAgentCollaborationSystem()
    
    try:
        # Create specialized agents
        print("\n🔧 Creating Specialized Agents")
        print("-" * 35)
        
        # Research specialist
        research_agent = CollaborativeAgent(
            agent_id="agent_research_01",
            agent_name="Dr. ResearchBot",
            specialization="research and analysis"
        )
        await research_agent.initialize_agent([
            "data_analysis", "literature_review", "hypothesis_testing", "report_writing"
        ])
        
        # Technical specialist
        tech_agent = CollaborativeAgent(
            agent_id="agent_tech_01", 
            agent_name="TechExpert",
            specialization="technical implementation"
        )
        await tech_agent.initialize_agent([
            "software_development", "system_architecture", "debugging", "optimization"
        ])
        
        # Creative specialist
        creative_agent = CollaborativeAgent(
            agent_id="agent_creative_01",
            agent_name="CreativeGenius", 
            specialization="creative problem solving"
        )
        await creative_agent.initialize_agent([
            "ideation", "design_thinking", "innovation", "user_experience"
        ])
        
        # Register agents
        collaboration_system.register_agent(research_agent)
        collaboration_system.register_agent(tech_agent)
        collaboration_system.register_agent(creative_agent)
        
        # Demo 1: Orchestrated Collaboration
        print("\n🎯 Demo 1: Orchestrated Collaboration")
        print("-" * 45)
        
        task = "Develop an innovative AI-powered learning platform"
        collaboration_result = await collaboration_system.orchestrate_collaboration(
            task=task,
            required_specializations=["research", "technical", "creative"]
        )
        
        session_id = collaboration_result["session_id"]
        print(f"Task: {task}")
        print(f"Session ID: {session_id}")
        print(f"Participating agents: {', '.join(collaboration_result['participating_agents'])}")
        
        # Demo 2: Knowledge Sharing
        print("\n📤 Demo 2: Knowledge Sharing Between Agents")
        print("-" * 50)
        
        # Research agent shares findings
        await research_agent.share_knowledge(
            "Recent studies show that personalized learning increases retention by 40%. Adaptive algorithms are key to success.",
            target_agents=["agent_tech_01", "agent_creative_01"],
            knowledge_type="research_findings"
        )
        
        # Tech agent shares technical insights
        await tech_agent.share_knowledge(
            "For scalable AI platforms, microservices architecture with containerization provides optimal performance and maintainability.",
            target_agents=["agent_research_01", "agent_creative_01"],
            knowledge_type="technical_insights"
        )
        
        # Creative agent shares design principles
        await creative_agent.share_knowledge(
            "User-centered design principles suggest intuitive interfaces with gamification elements enhance engagement significantly.",
            target_agents=["agent_research_01", "agent_tech_01"],
            knowledge_type="design_principles"
        )
        
        # Demo 3: Knowledge Exchange Facilitation
        print("\n🔄 Demo 3: Facilitated Knowledge Exchange")
        print("-" * 48)
        
        knowledge_requests = [
            {"agent_id": "agent_research_01", "topic": "user interface design"},
            {"agent_id": "agent_tech_01", "topic": "learning algorithms"},
            {"agent_id": "agent_creative_01", "topic": "performance optimization"}
        ]
        
        exchange_results = await collaboration_system.facilitate_knowledge_exchange(
            session_id, knowledge_requests
        )
        
        print(f"Knowledge exchanges: {len(exchange_results['exchanges'])}")
        print(f"Total knowledge shared: {exchange_results['total_knowledge_shared']}")
        print(f"Collaboration score: {exchange_results['collaboration_score']}")
        
        # Demo 4: Collaborative Problem Solving
        print("\n🎯 Demo 4: Collaborative Problem Solving")
        print("-" * 48)
        
        problem = "How to ensure the learning platform adapts to different learning styles while maintaining high performance?"
        
        problem_solution = await collaboration_system.resolve_collaborative_problem(
            session_id, problem
        )
        
        print(f"Problem: {problem}")
        print(f"Primary solvers: {', '.join(problem_solution['primary_solvers'])}")
        print(f"Individual solutions: {len(problem_solution['individual_solutions'])}")
        print(f"Collaborative solution preview: {problem_solution['collaborative_solution'][:200]}...")
        
        # Demo 5: Agent Assistance Requests
        print("\n🆘 Demo 5: Agent Assistance Requests")
        print("-" * 44)
        
        # Creative agent requests technical assistance
        assistance_request = await creative_agent.request_assistance(
            "I need help implementing responsive design patterns that work across all devices",
            preferred_agents=["agent_tech_01"]
        )
        
        print(f"Assistance requested by: {creative_agent.agent_name}")
        print(f"Problem: {assistance_request['problem']}")
        print(f"Relevant knowledge found: {len(assistance_request['relevant_knowledge'])}")
        
        # Tech agent provides assistance
        assistance_response = await tech_agent.provide_assistance(
            assistance_request["request_id"],
            assistance_request["problem"],
            "Use CSS Grid and Flexbox with media queries. Implement mobile-first approach with breakpoints at 768px and 1024px. Consider using CSS frameworks like Tailwind for rapid development."
        )
        
        print(f"Assistance provided by: {assistance_response['providing_agent']}")
        print(f"Solution confidence: {assistance_response['confidence']}")
        
        # Demo 6: Learning from Collaboration
        print("\n🧠 Demo 6: Collaborative Learning")
        print("-" * 40)
        
        # Simulate collaboration data
        collaboration_data = {
            "successful_collaborations": 1,
            "knowledge_shared": 3,
            "assistance_provided": 1
        }
        
        # Each agent learns from the collaboration
        for agent in [research_agent, tech_agent, creative_agent]:
            await agent.learn_from_collaboration(collaboration_data)
        
        # Demo 7: Shared Knowledge Retrieval
        print("\n📚 Demo 7: Shared Knowledge Retrieval")
        print("-" * 45)
        
        # Each agent retrieves shared knowledge on AI platforms
        for agent in [research_agent, tech_agent, creative_agent]:
            shared_knowledge = await agent.get_shared_knowledge("AI learning platform", limit=5)
            print(f"{agent.agent_name} found {len(shared_knowledge)} shared knowledge items")
        
        # Demo 8: Collaboration Report
        print("\n📊 Demo 8: Collaboration Report Generation")
        print("-" * 52)
        
        collaboration_report = await collaboration_system.generate_collaboration_report(session_id)
        
        print(f"Session duration: {collaboration_report['duration']:.1f} minutes")
        print(f"Participating agents: {collaboration_report['participating_agents']}")
        print(f"Knowledge sharing events: {collaboration_report['knowledge_sharing_events']}")
        print(f"Assistance requests: {collaboration_report['assistance_requests']}")
        print(f"Collaboration effectiveness: {collaboration_report['collaboration_effectiveness']:.1f}%")
        
        print(f"\nAgent Contributions:")
        for agent_detail in collaboration_report["agent_details"]:
            print(f"  {agent_detail['agent_name']}: {agent_detail['total_contributions']} contributions")
            print(f"    Knowledge sharing: {agent_detail['knowledge_sharing']}")
            print(f"    Assistance provided: {agent_detail['assistance_provided']}")
        
        # End collaboration session
        await collaboration_system.end_collaboration_session(session_id)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ Multi-Agent Collaboration Demo Complete!")
    print("="*70)
    print("Key features demonstrated:")
    print("• Multi-agent system orchestration")
    print("• Shared memory spaces and knowledge exchange")
    print("• Collaborative problem-solving workflows")
    print("• Agent specialization and expertise matching")
    print("• Cross-agent learning and assistance")
    print("• Comprehensive collaboration analytics")

if __name__ == "__main__":
    asyncio.run(demo_multi_agent_collaboration())

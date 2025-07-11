"""
Conversation Continuity Example

This example demonstrates how NeuronMemory enables seamless conversation continuity:
- Resume conversations across sessions
- Maintain context over time gaps
- Reference previous discussions
- Build on past interactions
- Track conversation threads

Shows beginner-friendly patterns for conversation persistence.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
from neuron_memory import NeuronMemoryAPI

class ConversationContinuityAgent:
    """Agent demonstrating conversation continuity with NeuronMemory"""
    
    def __init__(self, agent_name="ChatBot"):
        self.memory = NeuronMemoryAPI()
        self.agent_name = agent_name
        self.current_session = None
        
    async def start_session(self, user_id: str, topic: str = "general") -> str:
        """Start a new conversation session"""
        self.current_session = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
        
        await self.memory.start_session(
            session_id=self.current_session,
            user_id=user_id,
            task=f"Conversation about {topic}",
            domain="chat"
        )
        
        # Check for previous conversations
        greeting = await self._create_contextual_greeting(user_id)
        
        # Store session start
        await self._store_conversation_turn(
            "session_start", greeting, user_id, "greeting"
        )
        
        return greeting
    
    async def continue_conversation(self, user_message: str, user_id: str) -> Dict[str, Any]:
        """Continue conversation with memory context"""
        
        # Get conversation history and context
        conversation_context = await self.memory.get_context_for_llm(
            query=user_message,
            user_id=user_id,
            session_id=self.current_session,
            max_context_length=800
        )
        
        # Analyze for conversation threads
        thread_analysis = await self._analyze_conversation_thread(user_message, user_id)
        
        # Generate contextual response
        response = await self._generate_contextual_response(
            user_message, conversation_context, thread_analysis, user_id
        )
        
        # Store the conversation turn
        await self._store_conversation_turn(
            user_message, response, user_id, "conversation"
        )
        
        return {
            "response": response,
            "context_used": bool(conversation_context),
            "thread_references": thread_analysis["references"],
            "continuation_type": thread_analysis["type"],
            "context_length": len(conversation_context) if conversation_context else 0
        }
    
    async def _create_contextual_greeting(self, user_id: str) -> str:
        """Create greeting based on conversation history"""
        try:
            # Get recent conversations
            recent_conversations = await self.memory.search_memories(
                query=f"user {user_id} conversation session",
                user_id=user_id,
                memory_types=["episodic"],
                limit=5
            )
            
            if not recent_conversations:
                return f"Hello! I'm {self.agent_name}. How can I help you today?"
            
            # Get the most recent conversation topic
            last_conversation = recent_conversations[0]
            last_content = last_conversation.get("content", "").lower()
            
            # Determine time since last conversation
            last_time = last_conversation.get("created_at")
            greeting = f"Hello again! "
            
            # Reference previous topics if relevant
            if "project" in last_content:
                greeting += "I remember we were discussing your project. How's that going?"
            elif "learning" in last_content:
                greeting += "How has your learning journey been progressing?"
            elif "problem" in last_content or "issue" in last_content:
                greeting += "I hope that issue we discussed got resolved. What's on your mind today?"
            else:
                greeting += "Good to see you back! What would you like to talk about today?"
            
            return greeting
            
        except Exception as e:
            print(f"⚠️ Error creating contextual greeting: {e}")
            return f"Hello! I'm {self.agent_name}. How can I help you today?"
    
    async def _analyze_conversation_thread(self, message: str, user_id: str) -> Dict[str, Any]:
        """Analyze message for conversation thread references"""
        
        analysis = {
            "type": "new_topic",
            "references": [],
            "continuation_signals": [],
            "topic_shift": False
        }
        
        message_lower = message.lower()
        
        # Check for continuation signals
        continuation_phrases = [
            "as we discussed", "like we talked about", "from before", 
            "remember when", "you mentioned", "continuing", "also",
            "speaking of", "regarding", "about that"
        ]
        
        for phrase in continuation_phrases:
            if phrase in message_lower:
                analysis["continuation_signals"].append(phrase)
                analysis["type"] = "continuation"
        
        # Check for reference to previous topics
        try:
            previous_topics = await self.memory.search_memories(
                query=f"user {user_id} discussed topic subject",
                user_id=user_id,
                memory_types=["semantic", "episodic"],
                limit=10
            )
            
            for memory in previous_topics:
                memory_content = memory["content"].lower()
                # Simple keyword matching for topic references
                common_words = set(message_lower.split()) & set(memory_content.split())
                if len(common_words) >= 2:  # At least 2 words in common
                    analysis["references"].append(memory["content"][:100])
                    analysis["type"] = "reference"
        
        except Exception:
            pass
        
        # Check for topic shifts
        topic_shift_phrases = ["by the way", "changing topics", "something else", "different question"]
        if any(phrase in message_lower for phrase in topic_shift_phrases):
            analysis["topic_shift"] = True
            analysis["type"] = "topic_shift"
        
        return analysis
    
    async def _generate_contextual_response(
        self, 
        message: str, 
        context: str, 
        thread_analysis: Dict, 
        user_id: str
    ) -> str:
        """Generate response that maintains conversation continuity"""
        
        continuation_type = thread_analysis["type"]
        
        # Base response based on continuation type
        if continuation_type == "continuation":
            response_start = "Yes, building on our previous discussion, "
        elif continuation_type == "reference":
            response_start = "I remember we talked about this before. "
        elif continuation_type == "topic_shift":
            response_start = "Sure, let's talk about something new. "
        else:
            response_start = ""
        
        # Simple response generation (in production, use an LLM)
        message_lower = message.lower()
        
        if "how" in message_lower and "project" in message_lower:
            response = response_start + "Projects can be challenging but rewarding. What specific aspect would you like to discuss?"
        elif "learn" in message_lower or "understand" in message_lower:
            response = response_start + "Learning is a great journey! What topic interests you most?"
        elif "problem" in message_lower or "issue" in message_lower:
            response = response_start + "I'd be happy to help you work through this problem. Can you tell me more details?"
        elif "thank" in message_lower:
            response = "You're very welcome! I'm glad I could help. Is there anything else you'd like to discuss?"
        else:
            response = response_start + "That's an interesting point. Could you tell me more about what you're thinking?"
        
        # Add context references if available
        if context and continuation_type in ["continuation", "reference"]:
            response += " Based on what we've discussed, this connects well with your previous interests."
        
        # Reference specific previous topics if found
        if thread_analysis["references"]:
            response += f" This reminds me of when we talked about similar topics before."
        
        return response
    
    async def _store_conversation_turn(
        self, 
        user_message: str, 
        agent_response: str, 
        user_id: str, 
        turn_type: str
    ):
        """Store conversation turn in memory"""
        try:
            # Store as episodic memory
            conversation_content = f"User: {user_message}\nAgent: {agent_response}"
            
            await self.memory.create_episodic_memory(
                content=conversation_content,
                participants=[user_id, self.agent_name],
                location="chat_conversation",
                emotional_state="conversational",
                user_id=user_id,
                session_id=self.current_session
            )
            
            # Store topics discussed
            if turn_type == "conversation":
                # Extract key topics (simplified)
                topics = []
                message_words = user_message.lower().split()
                topic_keywords = ["project", "learning", "problem", "work", "study", "issue"]
                
                for keyword in topic_keywords:
                    if keyword in message_words:
                        topics.append(keyword)
                
                if topics:
                    topic_memory = f"User discussed topics: {', '.join(topics)}"
                    await self.memory.create_semantic_memory(
                        content=topic_memory,
                        domain="conversation_topics",
                        confidence=0.7,
                        user_id=user_id,
                        session_id=self.current_session
                    )
            
        except Exception as e:
            print(f"⚠️ Error storing conversation turn: {e}")
    
    async def get_conversation_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of conversation history with user"""
        try:
            all_conversations = await self.memory.search_memories(
                query=f"user {user_id} conversation",
                user_id=user_id,
                limit=30
            )
            
            topic_memories = await self.memory.search_memories(
                query=f"user {user_id} topics discussed",
                user_id=user_id,
                memory_types=["semantic"],
                limit=15
            )
            
            summary = {
                "total_conversations": len(all_conversations),
                "topics_discussed": [],
                "conversation_patterns": [],
                "recent_themes": []
            }
            
            # Extract topics
            for memory in topic_memories:
                content = memory["content"]
                if "topics:" in content:
                    topics = content.split("topics:")[-1].strip()
                    summary["topics_discussed"].extend(topics.split(", "))
            
            # Get recent conversation themes
            for memory in all_conversations[:5]:  # Last 5 conversations
                content = memory["content"].lower()
                if "project" in content:
                    summary["recent_themes"].append("project discussion")
                elif "learning" in content:
                    summary["recent_themes"].append("learning topics")
                elif "problem" in content:
                    summary["recent_themes"].append("problem solving")
            
            # Remove duplicates
            summary["topics_discussed"] = list(set(summary["topics_discussed"]))
            summary["recent_themes"] = list(set(summary["recent_themes"]))
            
            return summary
            
        except Exception as e:
            print(f"⚠️ Error generating conversation summary: {e}")
            return {"error": str(e)}
    
    async def end_session(self, user_id: str):
        """End conversation session"""
        if self.current_session:
            # Store session end
            await self._store_conversation_turn(
                "session_end", "Conversation session ended", user_id, "goodbye"
            )
            
            await self.memory.end_session(self.current_session)
            print(f"🏁 Ended conversation session: {self.current_session}")
            self.current_session = None

async def demo_conversation_continuity():
    """Demonstrate conversation continuity across multiple sessions"""
    
    print("="*70)
    print("💬 Conversation Continuity Demo")
    print("="*70)
    print("This demo shows how conversations can continue seamlessly across sessions")
    
    agent = ConversationContinuityAgent("ChatBot")
    user_id = "demo_user_alice"
    
    try:
        # Session 1: Initial conversation about a project
        print("\n🔄 Session 1: Starting a Project Discussion")
        print("-" * 45)
        
        greeting1 = await agent.start_session(user_id, "project_planning")
        print(f"Agent: {greeting1}")
        
        response1 = await agent.continue_conversation(
            "Hi! I'm starting a new web development project and need some guidance.", 
            user_id
        )
        print(f"User: Hi! I'm starting a new web development project and need some guidance.")
        print(f"Agent: {response1['response']}")
        print(f"Context used: {response1['context_used']}")
        
        response2 = await agent.continue_conversation(
            "It's a React application with a Node.js backend. I'm learning as I go.", 
            user_id
        )
        print(f"User: It's a React application with a Node.js backend. I'm learning as I go.")
        print(f"Agent: {response2['response']}")
        
        await agent.end_session(user_id)
        
        # Simulate time gap (in real usage, this would be hours/days later)
        print("\n⏰ [Simulating time gap - user returns later]")
        
        # Session 2: Returning to continue the project discussion
        print("\n🔄 Session 2: Returning to Continue Discussion")
        print("-" * 50)
        
        greeting2 = await agent.start_session(user_id, "project_update")
        print(f"Agent: {greeting2}")
        
        response3 = await agent.continue_conversation(
            "I made some progress on the project we discussed. The React components are working well!", 
            user_id
        )
        print(f"User: I made some progress on the project we discussed. The React components are working well!")
        print(f"Agent: {response3['response']}")
        print(f"Continuation type: {response3['continuation_type']}")
        print(f"Thread references found: {len(response3['thread_references'])}")
        
        response4 = await agent.continue_conversation(
            "But I'm having some issues with the backend API. Remember you mentioned Node.js before?", 
            user_id
        )
        print(f"User: But I'm having some issues with the backend API. Remember you mentioned Node.js before?")
        print(f"Agent: {response4['response']}")
        print(f"References previous conversation: {response4['continuation_type'] == 'reference'}")
        
        await agent.end_session(user_id)
        
        # Session 3: Topic shift but still building on relationship
        print("\n🔄 Session 3: New Topic but Maintained Relationship")
        print("-" * 55)
        
        greeting3 = await agent.start_session(user_id, "new_topic")
        print(f"Agent: {greeting3}")
        
        response5 = await agent.continue_conversation(
            "By the way, I want to ask about something different - learning resources for Python.", 
            user_id
        )
        print(f"User: By the way, I want to ask about something different - learning resources for Python.")
        print(f"Agent: {response5['response']}")
        print(f"Topic shift detected: {'topic_shift' in response5['continuation_type']}")
        
        response6 = await agent.continue_conversation(
            "Thank you for all your help! You've been really supportive with my learning journey.", 
            user_id
        )
        print(f"User: Thank you for all your help! You've been really supportive with my learning journey.")
        print(f"Agent: {response6['response']}")
        
        await agent.end_session(user_id)
        
        # Show conversation summary
        print("\n📊 Conversation History Summary")
        print("-" * 35)
        
        summary = await agent.get_conversation_summary(user_id)
        print(f"Total conversations: {summary['total_conversations']}")
        print(f"Topics discussed: {', '.join(summary['topics_discussed'][:5])}")  # Show first 5
        print(f"Recent themes: {', '.join(summary['recent_themes'])}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ Conversation Continuity Demo Complete!")
    print("="*70)
    print("Key features demonstrated:")
    print("• Contextual greetings based on history")
    print("• Reference to previous conversations")
    print("• Topic thread continuation")
    print("• Seamless session transitions")
    print("• Memory-driven relationship building")
    print("• Topic shift handling")

if __name__ == "__main__":
    asyncio.run(demo_conversation_continuity())

"""
Personal AI Assistant Example

This example demonstrates building a personal AI assistant that:
- Remembers user preferences and habits
- Learns from conversations over time
- Provides personalized responses
- Maintains context across sessions
- Adapts communication style

Shows intermediate-level NeuronMemory usage patterns.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any
from neuron_memory import NeuronMemoryAPI

class PersonalAIAssistant:
    """Memory-driven personal AI assistant"""
    
    def __init__(self, assistant_name="Alex"):
        self.memory = NeuronMemoryAPI()
        self.assistant_name = assistant_name
        self.current_session = None
        
    async def start_conversation(self, user_id: str, context: str = "general") -> str:
        """Start a new conversation session"""
        session_id = f"assistant_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
        
        await self.memory.start_session(
            session_id=session_id,
            user_id=user_id,
            task=f"Personal assistant conversation - {context}",
            domain="personal_assistance"
        )
        
        self.current_session = session_id
        
        # Get user's name and preferences for greeting
        user_info = await self._get_user_info(user_id)
        greeting = await self._create_personalized_greeting(user_info)
        
        # Store this interaction
        await self._store_interaction(
            "session_start", greeting, user_id, session_id, "greeting"
        )
        
        return greeting
    
    async def respond(self, user_message: str, user_id: str) -> Dict[str, Any]:
        """Generate a personalized response to user message"""
        
        # Get relevant memory context
        memory_context = await self.memory.get_context_for_llm(
            query=user_message,
            user_id=user_id,
            session_id=self.current_session,
            max_context_length=1000
        )
        
        # Analyze user message for learning opportunities
        analysis = await self._analyze_user_message(user_message, user_id)
        
        # Generate response based on context and user preferences
        response = await self._generate_personalized_response(
            user_message, memory_context, analysis, user_id
        )
        
        # Store the interaction for learning
        await self._store_interaction(
            user_message, response, user_id, self.current_session, "conversation"
        )
        
        # Update user model based on interaction
        await self._update_user_model(user_message, analysis, user_id)
        
        return {
            "response": response,
            "analysis": analysis,
            "memory_used": bool(memory_context),
            "context_length": len(memory_context) if memory_context else 0
        }
    
    async def _get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user information from memory"""
        try:
            user_memories = await self.memory.search_memories(
                query=f"user {user_id} name preferences profile",
                user_id=user_id,
                memory_types=["social", "semantic"],
                limit=10
            )
            
            info = {
                "name": None,
                "communication_style": "friendly",
                "interests": []
            }
            
            for memory in user_memories:
                content = memory["content"].lower()
                if "name is" in content or "call me" in content:
                    words = content.split()
                    for i, word in enumerate(words):
                        if word in ["name", "called"] and i < len(words) - 1:
                            info["name"] = words[i + 1].strip(".,!?")
                
                if "prefers" in content or "likes" in content:
                    info["interests"].append(memory["content"])
                
                if "formal" in content:
                    info["communication_style"] = "formal"
                elif "casual" in content:
                    info["communication_style"] = "casual"
            
            return info
            
        except Exception as e:
            print(f"⚠️ Error getting user info: {e}")
            return {"name": None, "communication_style": "friendly", "interests": []}
    
    async def _create_personalized_greeting(self, user_info: Dict) -> str:
        """Create a personalized greeting based on user info"""
        name_part = ""
        if user_info.get("name"):
            name_part = f", {user_info['name']}"
        
        style = user_info.get("communication_style", "friendly")
        
        if style == "formal":
            return f"Good day{name_part}. How may I assist you today?"
        elif style == "casual":
            return f"Hey{name_part}! What's up? How can I help?"
        else:
            return f"Hello{name_part}! I'm {self.assistant_name}, your personal assistant. How can I help you today?"
    
    async def _analyze_user_message(self, message: str, user_id: str) -> Dict[str, Any]:
        """Analyze user message for patterns and learning opportunities"""
        analysis = {
            "intent": "general",
            "emotion": "neutral",
            "topics": [],
            "preferences_mentioned": []
        }
        
        message_lower = message.lower()
        
        # Intent detection
        if any(word in message_lower for word in ["help", "how", "what", "when", "where", "why"]):
            analysis["intent"] = "question"
        elif any(word in message_lower for word in ["schedule", "remind", "meeting", "appointment"]):
            analysis["intent"] = "scheduling"
        elif any(word in message_lower for word in ["i like", "i prefer", "i want", "my favorite"]):
            analysis["intent"] = "preference"
        elif any(word in message_lower for word in ["thank", "thanks", "appreciate"]):
            analysis["intent"] = "gratitude"
        
        # Extract preferences
        if "i like" in message_lower:
            preference_start = message_lower.find("i like") + 6
            preference = message[preference_start:].split(".")[0].strip()
            analysis["preferences_mentioned"].append(f"likes {preference}")
        
        if "i prefer" in message_lower:
            preference_start = message_lower.find("i prefer") + 8
            preference = message[preference_start:].split(".")[0].strip()
            analysis["preferences_mentioned"].append(f"prefers {preference}")
        
        return analysis
    
    async def _generate_personalized_response(
        self, 
        user_message: str, 
        memory_context: str, 
        analysis: Dict, 
        user_id: str
    ) -> str:
        """Generate a personalized response"""
        intent = analysis["intent"]
        
        responses = {
            "question": "That's a great question! Let me help you with that.",
            "scheduling": "Let me help you with your schedule.",
            "preference": "Thanks for letting me know your preference! I'll remember that.",
            "gratitude": "You're very welcome! Happy to help!",
            "general": "I understand. Tell me more about that."
        }
        
        base_response = responses.get(intent, responses["general"])
        
        # Personalize based on memory context
        if memory_context and "likes" in memory_context:
            base_response += " I remember you mentioned liking similar things before."
        
        return base_response
    
    async def _store_interaction(
        self, 
        user_message: str, 
        assistant_response: str, 
        user_id: str, 
        session_id: str, 
        interaction_type: str
    ):
        """Store the interaction in memory"""
        try:
            conversation_content = f"User: {user_message}\nAssistant: {assistant_response}"
            
            await self.memory.create_episodic_memory(
                content=conversation_content,
                participants=[user_id, self.assistant_name],
                location="personal_assistant_chat",
                emotional_state="helpful",
                user_id=user_id,
                session_id=session_id
            )
            
        except Exception as e:
            print(f"⚠️ Error storing interaction: {e}")
    
    async def _update_user_model(self, user_message: str, analysis: Dict, user_id: str):
        """Update the user model based on the interaction"""
        try:
            # Store preferences
            for preference in analysis["preferences_mentioned"]:
                await self.memory.create_social_memory(
                    content=f"User {preference}",
                    person_id=user_id,
                    relationship_type="user",
                    user_id=user_id,
                    session_id=self.current_session
                )
            
        except Exception as e:
            print(f"⚠️ Error updating user model: {e}")
    
    async def end_conversation(self, user_id: str):
        """End the current conversation session"""
        if self.current_session:
            await self.memory.end_session(self.current_session)
            print(f"🏁 Ended conversation session: {self.current_session}")
            self.current_session = None

async def demo_personal_assistant():
    """Demonstrate the personal AI assistant"""
    
    print("="*60)
    print("🤖 Personal AI Assistant Demo")
    print("="*60)
    
    assistant = PersonalAIAssistant("Alex")
    user_id = "demo_user_sarah"
    
    # Session 1: First Meeting
    print("\n🔄 Session 1: First Meeting")
    print("-" * 30)
    
    greeting = await assistant.start_conversation(user_id, "first_meeting")
    print(f"Assistant: {greeting}")
    
    response1 = await assistant.respond("Hi! My name is Sarah and I like coffee and reading books.", user_id)
    print(f"User: Hi! My name is Sarah and I like coffee and reading books.")
    print(f"Assistant: {response1['response']}")
    
    response2 = await assistant.respond("Can you help me plan my day?", user_id)
    print(f"User: Can you help me plan my day?")
    print(f"Assistant: {response2['response']}")
    
    await assistant.end_conversation(user_id)
    
    # Session 2: Return Visit
    print("\n🔄 Session 2: Return Visit")
    print("-" * 30)
    
    greeting2 = await assistant.start_conversation(user_id, "return_visit")
    print(f"Assistant: {greeting2}")
    
    response3 = await assistant.respond("I prefer morning meetings and I really love Italian food.", user_id)
    print(f"User: I prefer morning meetings and I really love Italian food.")
    print(f"Assistant: {response3['response']}")
    
    await assistant.end_conversation(user_id)
    
    print("\n" + "="*60)
    print("✅ Personal Assistant Demo Complete!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(demo_personal_assistant()) 
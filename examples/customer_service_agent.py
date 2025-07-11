"""
Customer Service Agent Example

This example demonstrates building a customer service AI agent that:
- Remembers customer interaction history
- Tracks support tickets and resolutions
- Learns customer preferences and communication style
- Provides personalized support experiences
- Escalates issues based on customer history

Shows practical enterprise application of NeuronMemory.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from neuron_memory import NeuronMemoryAPI

class CustomerServiceAgent:
    """AI customer service agent with memory-driven personalization"""
    
    def __init__(self, agent_name="Alex"):
        self.memory = NeuronMemoryAPI()
        self.agent_name = agent_name
        self.current_session = None
        
        # Sample knowledge base (replace with your actual KB)
        self.knowledge_base = {
            "billing": {
                "payment_issues": "For payment issues, check your payment method, ensure sufficient funds, and verify billing address matches your card.",
                "refund_policy": "Refunds are processed within 5-7 business days. Premium users get priority processing.",
                "subscription_changes": "You can upgrade/downgrade your subscription anytime. Changes take effect at the next billing cycle."
            },
            "technical": {
                "login_problems": "Reset your password using the 'Forgot Password' link. Clear browser cache if issues persist.",
                "performance_issues": "Try refreshing the page, check your internet connection, or try a different browser.",
                "feature_requests": "We track all feature requests. Premium users' requests get higher priority in our roadmap."
            },
            "account": {
                "profile_updates": "Update your profile in Account Settings. Some changes may require email verification.",
                "data_export": "You can export your data anytime from the Privacy section in your account settings.",
                "account_deletion": "Account deletion is permanent and cannot be undone. We can help with account suspension instead."
            }
        }
    
    async def start_customer_session(self, customer_id: str, contact_reason: str = "general") -> str:
        """Start a new customer service session"""
        self.current_session = f"support_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{customer_id}"
        
        await self.memory.start_session(
            session_id=self.current_session,
            user_id=customer_id,
            task=f"Customer support - {contact_reason}",
            domain="customer_service"
        )
        
        # Get customer profile and history
        customer_profile = await self._get_customer_profile(customer_id)
        greeting = await self._create_personalized_greeting(customer_profile, contact_reason)
        
        # Log session start
        await self._log_interaction(
            "session_start", greeting, customer_id, "greeting", {}
        )
        
        print(f"🎧 Started customer support session: {self.current_session}")
        return greeting
    
    async def handle_customer_message(self, message: str, customer_id: str) -> Dict[str, Any]:
        """Handle customer message with context and personalization"""
        
        print(f"\n💬 Customer: {message}")
        
        # Analyze customer message
        message_analysis = await self._analyze_customer_message(message)
        
        # Get relevant customer history
        customer_context = await self.memory.get_context_for_llm(
            query=message,
            user_id=customer_id,
            session_id=self.current_session,
            max_context_length=800
        )
        
        # Get customer profile for personalization
        customer_profile = await self._get_customer_profile(customer_id)
        
        # Check for escalation needs
        escalation_needed = await self._check_escalation_needs(message_analysis, customer_profile)
        
        # Generate appropriate response
        if escalation_needed:
            response = await self._handle_escalation(message_analysis, customer_profile)
        else:
            response = await self._generate_support_response(
                message, message_analysis, customer_context, customer_profile
            )
        
        # Log the interaction
        await self._log_interaction(
            message, response, customer_id, message_analysis["category"], message_analysis
        )
        
        # Update customer profile
        await self._update_customer_profile(message_analysis, customer_id)
        
        return {
            "response": response,
            "category": message_analysis["category"],
            "sentiment": message_analysis["sentiment"],
            "urgency": message_analysis["urgency"],
            "escalation_needed": escalation_needed,
            "context_used": bool(customer_context)
        }
    
    async def _get_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        """Get comprehensive customer profile from memory"""
        try:
            # Get customer information
            customer_info = await self.memory.search_memories(
                query=f"customer {customer_id} profile tier preferences",
                user_id=customer_id,
                memory_types=["social", "semantic"],
                limit=15
            )
            
            # Get recent support history
            support_history = await self.memory.search_memories(
                query=f"customer {customer_id} support issue ticket",
                user_id=customer_id,
                memory_types=["episodic"],
                limit=10
            )
            
            profile = {
                "customer_tier": "standard",
                "communication_style": "friendly",
                "technical_level": "basic",
                "preferred_contact": "chat",
                "recent_issues": [],
                "satisfaction_history": [],
                "escalation_count": 0,
                "total_interactions": len(support_history)
            }
            
            # Extract profile information
            for memory in customer_info:
                content = memory["content"].lower()
                
                if "premium" in content or "vip" in content:
                    profile["customer_tier"] = "premium"
                elif "enterprise" in content:
                    profile["customer_tier"] = "enterprise"
                
                if "technical" in content or "developer" in content:
                    profile["technical_level"] = "advanced"
                elif "beginner" in content or "basic" in content:
                    profile["technical_level"] = "basic"
                
                if "formal" in content:
                    profile["communication_style"] = "formal"
                elif "casual" in content:
                    profile["communication_style"] = "casual"
            
            # Extract recent issues
            for memory in support_history[-5:]:  # Last 5 interactions
                if "issue" in memory["content"].lower():
                    profile["recent_issues"].append(memory["content"][:100])
            
            return profile
            
        except Exception as e:
            print(f"⚠️ Error getting customer profile: {e}")
            return {
                "customer_tier": "standard",
                "communication_style": "friendly",
                "technical_level": "basic",
                "recent_issues": [],
                "total_interactions": 0
            }
    
    async def _create_personalized_greeting(self, customer_profile: Dict, contact_reason: str) -> str:
        """Create personalized greeting based on customer profile"""
        
        tier = customer_profile.get("customer_tier", "standard")
        style = customer_profile.get("communication_style", "friendly")
        total_interactions = customer_profile.get("total_interactions", 0)
        
        # Tier-based greetings
        if tier == "premium":
            base_greeting = f"Hello! I'm {self.agent_name}, your premium support specialist."
        elif tier == "enterprise":
            base_greeting = f"Good day! I'm {self.agent_name} from your dedicated enterprise support team."
        else:
            base_greeting = f"Hi there! I'm {self.agent_name}, your customer support agent."
        
        # Add returning customer recognition
        if total_interactions > 0:
            if style == "formal":
                base_greeting += " I see you've contacted us before. How may I assist you today?"
            else:
                base_greeting += " Great to hear from you again! What can I help you with today?"
        else:
            if style == "formal":
                base_greeting += " How may I assist you today?"
            else:
                base_greeting += " How can I help you today?"
        
        # Add context-specific information
        if contact_reason != "general":
            base_greeting += f" I understand you're contacting us about {contact_reason.replace('_', ' ')}."
        
        return base_greeting
    
    async def _analyze_customer_message(self, message: str) -> Dict[str, Any]:
        """Analyze customer message for category, sentiment, and urgency"""
        
        analysis = {
            "category": "general",
            "sentiment": "neutral",
            "urgency": "normal",
            "keywords": [],
            "frustration_indicators": []
        }
        
        message_lower = message.lower()
        
        # Category detection
        if any(word in message_lower for word in ["bill", "payment", "charge", "refund", "subscription"]):
            analysis["category"] = "billing"
        elif any(word in message_lower for word in ["login", "password", "error", "bug", "broken", "not working"]):
            analysis["category"] = "technical"
        elif any(word in message_lower for word in ["account", "profile", "settings", "delete", "export"]):
            analysis["category"] = "account"
        elif any(word in message_lower for word in ["feature", "request", "suggestion", "improvement"]):
            analysis["category"] = "feature_request"
        
        # Sentiment detection
        positive_words = ["great", "good", "excellent", "happy", "satisfied", "thank"]
        negative_words = ["bad", "awful", "terrible", "frustrated", "angry", "disappointed"]
        
        if any(word in message_lower for word in positive_words):
            analysis["sentiment"] = "positive"
        elif any(word in message_lower for word in negative_words):
            analysis["sentiment"] = "negative"
        
        # Urgency detection
        urgent_words = ["urgent", "asap", "immediately", "emergency", "critical", "can't work"]
        if any(word in message_lower for word in urgent_words):
            analysis["urgency"] = "high"
        
        # Frustration indicators
        frustration_words = ["again", "still", "multiple times", "keeps happening", "already tried"]
        analysis["frustration_indicators"] = [word for word in frustration_words if word in message_lower]
        
        return analysis
    
    async def _check_escalation_needs(self, message_analysis: Dict, customer_profile: Dict) -> bool:
        """Check if the issue needs escalation"""
        
        # High urgency issues
        if message_analysis["urgency"] == "high":
            return True
        
        # Frustrated premium customers
        if (customer_profile.get("customer_tier") == "premium" and 
            message_analysis["sentiment"] == "negative" and 
            message_analysis["frustration_indicators"]):
            return True
        
        # Customers with multiple recent issues
        if len(customer_profile.get("recent_issues", [])) >= 3:
            return True
        
        # Technical issues for non-technical customers
        if (message_analysis["category"] == "technical" and 
            customer_profile.get("technical_level") == "basic" and
            message_analysis["sentiment"] == "negative"):
            return True
        
        return False
    
    async def _handle_escalation(self, message_analysis: Dict, customer_profile: Dict) -> str:
        """Handle escalated issues"""
        
        tier = customer_profile.get("customer_tier", "standard")
        
        if tier == "premium":
            response = "I understand this is urgent and important to you. I'm escalating this to our premium support team who will contact you within 15 minutes."
        elif tier == "enterprise":
            response = "I'm immediately escalating this to your dedicated enterprise support manager. You should receive a call within 10 minutes."
        else:
            response = "I understand your frustration and want to get this resolved quickly. I'm escalating this to our specialist team who will prioritize your case."
        
        # Add specific escalation information
        if message_analysis["urgency"] == "high":
            response += " Given the urgent nature of this issue, it will be marked as high priority."
        
        return response
    
    async def _generate_support_response(
        self, 
        message: str, 
        message_analysis: Dict, 
        customer_context: str, 
        customer_profile: Dict
    ) -> str:
        """Generate appropriate support response"""
        
        category = message_analysis["category"]
        sentiment = message_analysis["sentiment"]
        tier = customer_profile.get("customer_tier", "standard")
        tech_level = customer_profile.get("technical_level", "basic")
        
        # Get relevant knowledge base information
        kb_info = self._get_knowledge_base_info(category, message)
        
        # Start with empathy if customer is frustrated
        if sentiment == "negative":
            response = "I understand your frustration, and I'm here to help resolve this for you. "
        else:
            response = "I'd be happy to help you with this. "
        
        # Add knowledge base information
        if kb_info:
            response += kb_info
        else:
            response += "Let me look into this specific situation for you."
        
        # Add tier-specific information
        if tier == "premium" and category == "billing":
            response += "\n\nAs a premium customer, your billing issues receive priority processing."
        elif tier == "enterprise":
            response += "\n\nFor enterprise accounts, we also offer phone support if you prefer."
        
        # Adjust technical detail based on customer level
        if tech_level == "basic" and category == "technical":
            response += "\n\nI'll provide step-by-step instructions to make this as easy as possible."
        elif tech_level == "advanced" and category == "technical":
            response += "\n\nI can also provide more technical details if you need them."
        
        # Reference previous interactions if relevant
        if customer_context and "similar" in customer_context.lower():
            response += "\n\nI see we've helped you with similar issues before. Let me check what worked last time."
        
        return response
    
    def _get_knowledge_base_info(self, category: str, message: str) -> str:
        """Get relevant information from knowledge base"""
        
        message_lower = message.lower()
        
        if category in self.knowledge_base:
            category_kb = self.knowledge_base[category]
            
            # Find most relevant subcategory
            for subcategory, info in category_kb.items():
                if any(word in message_lower for word in subcategory.split("_")):
                    return info
            
            # Return first item if no specific match
            return list(category_kb.values())[0] if category_kb else ""
        
        return ""
    
    async def _log_interaction(
        self, 
        customer_message: str, 
        agent_response: str, 
        customer_id: str, 
        category: str, 
        analysis: Dict
    ):
        """Log customer interaction in memory"""
        try:
            # Create episodic memory of the interaction
            interaction_content = f"Customer issue: {customer_message}\nAgent response: {agent_response}\nCategory: {category}"
            
            await self.memory.create_episodic_memory(
                content=interaction_content,
                participants=[customer_id, self.agent_name],
                location="customer_support_chat",
                emotional_state=analysis.get("sentiment", "neutral"),
                user_id=customer_id,
                session_id=self.current_session
            )
            
            # Store issue categorization
            if category != "general":
                await self.memory.create_semantic_memory(
                    content=f"Customer contacted support about {category}: {customer_message[:100]}",
                    domain="support_issues",
                    confidence=0.8,
                    user_id=customer_id,
                    session_id=self.current_session
                )
            
        except Exception as e:
            print(f"⚠️ Error logging interaction: {e}")
    
    async def _update_customer_profile(self, message_analysis: Dict, customer_id: str):
        """Update customer profile based on interaction"""
        try:
            # Update technical level assessment
            if message_analysis["category"] == "technical":
                technical_indicators = ["API", "database", "configuration", "integration"]
                if any(indicator in message_analysis.get("keywords", []) for indicator in technical_indicators):
                    await self.memory.create_social_memory(
                        content="Customer demonstrates advanced technical knowledge",
                        person_id=customer_id,
                        relationship_type="customer",
                        user_id=customer_id,
                        session_id=self.current_session
                    )
            
            # Track communication style
            if message_analysis["sentiment"] == "positive":
                await self.memory.create_social_memory(
                    content="Customer communicates positively and appreciates help",
                    person_id=customer_id,
                    relationship_type="customer",
                    user_id=customer_id,
                    session_id=self.current_session
                )
            
        except Exception as e:
            print(f"⚠️ Error updating customer profile: {e}")
    
    async def get_customer_summary(self, customer_id: str) -> Dict[str, Any]:
        """Generate comprehensive customer summary"""
        try:
            all_interactions = await self.memory.search_memories(
                query=f"customer {customer_id}",
                user_id=customer_id,
                limit=50
            )
            
            summary = {
                "total_interactions": len(all_interactions),
                "issue_categories": {},
                "sentiment_trend": [],
                "escalations": 0,
                "satisfaction_indicators": 0
            }
            
            for memory in all_interactions:
                content = memory["content"].lower()
                
                # Count issue categories
                for category in ["billing", "technical", "account"]:
                    if category in content:
                        summary["issue_categories"][category] = summary["issue_categories"].get(category, 0) + 1
                
                # Track sentiment
                if "positive" in content or "satisfied" in content:
                    summary["satisfaction_indicators"] += 1
                elif "escalat" in content:
                    summary["escalations"] += 1
            
            return summary
            
        except Exception as e:
            print(f"⚠️ Error generating customer summary: {e}")
            return {"error": str(e)}
    
    async def end_customer_session(self, customer_id: str, resolution_status: str = "resolved"):
        """End customer support session"""
        if self.current_session:
            # Create session summary
            summary = f"Customer support session ended with status: {resolution_status}"
            
            await self.memory.create_episodic_memory(
                content=summary,
                participants=[customer_id, self.agent_name],
                location="customer_support_chat",
                emotional_state="completed",
                user_id=customer_id,
                session_id=self.current_session
            )
            
            await self.memory.end_session(self.current_session)
            print(f"🏁 Ended customer support session: {self.current_session}")
            self.current_session = None

async def demo_customer_service():
    """Demonstrate the customer service agent"""
    
    print("="*70)
    print("🎧 Customer Service Agent Demo")
    print("="*70)
    
    agent = CustomerServiceAgent("Alex")
    customer_id = "customer_jane_doe"
    
    try:
        # Session 1: First interaction - billing issue
        print("\n🔄 Session 1: New Customer - Billing Issue")
        print("-" * 45)
        
        greeting = await agent.start_customer_session(customer_id, "billing")
        print(f"Agent: {greeting}")
        
        response1 = await agent.handle_customer_message(
            "Hi, I was charged twice for my subscription this month. Can you help me with a refund?", 
            customer_id
        )
        print(f"Agent: {response1['response']}")
        print(f"Category: {response1['category']}, Sentiment: {response1['sentiment']}")
        
        response2 = await agent.handle_customer_message(
            "Thanks! That's helpful. I'm a premium customer, does that change anything?", 
            customer_id
        )
        print(f"Customer: Thanks! That's helpful. I'm a premium customer, does that change anything?")
        print(f"Agent: {response2['response']}")
        
        await agent.end_customer_session(customer_id, "resolved")
        
        # Store customer tier information
        await agent.memory.create_social_memory(
            content="Customer is a premium tier subscriber",
            person_id=customer_id,
            relationship_type="customer",
            user_id=customer_id,
            session_id=None
        )
        
        # Session 2: Return customer - technical issue
        print("\n🔄 Session 2: Returning Premium Customer - Technical Issue")
        print("-" * 55)
        
        greeting2 = await agent.start_customer_session(customer_id, "technical")
        print(f"Agent: {greeting2}")
        
        response3 = await agent.handle_customer_message(
            "I'm having trouble logging in again. This is really frustrating as it happened last week too.", 
            customer_id
        )
        print(f"Customer: I'm having trouble logging in again. This is really frustrating as it happened last week too.")
        print(f"Agent: {response3['response']}")
        print(f"Escalation needed: {response3['escalation_needed']}")
        
        await agent.end_customer_session(customer_id, "escalated")
        
        # Session 3: Follow-up
        print("\n🔄 Session 3: Follow-up - Feature Request")
        print("-" * 40)
        
        greeting3 = await agent.start_customer_session(customer_id, "general")
        print(f"Agent: {greeting3}")
        
        response4 = await agent.handle_customer_message(
            "The login issue was resolved, thank you! I'd like to suggest a new feature for the mobile app.", 
            customer_id
        )
        print(f"Customer: The login issue was resolved, thank you! I'd like to suggest a new feature for the mobile app.")
        print(f"Agent: {response4['response']}")
        
        await agent.end_customer_session(customer_id, "resolved")
        
        # Show customer summary
        print("\n📊 Customer Profile Summary")
        print("-" * 30)
        
        summary = await agent.get_customer_summary(customer_id)
        print(f"Total interactions: {summary['total_interactions']}")
        print(f"Issue categories: {summary['issue_categories']}")
        print(f"Escalations: {summary['escalations']}")
        print(f"Satisfaction indicators: {summary['satisfaction_indicators']}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ Customer Service Agent Demo Complete!")
    print("="*70)
    print("Key features demonstrated:")
    print("• Customer profile building and recognition")
    print("• Tier-based personalized service")
    print("• Automatic escalation for frustrated customers")
    print("• Issue categorization and tracking")
    print("• Context-aware responses based on history")
    print("• Sentiment analysis and appropriate responses")

if __name__ == "__main__":
    asyncio.run(demo_customer_service())

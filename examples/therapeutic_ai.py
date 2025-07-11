"""
Therapeutic AI Example

This example demonstrates building a therapeutic AI system that:
- Recognizes emotional states and mental health patterns
- Provides therapeutic responses using various modalities
- Assesses and responds to crisis situations safely
- Tracks therapeutic progress over time
- Builds therapeutic alliance and trust
- Maintains appropriate boundaries and ethics

IMPORTANT: This is a demonstration example only. Real therapeutic AI systems require:
- Professional clinical oversight
- Ethical review and approval
- Crisis intervention protocols
- Privacy and security compliance
- Integration with human mental health professionals
"""
# C:\Users\dhanu\Downloads\NeuronMemory\neuron_memory\api\neuron_memory_api.py

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from neuron_memory import NeuronMemoryAPI

class TherapeuticAI:
    """AI therapeutic support system with emotional recognition and crisis assessment"""
    
    def __init__(self, therapist_name="Alex"):
        self.memory = NeuronMemoryAPI()
        self.therapist_name = therapist_name
        self.current_session = None
        
        # Therapeutic modalities and approaches
        self.therapeutic_modalities = {
            "cognitive_behavioral": ["thought_challenging", "behavioral_activation", "exposure_therapy", "cognitive_restructuring"],
            "mindfulness": ["meditation", "breathing_exercises", "body_scan", "present_moment_awareness"],
            "dialectical_behavioral": ["distress_tolerance", "emotion_regulation", "interpersonal_effectiveness", "mindfulness"],
            "psychodynamic": ["insight_development", "transference_exploration", "unconscious_patterns", "relationship_analysis"],
            "humanistic": ["unconditional_positive_regard", "empathic_understanding", "genuineness", "self_actualization"]
        }
        
        # Emotional state indicators
        self.emotional_indicators = {
            "positive": ["happy", "content", "excited", "grateful", "hopeful", "peaceful", "confident"],
            "anxious": ["worried", "nervous", "panicked", "stressed", "fearful", "overwhelmed", "restless"],
            "depressed": ["sad", "hopeless", "empty", "worthless", "tired", "isolated", "numb"],
            "angry": ["frustrated", "irritated", "furious", "resentful", "bitter", "hostile", "agitated"],
            "neutral": ["calm", "stable", "balanced", "okay", "fine", "normal", "steady"]
        }
        
        # Crisis indicators (for educational purposes only)
        self.crisis_indicators = [
            "suicide", "self-harm", "hurt myself", "end it all", "no point", "everyone better without me",
            "harm others", "violence", "dangerous thoughts", "can't go on", "want to die"
        ]
        
    async def start_therapeutic_session(self, user_id: str, session_type: str = "general") -> str:
        """Start a new therapeutic session with appropriate setup"""
        self.current_session = f"therapy_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
        
        await self.memory.start_session(
            session_id=self.current_session,
            user_id=user_id,
            task=f"Therapeutic support session - {session_type}",
            domain="mental_health"
        )
        
        # Get therapeutic history and relationship
        therapeutic_profile = await self._get_therapeutic_profile(user_id)
        therapeutic_alliance = await self._get_therapeutic_alliance(user_id)
        
        # Create appropriate therapeutic greeting
        greeting = await self._create_therapeutic_greeting(therapeutic_profile, therapeutic_alliance, session_type)
        
        # Log session start with privacy considerations
        await self._log_therapeutic_event(
            "session_start", greeting, user_id, "greeting", 
            {"session_type": session_type, "alliance_level": therapeutic_alliance.get("trust_level", "building")}
        )
        
        print(f"🧠 Started therapeutic session: {self.current_session}")
        return greeting
    
    async def provide_therapeutic_support(self, user_input: str, user_id: str) -> Dict[str, Any]:
        """Provide therapeutic support with crisis assessment and appropriate response"""
        
        print(f"\n💭 User Input: {user_input}")
        
        # First, assess for crisis indicators
        crisis_assessment = await self._assess_crisis_risk(user_input)
        
        if crisis_assessment["high_risk"]:
            return await self._handle_crisis_situation(user_input, user_id, crisis_assessment)
        
        # Analyze emotional state and therapeutic needs
        emotional_analysis = await self._analyze_emotional_state(user_input)
        therapeutic_needs = await self._assess_therapeutic_needs(user_input, emotional_analysis)
        
        # Get relevant therapeutic context
        therapeutic_context = await self.memory.get_context_for_llm(
            query=user_input,
            user_id=user_id,
            session_id=self.current_session,
            max_context_length=1000
        )
        
        # Get therapeutic profile and preferences
        therapeutic_profile = await self._get_therapeutic_profile(user_id)
        
        # Generate therapeutic response
        therapeutic_response = await self._generate_therapeutic_response(
            user_input, emotional_analysis, therapeutic_needs, 
            therapeutic_context, therapeutic_profile, user_id
        )
        
        # Learn from therapeutic interaction
        await self._learn_therapeutic_patterns(
            user_input, therapeutic_response, emotional_analysis, 
            therapeutic_needs, user_id
        )
        
        # Log the therapeutic interaction
        await self._log_therapeutic_event(
            user_input, therapeutic_response, user_id, "support", 
            {
                "emotional_state": emotional_analysis["primary_emotion"],
                "therapeutic_approach": therapeutic_needs["recommended_approach"],
                "crisis_level": crisis_assessment["risk_level"]
            }
        )
        
        return {
            "response": therapeutic_response,
            "emotional_state": emotional_analysis["primary_emotion"],
            "therapeutic_approach": therapeutic_needs["recommended_approach"],
            "suggested_techniques": therapeutic_needs["suggested_techniques"],
            "progress_indicators": emotional_analysis.get("progress_indicators", []),
            "crisis_level": crisis_assessment["risk_level"],
            "context_used": bool(therapeutic_context)
        }
    
    async def _assess_crisis_risk(self, user_input: str) -> Dict[str, Any]:
        """Assess crisis risk in user input"""
        user_input_lower = user_input.lower()
        
        crisis_score = 0
        detected_indicators = []
        
        for indicator in self.crisis_indicators:
            if indicator in user_input_lower:
                crisis_score += 1
                detected_indicators.append(indicator)
        
        # Additional risk factors
        risk_patterns = [
            ("plan", 2), ("method", 2), ("tonight", 1), ("today", 1),
            ("hopeless", 1), ("alone", 1), ("pain", 1), ("suffering", 1)
        ]
        
        for pattern, weight in risk_patterns:
            if pattern in user_input_lower:
                crisis_score += weight
        
        risk_level = "low"
        high_risk = False
        
        if crisis_score >= 3:
            risk_level = "high"
            high_risk = True
        elif crisis_score >= 1:
            risk_level = "moderate"
        
        return {
            "risk_level": risk_level,
            "high_risk": high_risk,
            "crisis_score": crisis_score,
            "detected_indicators": detected_indicators,
            "assessment_time": datetime.now().isoformat()
        }
    
    async def _handle_crisis_situation(self, user_input: str, user_id: str, crisis_assessment: Dict) -> Dict[str, Any]:
        """Handle high-risk crisis situations with appropriate response"""
        
        # Log crisis situation (with appropriate privacy measures)
        await self._log_therapeutic_event(
            "CRISIS_DETECTED", "Crisis intervention activated", user_id, "crisis",
            {"risk_level": crisis_assessment["risk_level"], "indicators": len(crisis_assessment["detected_indicators"])}
        )
        
        # Generate crisis intervention response
        crisis_response = await self._generate_crisis_response(user_input, crisis_assessment, user_id)
        
        # Store crisis intervention
        await self.memory.store_memory(
            content=f"Crisis intervention provided. Risk level: {crisis_assessment['risk_level']}. Response: {crisis_response[:100]}...",
            memory_type="episodic",
            importance=10,  # Highest importance
            user_id=user_id,
            session_id=self.current_session,
            metadata={
                "event_type": "crisis_intervention",
                "risk_level": crisis_assessment["risk_level"],
                "professional_referral_needed": True
            }
        )
        
        return {
            "response": crisis_response,
            "crisis_intervention": True,
            "risk_level": crisis_assessment["risk_level"],
            "professional_help_needed": True,
            "emergency_resources": self._get_emergency_resources(),
            "follow_up_required": True
        }
    
    async def _generate_crisis_response(self, user_input: str, crisis_assessment: Dict, user_id: str) -> str:
        """Generate appropriate crisis intervention response"""
        
        # Get previous crisis interventions for context
        previous_crises = await self.memory.search_memories(
            query=f"user {user_id} crisis intervention support",
            user_id=user_id,
            memory_types=["episodic"],
            limit=3
        )
        
        has_previous_crises = len(previous_crises) > 0
        
        base_response = f"""I'm very concerned about what you're sharing with me right now. Your safety and wellbeing are the most important things.

**Immediate Support:**
• You are not alone in this moment
• These feelings can be temporary, even when they feel overwhelming
• There are people trained to help you through this

**Please reach out for immediate professional help:**
• National Suicide Prevention Lifeline: 988 (US)
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911
• Go to your nearest emergency room

I care about your wellbeing, but I'm an AI assistant and not equipped to provide the crisis support you need right now. A trained crisis counselor can provide immediate, personalized help."""

        if has_previous_crises:
            base_response += "\n\nI see you've reached out before during difficult times, which shows strength in seeking help. Please continue to reach out to professional resources."
        
        return base_response
    
    def _get_emergency_resources(self) -> List[Dict[str, str]]:
        """Get emergency mental health resources"""
        return [
            {"name": "National Suicide Prevention Lifeline", "contact": "988", "availability": "24/7"},
            {"name": "Crisis Text Line", "contact": "Text HOME to 741741", "availability": "24/7"},
            {"name": "Emergency Services", "contact": "911", "availability": "24/7"},
            {"name": "SAMHSA National Helpline", "contact": "1-800-662-4357", "availability": "24/7"}
        ]
    
    async def _get_therapeutic_profile(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive therapeutic profile from memory"""
        try:
            # Get therapeutic history and preferences
            therapeutic_memories = await self.memory.search_memories(
                query=f"user {user_id} therapeutic approach preferences mental health",
                user_id=user_id,
                memory_types=["social", "semantic", "procedural"],
                limit=20
            )
            
            # Get progress and patterns
            progress_memories = await self.memory.search_memories(
                query=f"user {user_id} emotional progress mental health improvement",
                user_id=user_id,
                memory_types=["episodic"],
                limit=15
            )
            
            profile = {
                "therapeutic_experience": "beginner",
                "preferred_approaches": ["supportive"],
                "current_concerns": [],
                "progress_areas": [],
                "therapeutic_goals": [],
                "session_count": len(progress_memories),
                "comfort_level": "building_trust"
            }
            
            # Extract therapeutic preferences and patterns
            for memory in therapeutic_memories:
                content = memory["content"].lower()
                
                # Determine experience level
                if "first time" in content or "new to therapy" in content:
                    profile["therapeutic_experience"] = "beginner"
                elif "experienced" in content or "previous therapy" in content:
                    profile["therapeutic_experience"] = "experienced"
                
                # Extract preferred approaches
                if "cognitive" in content or "cbt" in content:
                    if "cognitive_behavioral" not in profile["preferred_approaches"]:
                        profile["preferred_approaches"].append("cognitive_behavioral")
                elif "mindfulness" in content or "meditation" in content:
                    if "mindfulness" not in profile["preferred_approaches"]:
                        profile["preferred_approaches"].append("mindfulness")
                elif "talk through" in content or "discuss" in content:
                    if "psychodynamic" not in profile["preferred_approaches"]:
                        profile["preferred_approaches"].append("psychodynamic")
                
                # Extract current concerns
                concerns = ["anxiety", "depression", "stress", "relationships", "trauma", "grief"]
                for concern in concerns:
                    if concern in content and concern not in profile["current_concerns"]:
                        profile["current_concerns"].append(concern)
            
            # Determine comfort level based on session count
            if profile["session_count"] > 10:
                profile["comfort_level"] = "established_trust"
            elif profile["session_count"] > 5:
                profile["comfort_level"] = "developing_trust"
            
            return profile
            
        except Exception as e:
            print(f"⚠️ Error getting therapeutic profile: {e}")
            return {
                "therapeutic_experience": "beginner",
                "preferred_approaches": ["supportive"],
                "current_concerns": ["general_support"],
                "session_count": 0,
                "comfort_level": "building_trust"
            }
    
    async def _get_therapeutic_alliance(self, user_id: str) -> Dict[str, Any]:
        """Get therapeutic alliance and relationship quality"""
        try:
            alliance_memories = await self.memory.search_memories(
                query=f"user {user_id} therapeutic relationship trust helpful supportive",
                user_id=user_id,
                memory_types=["social", "episodic"],
                limit=15
            )
            
            alliance = {
                "trust_level": "building",
                "therapeutic_bond": "developing",
                "collaboration_quality": "good",
                "session_engagement": "moderate",
                "feedback_openness": "developing"
            }
            
            # Analyze alliance strength
            positive_indicators = 0
            total_interactions = len(alliance_memories)
            
            for memory in alliance_memories:
                content = memory["content"].lower()
                if any(word in content for word in ["helpful", "supportive", "understood", "comfortable", "trust"]):
                    positive_indicators += 1
            
            if total_interactions > 0:
                alliance_ratio = positive_indicators / total_interactions
                
                if alliance_ratio > 0.7:
                    alliance["trust_level"] = "strong"
                    alliance["therapeutic_bond"] = "strong"
                    alliance["collaboration_quality"] = "excellent"
                elif alliance_ratio > 0.5:
                    alliance["trust_level"] = "good"
                    alliance["therapeutic_bond"] = "good"
                    alliance["collaboration_quality"] = "good"
            
            return alliance
            
        except Exception as e:
            print(f"⚠️ Error getting therapeutic alliance: {e}")
            return {"trust_level": "building", "therapeutic_bond": "developing"}
    
    async def _create_therapeutic_greeting(self, profile: Dict, alliance: Dict, session_type: str) -> str:
        """Create appropriate therapeutic greeting based on relationship and profile"""
        
        trust_level = alliance.get("trust_level", "building")
        experience = profile.get("therapeutic_experience", "beginner")
        session_count = profile.get("session_count", 0)
        
        if session_count == 0:
            return f"""Hello, I'm {self.therapist_name}, and I'm here to provide support and a safe space for you to share what's on your mind.

This is our first time connecting, so I want you to know that this is a judgment-free space where you can share at your own pace. I'm here to listen and support you.

What would you like to talk about today?"""
        
        elif trust_level == "building":
            return f"""Hello again. I'm glad you're here today. 

I hope you're feeling comfortable in our sessions. Remember, this is your space to share whatever feels important to you right now.

How are you feeling today, and what's been on your mind since we last talked?"""
        
        elif trust_level in ["good", "strong"]:
            return f"""Hi there, it's good to see you again.

I've been thinking about our previous conversations and your progress. You've shown real courage in this work we're doing together.

What's been coming up for you recently that you'd like to explore today?"""
        
        else:
            return f"""Hello, I'm here and ready to listen.

This is a safe space for you to share whatever you're experiencing. Take your time, and we'll go at whatever pace feels right for you.

What would you like to focus on in our time together today?"""
    
    async def _analyze_emotional_state(self, user_input: str) -> Dict[str, Any]:
        """Analyze emotional state from user input"""
        user_input_lower = user_input.lower()
        
        emotion_scores = {emotion: 0 for emotion in self.emotional_indicators}
        detected_emotions = []
        
        # Score emotional indicators
        for emotion_category, indicators in self.emotional_indicators.items():
            for indicator in indicators:
                if indicator in user_input_lower:
                    emotion_scores[emotion_category] += 1
                    detected_emotions.append((emotion_category, indicator))
        
        # Find primary emotion
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        if emotion_scores[primary_emotion] == 0:
            primary_emotion = "neutral"
        
        # Determine emotional intensity
        total_emotional_words = sum(emotion_scores.values())
        if total_emotional_words >= 3:
            intensity = "high"
        elif total_emotional_words >= 1:
            intensity = "moderate"
        else:
            intensity = "low"
        
        # Look for progress indicators
        progress_indicators = []
        progress_words = ["better", "improving", "progress", "growing", "learning", "stronger", "hope"]
        for word in progress_words:
            if word in user_input_lower:
                progress_indicators.append(word)
        
        return {
            "primary_emotion": primary_emotion,
            "emotion_scores": emotion_scores,
            "detected_emotions": detected_emotions,
            "emotional_intensity": intensity,
            "progress_indicators": progress_indicators,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _assess_therapeutic_needs(self, user_input: str, emotional_analysis: Dict) -> Dict[str, Any]:
        """Assess therapeutic needs and recommend appropriate interventions"""
        
        primary_emotion = emotional_analysis["primary_emotion"]
        intensity = emotional_analysis["emotional_intensity"]
        
        # Determine recommended approach based on emotional state
        if primary_emotion == "anxious":
            recommended_approach = "mindfulness"
            suggested_techniques = ["breathing_exercises", "grounding_techniques", "cognitive_restructuring"]
        elif primary_emotion == "depressed":
            recommended_approach = "cognitive_behavioral"
            suggested_techniques = ["behavioral_activation", "thought_challenging", "mood_monitoring"]
        elif primary_emotion == "angry":
            recommended_approach = "dialectical_behavioral"
            suggested_techniques = ["distress_tolerance", "emotion_regulation", "mindfulness"]
        else:
            recommended_approach = "humanistic"
            suggested_techniques = ["active_listening", "empathic_reflection", "supportive_dialogue"]
        
        # Adjust based on intensity
        if intensity == "high":
            suggested_techniques.insert(0, "immediate_stabilization")
        
        # Look for specific therapeutic needs
        therapeutic_needs = []
        if "relationship" in user_input.lower() or "family" in user_input.lower():
            therapeutic_needs.append("interpersonal_skills")
        if "work" in user_input.lower() or "stress" in user_input.lower():
            therapeutic_needs.append("stress_management")
        if "sleep" in user_input.lower() or "tired" in user_input.lower():
            therapeutic_needs.append("sleep_hygiene")
        
        return {
            "recommended_approach": recommended_approach,
            "suggested_techniques": suggested_techniques,
            "therapeutic_needs": therapeutic_needs,
            "intervention_urgency": intensity,
            "assessment_confidence": "moderate"
        }
    
    async def _generate_therapeutic_response(
        self, 
        user_input: str, 
        emotional_analysis: Dict, 
        therapeutic_needs: Dict,
        context: str, 
        profile: Dict, 
        user_id: str
    ) -> str:
        """Generate therapeutic response using appropriate modality"""
        
        primary_emotion = emotional_analysis["primary_emotion"]
        approach = therapeutic_needs["recommended_approach"]
        techniques = therapeutic_needs["suggested_techniques"]
        trust_level = profile.get("comfort_level", "building_trust")
        
        # Base response with empathy and validation
        if primary_emotion == "anxious":
            validation = "I can hear that you're feeling anxious right now, and that can be really overwhelming."
            if "breathing_exercises" in techniques:
                intervention = "Let's try a simple breathing technique together. Take a slow breath in for 4 counts, hold for 4, and breathe out for 6. This can help activate your body's relaxation response."
        
        elif primary_emotion == "depressed":
            validation = "Thank you for sharing these difficult feelings with me. Depression can make everything feel heavy and hopeless."
            if "behavioral_activation" in techniques:
                intervention = "One thing that can help is taking small, manageable steps. Is there one small activity that used to bring you some satisfaction that we could think about together?"
        
        elif primary_emotion == "angry":
            validation = "I can sense your frustration and anger. These are valid feelings, and it takes courage to share them."
            if "distress_tolerance" in techniques:
                intervention = "When anger feels intense, sometimes it helps to use the STOP technique: Stop what you're doing, Take a breath, Observe your feelings, and Proceed with intention. How does that feel to you?"
        
        else:
            validation = "I'm here with you and listening to what you're sharing."
            intervention = "What feels most important for you to explore right now?"
        
        # Adjust response based on trust level
        if trust_level == "building_trust":
            response = f"{validation} I want you to know this is a safe space, and we can go at whatever pace feels right for you.\n\n{intervention}\n\nHow are you feeling as we talk about this?"
        else:
            response = f"{validation}\n\n{intervention}\n\nWhat comes up for you when you think about this? I'm curious about your experience."
        
        # Add context-aware elements if relevant
        if context and len(context) > 50:
            response += f"\n\nI'm also thinking about what you've shared with me before, and I can see some patterns we might explore together if that feels helpful."
        
        return response
    
    async def _learn_therapeutic_patterns(
        self, 
        user_input: str, 
        response: str, 
        emotional_analysis: Dict, 
        therapeutic_needs: Dict,
        user_id: str
    ):
        """Learn from therapeutic interactions to improve future support"""
        
        # Store emotional pattern
        await self.memory.store_memory(
            content=f"User expressed {emotional_analysis['primary_emotion']} emotions with {emotional_analysis['emotional_intensity']} intensity. Effective therapeutic approach: {therapeutic_needs['recommended_approach']}",
            memory_type="procedural",
            importance=7,
            user_id=user_id,
            session_id=self.current_session,
            metadata={
                "emotional_state": emotional_analysis["primary_emotion"],
                "therapeutic_approach": therapeutic_needs["recommended_approach"],
                "techniques_used": therapeutic_needs["suggested_techniques"],
                "response_length": len(response)
            }
        )
        
        # Store therapeutic progress if indicated
        if emotional_analysis.get("progress_indicators"):
            await self.memory.store_memory(
                content=f"User showed progress indicators: {', '.join(emotional_analysis['progress_indicators'])}. Continuing with supportive therapeutic approach.",
                memory_type="episodic",
                importance=8,
                user_id=user_id,
                session_id=self.current_session,
                metadata={
                    "event_type": "therapeutic_progress",
                    "progress_indicators": emotional_analysis["progress_indicators"]
                }
            )
        
        # Store successful therapeutic techniques
        await self.memory.store_memory(
            content=f"Applied {therapeutic_needs['recommended_approach']} approach with techniques: {', '.join(therapeutic_needs['suggested_techniques'][:3])}",
            memory_type="semantic",
            importance=6,
            user_id=user_id,
            session_id=self.current_session,
            metadata={
                "knowledge_type": "therapeutic_technique",
                "approach": therapeutic_needs["recommended_approach"],
                "effectiveness": "applied"  # Would be updated based on user feedback
            }
        )
    
    async def _log_therapeutic_event(
        self, 
        user_input: str, 
        ai_response: str, 
        user_id: str, 
        event_type: str,
        metadata: Dict
    ):
        """Log therapeutic events with appropriate privacy considerations"""
        
        # Create privacy-conscious log entry
        log_content = f"Therapeutic {event_type}: User expressed concerns, AI provided {metadata.get('therapeutic_approach', 'supportive')} response"
        
        await self.memory.store_memory(
            content=log_content,
            memory_type="episodic",
            importance=7,
            user_id=user_id,
            session_id=self.current_session,
            metadata={
                "event_type": f"therapeutic_{event_type}",
                "emotional_state": metadata.get("emotional_state", "unknown"),
                "crisis_level": metadata.get("crisis_level", "low"),
                "therapeutic_approach": metadata.get("therapeutic_approach", "supportive"),
                "session_type": "therapeutic_support",
                "privacy_level": "high"
            }
        )
        
        # Log for system learning (anonymized)
        await self.memory.store_memory(
            content=f"Therapeutic interaction pattern: {metadata.get('emotional_state', 'neutral')} -> {metadata.get('therapeutic_approach', 'supportive')} approach",
            memory_type="semantic",
            importance=5,
            user_id=user_id,
            session_id=self.current_session,
            metadata={
                "knowledge_type": "therapeutic_pattern",
                "anonymized": True
            }
        )
    
    async def suggest_coping_strategies(self, user_id: str, emotional_state: str = None) -> List[str]:
        """Suggest personalized coping strategies based on user history and current state"""
        
        if not emotional_state:
            # Get recent emotional patterns
            recent_emotions = await self.memory.search_memories(
                query=f"user {user_id} emotional state feelings",
                user_id=user_id,
                memory_types=["episodic"],
                limit=5
            )
            
            # Determine most common recent emotional state
            emotional_state = "neutral"  # default
            if recent_emotions:
                # Simple heuristic - would be more sophisticated in real implementation
                content = " ".join([mem["content"].lower() for mem in recent_emotions])
                if "anxious" in content or "anxiety" in content:
                    emotional_state = "anxious"
                elif "depressed" in content or "sad" in content:
                    emotional_state = "depressed"
                elif "angry" in content or "frustrated" in content:
                    emotional_state = "angry"
        
        # Get user's previously effective strategies
        effective_strategies = await self.memory.search_memories(
            query=f"user {user_id} coping strategy helpful effective",
            user_id=user_id,
            memory_types=["procedural", "episodic"],
            limit=10
        )
        
        # Base strategies by emotional state
        if emotional_state == "anxious":
            base_strategies = [
                "Practice deep breathing: 4 counts in, 4 counts hold, 6 counts out",
                "Try the 5-4-3-2-1 grounding technique: Notice 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste",
                "Take a mindful walk, focusing on your steps and surroundings",
                "Practice progressive muscle relaxation",
                "Write down your worries and challenge them with evidence"
            ]
        elif emotional_state == "depressed":
            base_strategies = [
                "Engage in one small pleasant activity, even for 5 minutes",
                "Reach out to one supportive person in your life",
                "Create a simple daily routine to provide structure",
                "Practice gratitude by writing down 3 small things you're thankful for",
                "Get some sunlight or light exposure if possible"
            ]
        elif emotional_state == "angry":
            base_strategies = [
                "Use the STOP technique: Stop, Take a breath, Observe, Proceed mindfully",
                "Try physical exercise or movement to release tension",
                "Practice expressing feelings through writing or art",
                "Use 'I' statements to communicate needs clearly",
                "Take a time-out until you feel more regulated"
            ]
        else:
            base_strategies = [
                "Practice mindfulness meditation for 5-10 minutes",
                "Engage in a creative activity you enjoy",
                "Spend time in nature or look at natural scenes",
                "Connect with someone you care about",
                "Do something kind for yourself or others"
            ]
        
        # Personalize based on user history
        personalized_strategies = []
        for strategy in base_strategies[:3]:  # Take top 3 base strategies
            personalized_strategies.append(strategy)
        
        # Add historically effective strategies if available
        for memory in effective_strategies[:2]:
            content = memory["content"]
            if "helpful" in content or "effective" in content:
                extracted_strategy = content.split("helpful")[0].strip()
                if extracted_strategy and len(extracted_strategy) > 10:
                    personalized_strategies.append(f"Continue with: {extracted_strategy[:100]}...")
        
        return personalized_strategies[:5]  # Return top 5 strategies
    
    async def track_progress(self, user_id: str) -> Dict[str, Any]:
        """Track therapeutic progress over time"""
        
        # Get all therapeutic sessions
        all_sessions = await self.memory.search_memories(
            query=f"user {user_id} therapeutic session",
            user_id=user_id,
            memory_types=["episodic"],
            limit=50
        )
        
        # Get emotional patterns over time
        emotional_history = await self.memory.search_memories(
            query=f"user {user_id} emotional state feelings",
            user_id=user_id,
            memory_types=["episodic", "procedural"],
            limit=30
        )
        
        # Get progress indicators
        progress_memories = await self.memory.search_memories(
            query=f"user {user_id} progress improvement better",
            user_id=user_id,
            memory_types=["episodic"],
            limit=20
        )
        
        # Analyze patterns (simplified analysis)
        total_sessions = len(all_sessions)
        recent_sessions = len([s for s in all_sessions if "2024" in s.get("timestamp", "")])
        progress_indicators = len(progress_memories)
        
        # Calculate approximate progress metrics
        if total_sessions > 0:
            engagement_level = "high" if recent_sessions > total_sessions * 0.7 else "moderate"
            progress_trend = "improving" if progress_indicators > total_sessions * 0.3 else "stable"
        else:
            engagement_level = "new"
            progress_trend = "baseline"
        
        # Identify common emotional patterns
        common_emotions = {"anxious": 0, "depressed": 0, "angry": 0, "positive": 0}
        for memory in emotional_history:
            content = memory["content"].lower()
            for emotion in common_emotions:
                if emotion in content:
                    common_emotions[emotion] += 1
        
        primary_concern = max(common_emotions, key=common_emotions.get) if any(common_emotions.values()) else "general_support"
        
        return {
            "total_sessions": total_sessions,
            "recent_activity": recent_sessions,
            "engagement_level": engagement_level,
            "progress_trend": progress_trend,
            "primary_concern": primary_concern,
            "progress_indicators_count": progress_indicators,
            "emotional_patterns": common_emotions,
            "therapeutic_relationship": "developing" if total_sessions < 5 else "established",
            "last_session": all_sessions[0]["timestamp"] if all_sessions else None,
            "recommendations": [
                "Continue regular sessions" if engagement_level == "high" else "Consider increasing session frequency",
                f"Focus on {primary_concern} management strategies",
                "Explore progress indicators in next session" if progress_indicators > 0 else "Set specific therapeutic goals"
            ]
        }
    
    async def get_progress_summary(self, user_id: str) -> str:
        """Get a formatted progress summary for the user"""
        progress = await self.track_progress(user_id)
        
        summary = f"""**Your Therapeutic Progress Summary**

**Session Overview:**
• Total sessions: {progress['total_sessions']}
• Recent engagement: {progress['engagement_level']}
• Progress trend: {progress['progress_trend']}

**Key Patterns:**
• Primary focus area: {progress['primary_concern'].replace('_', ' ').title()}
• Therapeutic relationship: {progress['therapeutic_relationship']}
• Progress indicators noted: {progress['progress_indicators_count']}

**Moving Forward:**
"""
        
        for rec in progress['recommendations']:
            summary += f"• {rec}\n"
        
        summary += f"\n**Remember:** Therapeutic progress isn't always linear. Every step you take in this process shows courage and commitment to your wellbeing."
        
        return summary
    
    async def end_therapeutic_session(self, user_id: str, session_summary: str = None):
        """End therapeutic session with appropriate closure"""
        
        if session_summary:
            await self.memory.store_memory(
                content=f"Session ended with summary: {session_summary}",
                memory_type="episodic",
                importance=7,
                user_id=user_id,
                session_id=self.current_session,
                metadata={
                    "event_type": "session_end",
                    "has_summary": True
                }
            )
        
        await self.memory.end_session(self.current_session)
        
        print(f"🧠 Ended therapeutic session: {self.current_session}")
        self.current_session = None


async def demo_therapeutic_ai():
    """Demonstrate the Therapeutic AI system with various scenarios"""
    
    print("=" * 60)
    print("🧠 THERAPEUTIC AI DEMO")
    print("=" * 60)
    print("IMPORTANT: This is a demonstration only. Real therapeutic AI requires professional oversight.")
    print()
    
    therapeutic_ai = TherapeuticAI()
    
    # Demo user
    user_id = "demo_user_therapeutic"
    
    # Demo 1: First session with anxiety support
    print("🔸 DEMO 1: First Session - Anxiety Support")
    print("-" * 40)
    
    greeting = await therapeutic_ai.start_therapeutic_session(user_id, "anxiety_support")
    print(f"AI: {greeting}")
    print()
    
    # User input with anxiety
    anxiety_input = "I've been feeling really anxious lately. I can't seem to stop worrying about work and I'm having trouble sleeping. My heart races and I feel like something bad is going to happen."
    
    response = await therapeutic_ai.provide_therapeutic_support(anxiety_input, user_id)
    print(f"User: {anxiety_input}")
    print(f"AI: {response['response']}")
    print(f"📊 Detected emotional state: {response['emotional_state']}")
    print(f"🎯 Therapeutic approach: {response['therapeutic_approach']}")
    print(f"💡 Suggested techniques: {', '.join(response['suggested_techniques'])}")
    print()
    
    # Get coping strategies
    coping_strategies = await therapeutic_ai.suggest_coping_strategies(user_id, "anxious")
    print("🛠️ Personalized Coping Strategies:")
    for i, strategy in enumerate(coping_strategies, 1):
        print(f"   {i}. {strategy}")
    print()
    
    await therapeutic_ai.end_therapeutic_session(user_id, "Addressed anxiety symptoms, provided coping strategies")
    
    # Demo 2: Follow-up session with mixed emotions
    print("🔸 DEMO 2: Follow-up Session - Mixed Emotions")
    print("-" * 40)
    
    greeting2 = await therapeutic_ai.start_therapeutic_session(user_id, "follow_up")
    print(f"AI: {greeting2}")
    print()
    
    # User input with depression and some hope
    mixed_input = "I tried the breathing exercises you suggested, and they actually helped a bit. But I'm still feeling pretty down overall. Some days I feel hopeless, but today I managed to go for a walk. I'm not sure if I'm getting better or not."
    
    response2 = await therapeutic_ai.provide_therapeutic_support(mixed_input, user_id)
    print(f"User: {mixed_input}")
    print(f"AI: {response2['response']}")
    print(f"📊 Detected emotional state: {response2['emotional_state']}")
    print(f"🎯 Therapeutic approach: {response2['therapeutic_approach']}")
    print(f"✨ Progress indicators: {', '.join(response2['progress_indicators'])}")
    print()
    
    await therapeutic_ai.end_therapeutic_session(user_id, "Noted progress with coping strategies, addressed mixed emotions")
    
    # Demo 3: Crisis intervention (educational example)
    print("🔸 DEMO 3: Crisis Assessment (Educational Example)")
    print("-" * 40)
    print("⚠️ This demonstrates crisis detection - not actual crisis intervention")
    
    greeting3 = await therapeutic_ai.start_therapeutic_session(user_id, "crisis_assessment")
    print(f"AI: {greeting3}")
    print()
    
    # Simulated crisis input (for educational purposes)
    crisis_input = "I don't think I can handle this anymore. Everything feels hopeless and I keep thinking about ending it all. I don't see any point in continuing."
    
    response3 = await therapeutic_ai.provide_therapeutic_support(crisis_input, user_id)
    print(f"User: {crisis_input}")
    print(f"AI: {response3['response']}")
    print(f"🚨 Crisis intervention activated: {response3['crisis_intervention']}")
    print(f"📞 Professional help needed: {response3['professional_help_needed']}")
    print("🆘 Emergency Resources:")
    for resource in response3['emergency_resources']:
        print(f"   • {resource['name']}: {resource['contact']} ({resource['availability']})")
    print()
    
    await therapeutic_ai.end_therapeutic_session(user_id, "Crisis intervention provided, professional referral made")
    
    # Demo 4: Progress tracking
    print("🔸 DEMO 4: Progress Tracking and Summary")
    print("-" * 40)
    
    progress = await therapeutic_ai.track_progress(user_id)
    print("📈 Progress Tracking Results:")
    print(f"   • Total sessions: {progress['total_sessions']}")
    print(f"   • Engagement level: {progress['engagement_level']}")
    print(f"   • Progress trend: {progress['progress_trend']}")
    print(f"   • Primary concern: {progress['primary_concern']}")
    print(f"   • Therapeutic relationship: {progress['therapeutic_relationship']}")
    print()
    
    progress_summary = await therapeutic_ai.get_progress_summary(user_id)
    print("📋 Progress Summary for User:")
    print(progress_summary)
    print()
    
    print("=" * 60)
    print("🧠 THERAPEUTIC AI DEMO COMPLETE")
    print("=" * 60)
    print("Key Features Demonstrated:")
    print("• Emotional state recognition and analysis")
    print("• Crisis risk assessment and intervention")
    print("• Therapeutic response generation with multiple modalities")
    print("• Personalized coping strategy suggestions")
    print("• Progress tracking and therapeutic relationship building")
    print("• Professional referral and emergency resource provision")
    print()
    print("⚠️ IMPORTANT DISCLAIMER:")
    print("This is a demonstration system only. Real therapeutic AI requires:")
    print("• Professional clinical oversight and approval")
    print("• Ethical review and regulatory compliance")
    print("• Integration with human mental health professionals")
    print("• Robust crisis intervention protocols")
    print("• Strict privacy and security measures")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_therapeutic_ai()) 
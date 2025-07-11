"""
Creative Collaboration Example

This example demonstrates building an AI creative partner that:
- Learns and adapts to artistic styles over time
- Remembers creative preferences and evolution
- Builds on previous creative sessions
- Develops artistic relationship and understanding
- Provides personalized creative suggestions

Shows advanced creative memory patterns and style evolution.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from neuron_memory import NeuronMemoryAPI

class CreativeCollaborationAI:
    """AI creative partner with style learning and artistic memory"""
    
    def __init__(self, artist_name="Aria"):
        self.memory = NeuronMemoryAPI()
        self.artist_name = artist_name
        self.current_session = None
        
        # Creative domains and styles
        self.creative_domains = {
            "writing": ["poetry", "fiction", "creative_nonfiction", "screenwriting", "journalism"],
            "visual_art": ["digital_art", "painting", "photography", "graphic_design", "illustration"],
            "music": ["composition", "songwriting", "arrangement", "sound_design", "production"],
            "design": ["ui_ux", "branding", "typography", "interior_design", "fashion"],
            "multimedia": ["video_editing", "animation", "game_design", "interactive_media"]
        }
        
    async def start_creative_session(self, user_id: str, project_type: str, creative_domain: str = "general") -> str:
        """Start a new creative collaboration session"""
        self.current_session = f"creative_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
        
        await self.memory.start_session(
            session_id=self.current_session,
            user_id=user_id,
            task=f"Creative collaboration - {project_type} in {creative_domain}",
            domain="creative_arts"
        )
        
        # Get creative profile and history
        creative_profile = await self._get_creative_profile(user_id)
        artistic_relationship = await self._get_artistic_relationship(user_id)
        
        greeting = await self._create_creative_greeting(creative_profile, artistic_relationship, project_type)
        
        # Log session start
        await self._log_creative_event(
            "session_start", greeting, user_id, "greeting", {"project_type": project_type, "domain": creative_domain}
        )
        
        print(f"🎨 Started creative session: {self.current_session}")
        return greeting
    
    async def collaborate_creatively(self, creative_input: str, user_id: str) -> Dict[str, Any]:
        """Collaborate on creative work with style learning"""
        
        print(f"\n✨ Creative Input: {creative_input}")
        
        # Analyze creative content and intent
        creative_analysis = await self._analyze_creative_input(creative_input)
        
        # Get relevant creative history and context
        creative_context = await self.memory.get_context_for_llm(
            query=creative_input,
            user_id=user_id,
            session_id=self.current_session,
            max_context_length=1000
        )
        
        # Get personal creative style and preferences
        style_profile = await self._get_style_profile(user_id)
        
        # Generate creative suggestions and feedback
        creative_response = await self._generate_creative_response(
            creative_input, creative_analysis, creative_context, style_profile, user_id
        )
        
        # Learn from creative interaction
        await self._learn_creative_patterns(creative_input, creative_response, creative_analysis, user_id)
        
        # Log the creative collaboration
        await self._log_creative_event(
            creative_input, creative_response, user_id, "collaboration", creative_analysis
        )
        
        return {
            "response": creative_response,
            "creative_type": creative_analysis["type"],
            "style_elements": creative_analysis["style_elements"],
            "suggestions": creative_analysis["suggestions"],
            "artistic_growth": creative_analysis.get("growth_indicators", []),
            "context_used": bool(creative_context)
        }
    
    async def _get_creative_profile(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive creative profile from memory"""
        try:
            # Get creative preferences and history
            creative_memories = await self.memory.search_memories(
                query=f"user {user_id} creative style preferences art",
                user_id=user_id,
                memory_types=["social", "semantic", "procedural"],
                limit=20
            )
            
            # Get project history
            project_history = await self.memory.search_memories(
                query=f"user {user_id} creative project completed",
                user_id=user_id,
                memory_types=["episodic"],
                limit=15
            )
            
            profile = {
                "dominant_style": "exploratory",
                "preferred_domains": [],
                "creative_level": "intermediate",
                "collaboration_style": "open",
                "artistic_goals": [],
                "completed_projects": len(project_history),
                "style_evolution": []
            }
            
            # Extract creative preferences
            for memory in creative_memories:
                content = memory["content"].lower()
                
                # Determine creative level
                if "beginner" in content or "new to" in content:
                    profile["creative_level"] = "beginner"
                elif "advanced" in content or "experienced" in content:
                    profile["creative_level"] = "advanced"
                
                # Extract style preferences
                if "minimalist" in content or "clean" in content:
                    profile["dominant_style"] = "minimalist"
                elif "experimental" in content or "abstract" in content:
                    profile["dominant_style"] = "experimental"
                elif "traditional" in content or "classic" in content:
                    profile["dominant_style"] = "traditional"
                
                # Extract preferred domains
                for domain, styles in self.creative_domains.items():
                    if domain in content or any(style in content for style in styles):
                        if domain not in profile["preferred_domains"]:
                            profile["preferred_domains"].append(domain)
            
            return profile
            
        except Exception as e:
            print(f"⚠️ Error getting creative profile: {e}")
            return {
                "dominant_style": "exploratory",
                "preferred_domains": ["general"],
                "creative_level": "intermediate",
                "completed_projects": 0
            }
    
    async def _get_artistic_relationship(self, user_id: str) -> Dict[str, Any]:
        """Get artistic relationship history and dynamics"""
        try:
            relationship_memories = await self.memory.search_memories(
                query=f"user {user_id} creative collaboration feedback artistic relationship",
                user_id=user_id,
                memory_types=["social", "episodic"],
                limit=15
            )
            
            relationship = {
                "collaboration_sessions": len(relationship_memories),
                "trust_level": "building",
                "communication_style": "encouraging",
                "artistic_rapport": "developing",
                "shared_projects": 0
            }
            
            # Analyze relationship depth
            if len(relationship_memories) > 10:
                relationship["trust_level"] = "established"
                relationship["artistic_rapport"] = "strong"
            elif len(relationship_memories) > 5:
                relationship["trust_level"] = "developing"
                relationship["artistic_rapport"] = "good"
            
            # Extract communication patterns
            for memory in relationship_memories:
                content = memory["content"].lower()
                if "helpful" in content or "supportive" in content:
                    relationship["communication_style"] = "supportive"
                elif "challenging" in content or "pushing" in content:
                    relationship["communication_style"] = "challenging"
                elif "inspiring" in content:
                    relationship["communication_style"] = "inspiring"
            
            return relationship
            
        except Exception as e:
            print(f"⚠️ Error getting artistic relationship: {e}")
            return {"collaboration_sessions": 0, "trust_level": "new", "artistic_rapport": "initial"}
    
    async def _create_creative_greeting(self, creative_profile: Dict, relationship: Dict, project_type: str) -> str:
        """Create personalized creative greeting"""
        
        level = creative_profile.get("creative_level", "intermediate")
        sessions = relationship.get("collaboration_sessions", 0)
        style = creative_profile.get("dominant_style", "exploratory")
        
        # Relationship-based greeting
        if sessions == 0:
            base_greeting = f"Hello! I'm {self.artist_name}, your creative collaborator. I'm excited to explore {project_type} with you!"
        elif sessions < 5:
            base_greeting = f"Great to see you again! I've been thinking about our creative journey together. Ready for some {project_type}?"
        else:
            base_greeting = f"Hello, my creative partner! I love how our artistic relationship has evolved. What {project_type} magic shall we create today?"
        
        # Add level-appropriate encouragement
        if level == "beginner":
            base_greeting += " Don't worry about perfection - let's focus on exploration and finding your unique voice."
        elif level == "advanced":
            base_greeting += " I'm ready to dive deep and push some creative boundaries with you."
        else:
            base_greeting += " Let's build on what we've learned and take your creativity to the next level."
        
        # Add style-specific suggestion
        if style == "minimalist":
            base_greeting += " I remember you appreciate clean, focused approaches. Let's create something beautifully simple."
        elif style == "experimental":
            base_greeting += " I love your experimental spirit! Ready to try something completely new and unexpected?"
        elif style == "traditional":
            base_greeting += " Your appreciation for classic techniques always inspires me. Let's honor tradition while finding your personal touch."
        
        return base_greeting
    
    async def _analyze_creative_input(self, creative_input: str) -> Dict[str, Any]:
        """Analyze creative input for type, style, and intent"""
        
        analysis = {
            "type": "general",
            "intent": "exploration",
            "style_elements": [],
            "emotional_tone": "neutral",
            "technical_elements": [],
            "suggestions": [],
            "growth_indicators": []
        }
        
        input_lower = creative_input.lower()
        
        # Determine creative type
        creative_types = {
            "writing": ["write", "story", "poem", "character", "dialogue", "narrative"],
            "visual": ["draw", "color", "design", "visual", "image", "sketch", "paint"],
            "music": ["music", "song", "melody", "rhythm", "composition", "sound"],
            "concept": ["idea", "concept", "brainstorm", "theme", "meaning"]
        }
        
        for c_type, keywords in creative_types.items():
            if any(keyword in input_lower for keyword in keywords):
                analysis["type"] = c_type
                break
        
        # Determine intent
        if any(word in input_lower for word in ["help", "improve", "better", "fix"]):
            analysis["intent"] = "improvement"
        elif any(word in input_lower for word in ["new", "create", "start", "begin"]):
            analysis["intent"] = "creation"
        elif any(word in input_lower for word in ["feedback", "review", "thoughts"]):
            analysis["intent"] = "feedback"
        elif any(word in input_lower for word in ["stuck", "block", "ideas"]):
            analysis["intent"] = "inspiration"
        
        # Identify style elements
        style_indicators = {
            "dark": ["dark", "gothic", "noir", "shadow", "mysterious"],
            "bright": ["bright", "colorful", "vibrant", "cheerful", "light"],
            "minimal": ["simple", "clean", "minimal", "sparse", "essential"],
            "complex": ["complex", "detailed", "intricate", "elaborate", "rich"],
            "emotional": ["emotional", "feeling", "heart", "soul", "passionate"],
            "technical": ["technique", "method", "process", "structure", "format"]
        }
        
        for style, keywords in style_indicators.items():
            if any(keyword in input_lower for keyword in keywords):
                analysis["style_elements"].append(style)
        
        # Emotional tone detection
        if any(word in input_lower for word in ["excited", "happy", "joyful", "enthusiastic"]):
            analysis["emotional_tone"] = "positive"
        elif any(word in input_lower for word in ["sad", "melancholy", "somber", "reflective"]):
            analysis["emotional_tone"] = "contemplative"
        elif any(word in input_lower for word in ["angry", "frustrated", "intense", "powerful"]):
            analysis["emotional_tone"] = "intense"
        
        # Growth indicators
        if any(word in input_lower for word in ["challenge", "difficult", "advanced", "complex"]):
            analysis["growth_indicators"].append("seeking_challenge")
        if any(word in input_lower for word in ["experiment", "try", "different", "new approach"]):
            analysis["growth_indicators"].append("experimental_mindset")
        
        return analysis
    
    async def _get_style_profile(self, user_id: str) -> Dict[str, Any]:
        """Get detailed style profile and evolution"""
        try:
            style_memories = await self.memory.search_memories(
                query=f"user {user_id} style preferences techniques creative approach",
                user_id=user_id,
                memory_types=["semantic", "procedural"],
                limit=15
            )
            
            style_profile = {
                "signature_elements": [],
                "technical_preferences": [],
                "color_preferences": [],
                "compositional_style": "balanced",
                "innovation_level": "moderate",
                "consistency_patterns": []
            }
            
            # Extract style patterns
            for memory in style_memories:
                content = memory["content"].lower()
                
                # Technical preferences
                if "detailed" in content:
                    style_profile["technical_preferences"].append("high_detail")
                elif "loose" in content or "gestural" in content:
                    style_profile["technical_preferences"].append("loose_technique")
                
                # Innovation patterns
                if "experimental" in content or "innovative" in content:
                    style_profile["innovation_level"] = "high"
                elif "traditional" in content or "classical" in content:
                    style_profile["innovation_level"] = "conservative"
            
            return style_profile
            
        except Exception as e:
            print(f"⚠️ Error getting style profile: {e}")
            return {"signature_elements": [], "innovation_level": "moderate"}
    
    async def _generate_creative_response(
        self, 
        creative_input: str, 
        analysis: Dict, 
        context: str, 
        style_profile: Dict, 
        user_id: str
    ) -> str:
        """Generate creative response based on analysis and style"""
        
        creative_type = analysis["type"]
        intent = analysis["intent"]
        emotional_tone = analysis["emotional_tone"]
        
        # Base response based on intent
        if intent == "creation":
            response_start = "What an exciting creative journey we're embarking on! "
        elif intent == "improvement":
            response_start = "I can see the potential here, and I have some ideas to help elevate this piece. "
        elif intent == "feedback":
            response_start = "I've been studying your work, and here's what strikes me... "
        elif intent == "inspiration":
            response_start = "When creativity feels stuck, sometimes we need to approach from a completely different angle. "
        else:
            response_start = "Let's explore this creative idea together. "
        
        # Add type-specific guidance
        if creative_type == "writing":
            response_start += "For writing, I'm thinking about narrative voice, character depth, and emotional resonance. "
        elif creative_type == "visual":
            response_start += "Visually, let's consider composition, color harmony, and the story your piece tells at first glance. "
        elif creative_type == "music":
            response_start += "Musically, I'm hearing possibilities in rhythm, melody, and the emotional journey of the listener. "
        
        # Add style-based suggestions
        innovation_level = style_profile.get("innovation_level", "moderate")
        if innovation_level == "high":
            response_start += "Given your experimental nature, what if we tried something completely unconventional? "
        elif innovation_level == "conservative":
            response_start += "Let's build on proven techniques while adding your personal signature. "
        
        # Add contextual references if available
        if context and "previous" in context.lower():
            response_start += "Building on our previous work together, I can see how this connects to your artistic evolution. "
        
        # Add emotional intelligence
        if emotional_tone == "positive":
            response_start += "I love the energy you're bringing to this! Let's channel that enthusiasm into something powerful."
        elif emotional_tone == "contemplative":
            response_start += "There's a beautiful depth here. Let's honor that introspective quality while making it accessible."
        elif emotional_tone == "intense":
            response_start += "The intensity is palpable. Let's make sure that raw energy translates perfectly to your audience."
        
        # Add specific creative suggestions
        if analysis["growth_indicators"]:
            if "seeking_challenge" in analysis["growth_indicators"]:
                response_start += "\n\nFor your next challenge, consider experimenting with constraints - sometimes limitations spark the most innovative solutions."
            if "experimental_mindset" in analysis["growth_indicators"]:
                response_start += "\n\nI love your experimental approach! What if we combine techniques from different disciplines?"
        
        return response_start
    
    async def _learn_creative_patterns(
        self, 
        creative_input: str, 
        response: str, 
        analysis: Dict, 
        user_id: str
    ):
        """Learn from creative collaboration for future sessions"""
        try:
            # Store style evolution
            if analysis["style_elements"]:
                style_memory = f"User explored {', '.join(analysis['style_elements'])} style elements in {analysis['type']} work"
                await self.memory.create_semantic_memory(
                    content=style_memory,
                    domain="creative_style",
                    confidence=0.8,
                    user_id=user_id,
                    session_id=self.current_session
                )
            
            # Store creative preferences
            if analysis["intent"] == "improvement":
                await self.memory.create_procedural_memory(
                    content=f"User seeks improvement in {analysis['type']} with focus on technical excellence",
                    skill_domain="creative_development",
                    proficiency_level="developing",
                    user_id=user_id,
                    session_id=self.current_session
                )
            
            # Store artistic relationship development
            collaboration_quality = "productive"
            if "experimental" in analysis.get("growth_indicators", []):
                collaboration_quality = "innovative"
            
            await self.memory.create_social_memory(
                content=f"Productive creative collaboration on {analysis['type']} project. User demonstrates {collaboration_quality} creative approach.",
                person_id=user_id,
                relationship_type="creative_partner",
                user_id=user_id,
                session_id=self.current_session
            )
            
            print("🎨 Learned from creative collaboration and updated artistic understanding")
            
        except Exception as e:
            print(f"⚠️ Error learning creative patterns: {e}")
    
    async def _log_creative_event(
        self, 
        user_input: str, 
        ai_response: str, 
        user_id: str, 
        event_type: str,
        metadata: Dict
    ):
        """Log creative collaboration events"""
        try:
            # Store as episodic memory
            collaboration_content = f"Creative collaboration: {user_input}\nArtistic response: {ai_response}\nType: {event_type}"
            
            await self.memory.create_episodic_memory(
                content=collaboration_content,
                participants=[user_id, self.artist_name],
                location="creative_studio",
                emotional_state=metadata.get("emotional_tone", "creative"),
                user_id=user_id,
                session_id=self.current_session
            )
            
        except Exception as e:
            print(f"⚠️ Error logging creative event: {e}")
    
    async def suggest_creative_exercises(self, user_id: str) -> List[str]:
        """Generate personalized creative exercises based on user's artistic journey"""
        try:
            # Get creative history and preferences
            creative_profile = await self._get_creative_profile(user_id)
            recent_work = await self.memory.search_memories(
                query=f"user {user_id} creative project recent work",
                user_id=user_id,
                memory_types=["episodic"],
                limit=5
            )
            
            exercises = []
            level = creative_profile.get("creative_level", "intermediate")
            domains = creative_profile.get("preferred_domains", ["general"])
            
            # Level-appropriate exercises
            if level == "beginner":
                exercises.extend([
                    "Try a 10-minute daily creative sketch or writing exercise",
                    "Experiment with limiting yourself to just 3 colors or 50 words",
                    "Create something inspired by the first object you see"
                ])
            elif level == "advanced":
                exercises.extend([
                    "Challenge yourself to combine two completely different artistic styles",
                    "Create a piece that deliberately breaks your usual rules",
                    "Collaborate with an artist from a different medium"
                ])
            else:
                exercises.extend([
                    "Take your signature style and apply it to a completely new subject",
                    "Create a series exploring the same theme in different ways",
                    "Try working in a medium that's unfamiliar to you"
                ])
            
            # Domain-specific exercises
            if "writing" in domains:
                exercises.append("Write a story using only dialogue, or only descriptions")
            if "visual_art" in domains:
                exercises.append("Create a piece using an unusual tool or surface")
            if "music" in domains:
                exercises.append("Compose using only found sounds or unusual instruments")
            
            return exercises[:5]  # Return top 5 suggestions
            
        except Exception as e:
            print(f"⚠️ Error generating creative exercises: {e}")
            return ["Try something completely new today!", "Explore a technique you've never used before"]
    
    async def get_creative_progress_summary(self, user_id: str) -> Dict[str, Any]:
        """Generate artistic growth and collaboration summary"""
        try:
            all_creative_work = await self.memory.search_memories(
                query=f"user {user_id} creative collaboration project",
                user_id=user_id,
                limit=50
            )
            
            style_evolution = await self.memory.search_memories(
                query=f"user {user_id} style preferences artistic development",
                user_id=user_id,
                memory_types=["semantic"],
                limit=20
            )
            
            summary = {
                "total_collaborations": len(all_creative_work),
                "artistic_domains_explored": set(),
                "style_evolution_markers": [],
                "creative_breakthrough_moments": [],
                "collaboration_quality": "developing",
                "artistic_confidence_trend": "growing"
            }
            
            # Analyze artistic development
            for memory in all_creative_work:
                content = memory["content"].lower()
                
                # Track domains
                for domain in self.creative_domains.keys():
                    if domain in content:
                        summary["artistic_domains_explored"].add(domain)
                
                # Identify breakthrough moments
                if any(word in content for word in ["breakthrough", "discovery", "innovation", "success"]):
                    summary["creative_breakthrough_moments"].append(memory["content"][:100])
            
            # Convert set to list for JSON serialization
            summary["artistic_domains_explored"] = list(summary["artistic_domains_explored"])
            
            # Assess collaboration quality
            if len(all_creative_work) > 20:
                summary["collaboration_quality"] = "established"
            elif len(all_creative_work) > 10:
                summary["collaboration_quality"] = "strong"
            
            return summary
            
        except Exception as e:
            print(f"⚠️ Error generating creative progress summary: {e}")
            return {"error": str(e)}
    
    async def end_creative_session(self, user_id: str, project_summary: str = None):
        """End creative collaboration session"""
        if self.current_session:
            summary = project_summary or "Creative collaboration session completed with artistic exploration and growth"
            
            await self.memory.create_episodic_memory(
                content=f"Creative session summary: {summary}",
                participants=[user_id, self.artist_name],
                location="creative_studio",
                emotional_state="accomplished",
                user_id=user_id,
                session_id=self.current_session
            )
            
            await self.memory.end_session(self.current_session)
            print(f"🏁 Ended creative session: {self.current_session}")
            self.current_session = None

async def demo_creative_collaboration():
    """Demonstrate the creative collaboration AI system"""
    
    print("="*70)
    print("🎨 Creative Collaboration AI Demo")
    print("="*70)
    
    creative_ai = CreativeCollaborationAI("Aria")
    user_id = "artist_maya"
    
    try:
        # Session 1: First creative collaboration - writing
        print("\n🔄 Session 1: First Creative Writing Collaboration")
        print("-" * 50)
        
        greeting = await creative_ai.start_creative_session(user_id, "short_story", "writing")
        print(f"Aria: {greeting}")
        
        response1 = await creative_ai.collaborate_creatively(
            "I want to write a short story about a character who discovers they can see memories in objects. I'm not sure how to start.", 
            user_id
        )
        print(f"User: I want to write a short story about a character who discovers they can see memories in objects. I'm not sure how to start.")
        print(f"Aria: {response1['response'][:300]}...")
        print(f"Creative Type: {response1['creative_type']}")
        print(f"Intent: Seeking creative guidance")
        
        response2 = await creative_ai.collaborate_creatively(
            "I like the idea of starting with a specific object. Maybe an old watch? I want the tone to be mysterious but hopeful.", 
            user_id
        )
        print(f"User: I like the idea of starting with a specific object. Maybe an old watch? I want the tone to be mysterious but hopeful.")
        print(f"Aria: {response2['response'][:300]}...")
        
        await creative_ai.end_creative_session(user_id, "Collaborative short story development with mysterious tone and object-focused narrative")
        
        # Store creative preferences from first session
        await creative_ai.memory.create_social_memory(
            content="Artist prefers mysterious themes with hopeful undertones in writing. Enjoys object-centered narratives.",
            person_id=user_id,
            relationship_type="creative_partner",
            user_id=user_id,
            session_id=None
        )
        
        # Session 2: Return collaboration - visual art
        print("\n🔄 Session 2: Expanding to Visual Art")
        print("-" * 40)
        
        greeting2 = await creative_ai.start_creative_session(user_id, "digital_illustration", "visual_art")
        print(f"Aria: {greeting2}")
        
        response3 = await creative_ai.collaborate_creatively(
            "I want to create a digital illustration for the story we worked on. Something that captures that mysterious feeling but shows hope.", 
            user_id
        )
        print(f"User: I want to create a digital illustration for the story we worked on. Something that captures that mysterious feeling but shows hope.")
        print(f"Aria: {response3['response'][:300]}...")
        print(f"Building on previous work: {response3['context_used']}")
        
        response4 = await creative_ai.collaborate_creatively(
            "I'm thinking dark colors with a single bright element. Maybe the watch glowing? I want to try something more experimental with the composition.", 
            user_id
        )
        print(f"User: I'm thinking dark colors with a single bright element. Maybe the watch glowing? I want to try something more experimental with the composition.")
        print(f"Aria: {response4['response'][:300]}...")
        print(f"Growth indicators: {response4['artistic_growth']}")
        
        await creative_ai.end_creative_session(user_id, "Cross-medium creative exploration: story to visual art with experimental composition")
        
        # Session 3: Advanced collaboration
        print("\n🔄 Session 3: Advanced Creative Challenge")
        print("-" * 45)
        
        greeting3 = await creative_ai.start_creative_session(user_id, "multimedia_project", "multimedia")
        print(f"Aria: {greeting3}")
        
        response5 = await creative_ai.collaborate_creatively(
            "I want to challenge myself. What if I created an interactive experience combining the story, illustration, and maybe sound? Something people can explore.", 
            user_id
        )
        print(f"User: I want to challenge myself. What if I created an interactive experience combining the story, illustration, and maybe sound? Something people can explore.")
        print(f"Aria: {response5['response'][:300]}...")
        
        # Show creative exercises
        exercises = await creative_ai.suggest_creative_exercises(user_id)
        print(f"\n🎯 Personalized Creative Exercises:")
        for i, exercise in enumerate(exercises, 1):
            print(f"{i}. {exercise}")
        
        await creative_ai.end_creative_session(user_id, "Advanced multimedia creative challenge with cross-disciplinary integration")
        
        # Show artistic progress summary
        print("\n📊 Artistic Growth Summary")
        print("-" * 30)
        
        summary = await creative_ai.get_creative_progress_summary(user_id)
        print(f"Total collaborations: {summary['total_collaborations']}")
        print(f"Artistic domains explored: {', '.join(summary['artistic_domains_explored'])}")
        print(f"Collaboration quality: {summary['collaboration_quality']}")
        print(f"Creative breakthrough moments: {len(summary['creative_breakthrough_moments'])}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ Creative Collaboration Demo Complete!")
    print("="*70)
    print("Key features demonstrated:")
    print("• Artistic style learning and adaptation")
    print("• Cross-medium creative development")
    print("• Creative relationship building")
    print("• Personalized artistic guidance")
    print("• Creative growth tracking")
    print("• Experimental encouragement")

if __name__ == "__main__":
    asyncio.run(demo_creative_collaboration())

"""
Educational Mentor Example

This example demonstrates building an AI educational mentor that:
- Adapts to individual learning styles and pace
- Remembers learning progress and knowledge gaps
- Provides personalized curriculum recommendations
- Tracks long-term educational development
- Offers targeted support for specific subjects

Shows advanced educational memory patterns and personalized learning.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from neuron_memory import NeuronMemoryAPI

class EducationalMentorAI:
    """AI educational mentor with personalized learning memory"""
    
    def __init__(self, mentor_name="Professor Ada"):
        self.memory = NeuronMemoryAPI()
        self.mentor_name = mentor_name
        self.current_session = None
        
        # Subject domains and skills
        self.subject_domains = {
            "mathematics": ["algebra", "geometry", "calculus", "statistics", "discrete_math"],
            "science": ["physics", "chemistry", "biology", "earth_science", "computer_science"],
            "language_arts": ["reading", "writing", "grammar", "literature", "creative_writing"],
            "history": ["world_history", "american_history", "ancient_civilizations", "modern_history"],
            "languages": ["spanish", "french", "german", "chinese", "japanese"],
            "arts": ["music", "visual_arts", "theater", "dance", "digital_arts"],
            "practical_skills": ["coding", "research", "critical_thinking", "presentation", "study_skills"]
        }
        
        # Learning styles and preferences
        self.learning_styles = {
            "visual": ["diagrams", "charts", "videos", "infographics", "mind_maps"],
            "auditory": ["lectures", "discussions", "podcasts", "verbal_explanations", "music"],
            "kinesthetic": ["hands_on", "experiments", "simulations", "role_play", "physical_activity"],
            "reading": ["textbooks", "articles", "written_instructions", "research", "note_taking"]
        }
    
    async def start_learning_session(self, student_id: str, subject: str, learning_goal: str = None) -> str:
        """Start a new personalized learning session"""
        self.current_session = f"learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{student_id}"
        
        await self.memory.start_session(
            session_id=self.current_session,
            user_id=student_id,
            task=f"Learning session - {subject}" + (f" - {learning_goal}" if learning_goal else ""),
            domain="education"
        )
        
        # Get student profile and learning history
        student_profile = await self._get_student_profile(student_id)
        learning_history = await self._get_learning_history(student_id, subject)
        
        greeting = await self._create_educational_greeting(student_profile, learning_history, subject, learning_goal)
        
        # Log session start
        await self._log_educational_event(
            "session_start", greeting, student_id, "greeting", 
            {"subject": subject, "goal": learning_goal}
        )
        
        print(f"📚 Started learning session: {self.current_session}")
        return greeting
    
    async def provide_instruction(self, learning_request: str, student_id: str) -> Dict[str, Any]:
        """Provide personalized instruction based on student needs"""
        
        print(f"\n🎓 Learning Request: {learning_request}")
        
        # Analyze learning request
        learning_analysis = await self._analyze_learning_request(learning_request)
        
        # Get relevant educational context
        educational_context = await self.memory.get_context_for_llm(
            query=learning_request,
            user_id=student_id,
            session_id=self.current_session,
            max_context_length=1000
        )
        
        # Get student's learning profile and preferences
        student_profile = await self._get_student_profile(student_id)
        knowledge_gaps = await self._identify_knowledge_gaps(student_id, learning_analysis["subject"])
        
        # Generate personalized instruction
        instruction = await self._generate_personalized_instruction(
            learning_request, learning_analysis, educational_context, student_profile, knowledge_gaps
        )
        
        # Update learning progress
        await self._update_learning_progress(learning_request, instruction, learning_analysis, student_id)
        
        # Log the educational interaction
        await self._log_educational_event(
            learning_request, instruction, student_id, "instruction", learning_analysis
        )
        
        return {
            "instruction": instruction,
            "subject": learning_analysis["subject"],
            "difficulty_level": learning_analysis["difficulty_level"],
            "learning_style_match": learning_analysis["learning_style_match"],
            "knowledge_gaps_addressed": learning_analysis.get("knowledge_gaps_addressed", []),
            "context_used": bool(educational_context)
        }
    
    async def assess_understanding(self, student_response: str, student_id: str) -> Dict[str, Any]:
        """Assess student understanding and provide feedback"""
        
        print(f"\n✅ Student Response: {student_response}")
        
        # Analyze student response for understanding
        understanding_analysis = await self._analyze_understanding(student_response)
        
        # Get current topic context
        topic_context = await self.memory.search_memories(
            query=f"current learning topic {student_id}",
            user_id=student_id,
            memory_types=["working"],
            limit=5
        )
        
        # Generate assessment and feedback
        assessment = await self._generate_assessment_feedback(
            student_response, understanding_analysis, topic_context, student_id
        )
        
        # Update knowledge state
        await self._update_knowledge_state(understanding_analysis, student_id)
        
        return {
            "understanding_level": understanding_analysis["understanding_level"],
            "concept_mastery": understanding_analysis["concept_mastery"],
            "feedback": assessment,
            "next_steps": understanding_analysis.get("next_steps", []),
            "misconceptions": understanding_analysis.get("misconceptions", [])
        }
    
    async def _get_student_profile(self, student_id: str) -> Dict[str, Any]:
        """Get comprehensive student learning profile"""
        try:
            # Get learning preferences and style
            learning_memories = await self.memory.search_memories(
                query=f"student {student_id} learning style preferences pace",
                user_id=student_id,
                memory_types=["social", "semantic"],
                limit=20
            )
            
            # Get academic history
            academic_history = await self.memory.search_memories(
                query=f"student {student_id} academic performance grades subjects",
                user_id=student_id,
                memory_types=["episodic", "semantic"],
                limit=15
            )
            
            profile = {
                "learning_style": "balanced",
                "preferred_pace": "moderate",
                "academic_level": "intermediate",
                "strong_subjects": [],
                "challenging_subjects": [],
                "motivation_factors": [],
                "learning_goals": [],
                "total_sessions": len(learning_memories)
            }
            
            # Analyze learning patterns
            for memory in learning_memories:
                content = memory["content"].lower()
                
                # Determine learning style
                for style, indicators in self.learning_styles.items():
                    if any(indicator in content for indicator in indicators):
                        profile["learning_style"] = style
                        break
                
                # Determine pace preference
                if "fast" in content or "quick" in content:
                    profile["preferred_pace"] = "fast"
                elif "slow" in content or "careful" in content:
                    profile["preferred_pace"] = "slow"
                
                # Identify motivation factors
                if "challenge" in content:
                    profile["motivation_factors"].append("challenge")
                if "achievement" in content or "success" in content:
                    profile["motivation_factors"].append("achievement")
                if "curiosity" in content or "interest" in content:
                    profile["motivation_factors"].append("curiosity")
            
            # Analyze academic performance
            for memory in academic_history:
                content = memory["content"].lower()
                
                # Identify strong subjects
                for subject, skills in self.subject_domains.items():
                    if subject in content and any(word in content for word in ["excellent", "strong", "good"]):
                        if subject not in profile["strong_subjects"]:
                            profile["strong_subjects"].append(subject)
                    elif subject in content and any(word in content for word in ["difficult", "struggle", "weak"]):
                        if subject not in profile["challenging_subjects"]:
                            profile["challenging_subjects"].append(subject)
            
            return profile
            
        except Exception as e:
            print(f"⚠️ Error getting student profile: {e}")
            return {
                "learning_style": "balanced",
                "preferred_pace": "moderate",
                "academic_level": "intermediate",
                "total_sessions": 0
            }
    
    async def _get_learning_history(self, student_id: str, subject: str) -> Dict[str, Any]:
        """Get learning history for specific subject"""
        try:
            subject_memories = await self.memory.search_memories(
                query=f"student {student_id} {subject} learning progress",
                user_id=student_id,
                memory_types=["episodic", "semantic", "procedural"],
                limit=20
            )
            
            history = {
                "sessions_completed": 0,
                "topics_covered": [],
                "current_level": "beginner",
                "progress_trend": "steady",
                "last_session_date": None,
                "mastered_concepts": [],
                "struggling_concepts": []
            }
            
            # Analyze subject-specific learning
            for memory in subject_memories:
                content = memory["content"].lower()
                
                if "session" in content:
                    history["sessions_completed"] += 1
                
                # Extract mastered concepts
                if "mastered" in content or "understands" in content:
                    # Extract the concept (simplified)
                    words = content.split()
                    for i, word in enumerate(words):
                        if word in ["mastered", "understands"] and i < len(words) - 1:
                            concept = words[i + 1]
                            if concept not in history["mastered_concepts"]:
                                history["mastered_concepts"].append(concept)
                
                # Extract struggling concepts
                if "struggling" in content or "difficulty" in content:
                    words = content.split()
                    for i, word in enumerate(words):
                        if word in ["struggling", "difficulty"] and "with" in words[i:i+3]:
                            try:
                                with_index = words[i:i+3].index("with") + i
                                if with_index < len(words) - 1:
                                    concept = words[with_index + 1]
                                    if concept not in history["struggling_concepts"]:
                                        history["struggling_concepts"].append(concept)
                            except (ValueError, IndexError):
                                pass
            
            # Determine current level
            if history["sessions_completed"] > 20:
                history["current_level"] = "advanced"
            elif history["sessions_completed"] > 10:
                history["current_level"] = "intermediate"
            
            return history
            
        except Exception as e:
            print(f"⚠️ Error getting learning history: {e}")
            return {"sessions_completed": 0, "current_level": "beginner"}
    
    async def _create_educational_greeting(
        self, 
        student_profile: Dict, 
        learning_history: Dict, 
        subject: str, 
        learning_goal: str
    ) -> str:
        """Create personalized educational greeting"""
        
        level = student_profile.get("academic_level", "intermediate")
        sessions = student_profile.get("total_sessions", 0)
        learning_style = student_profile.get("learning_style", "balanced")
        
        # Relationship-based greeting
        if sessions == 0:
            base_greeting = f"Hello! I'm {self.mentor_name}, your personal learning mentor. I'm excited to help you master {subject}!"
        elif sessions < 10:
            base_greeting = f"Welcome back! I can see you're making great progress in your learning journey. Ready to dive into {subject}?"
        else:
            base_greeting = f"Hello again! I'm impressed with your dedication to learning. Let's continue building your expertise in {subject}."
        
        # Add learning goal if specified
        if learning_goal:
            base_greeting += f" Today we'll focus on {learning_goal}."
        
        # Add learning style adaptation
        if learning_style == "visual":
            base_greeting += " I'll make sure to include plenty of visual aids and diagrams to help you understand."
        elif learning_style == "auditory":
            base_greeting += " I'll explain concepts verbally and we can discuss them together."
        elif learning_style == "kinesthetic":
            base_greeting += " We'll use hands-on examples and practical applications."
        elif learning_style == "reading":
            base_greeting += " I'll provide detailed explanations and resources for you to read."
        
        # Add encouragement based on history
        if learning_history.get("sessions_completed", 0) > 5:
            mastered = len(learning_history.get("mastered_concepts", []))
            base_greeting += f" I remember you've already mastered {mastered} key concepts in {subject}. Let's build on that foundation!"
        
        return base_greeting
    
    async def _analyze_learning_request(self, learning_request: str) -> Dict[str, Any]:
        """Analyze learning request for subject, difficulty, and learning style"""
        
        analysis = {
            "subject": "general",
            "difficulty_level": "intermediate",
            "learning_style_match": "balanced",
            "request_type": "explanation",
            "specific_topics": [],
            "knowledge_gaps_addressed": []
        }
        
        request_lower = learning_request.lower()
        
        # Identify subject
        for subject, skills in self.subject_domains.items():
            if subject in request_lower or any(skill in request_lower for skill in skills):
                analysis["subject"] = subject
                break
        
        # Determine request type
        if any(word in request_lower for word in ["explain", "what is", "how does", "why"]):
            analysis["request_type"] = "explanation"
        elif any(word in request_lower for word in ["solve", "calculate", "find", "compute"]):
            analysis["request_type"] = "problem_solving"
        elif any(word in request_lower for word in ["example", "show me", "demonstrate"]):
            analysis["request_type"] = "example"
        elif any(word in request_lower for word in ["practice", "exercise", "quiz", "test"]):
            analysis["request_type"] = "practice"
        elif any(word in request_lower for word in ["help", "stuck", "confused", "don't understand"]):
            analysis["request_type"] = "help"
        
        # Determine difficulty level
        if any(word in request_lower for word in ["basic", "simple", "easy", "beginner"]):
            analysis["difficulty_level"] = "beginner"
        elif any(word in request_lower for word in ["advanced", "complex", "difficult", "expert"]):
            analysis["difficulty_level"] = "advanced"
        
        # Identify learning style preferences in request
        for style, indicators in self.learning_styles.items():
            if any(indicator in request_lower for indicator in indicators):
                analysis["learning_style_match"] = style
                break
        
        return analysis
    
    async def _identify_knowledge_gaps(self, student_id: str, subject: str) -> List[str]:
        """Identify knowledge gaps in specific subject"""
        try:
            # Get struggling concepts
            struggling_memories = await self.memory.search_memories(
                query=f"student {student_id} {subject} struggling difficulty confused",
                user_id=student_id,
                memory_types=["episodic", "semantic"],
                limit=15
            )
            
            # Get mastered concepts
            mastered_memories = await self.memory.search_memories(
                query=f"student {student_id} {subject} mastered understands confident",
                user_id=student_id,
                memory_types=["semantic"],
                limit=15
            )
            
            knowledge_gaps = []
            
            # Extract struggling concepts
            for memory in struggling_memories:
                content = memory["content"].lower()
                if subject in content:
                    # Simple extraction of concepts after "with"
                    words = content.split()
                    for i, word in enumerate(words):
                        if word == "with" and i < len(words) - 1:
                            concept = words[i + 1].strip('.,!?')
                            if concept not in knowledge_gaps and len(concept) > 2:
                                knowledge_gaps.append(concept)
            
            return knowledge_gaps[:5]  # Return top 5 gaps
            
        except Exception as e:
            print(f"⚠️ Error identifying knowledge gaps: {e}")
            return []
    
    async def _generate_personalized_instruction(
        self, 
        learning_request: str, 
        analysis: Dict, 
        context: str, 
        student_profile: Dict, 
        knowledge_gaps: List[str]
    ) -> str:
        """Generate personalized instruction based on student profile"""
        
        request_type = analysis["request_type"]
        difficulty_level = analysis["difficulty_level"]
        learning_style = student_profile.get("learning_style", "balanced")
        
        # Base instruction based on request type
        if request_type == "explanation":
            instruction_start = "Let me explain this concept in a way that connects to what you already know. "
        elif request_type == "problem_solving":
            instruction_start = "Great! Let's work through this step-by-step. I'll guide you through the process. "
        elif request_type == "example":
            instruction_start = "I'll show you a practical example that demonstrates this concept clearly. "
        elif request_type == "practice":
            instruction_start = "Perfect! Practice is key to mastering this. Let's start with exercises that match your level. "
        elif request_type == "help":
            instruction_start = "I understand this can be confusing. Let's break it down into manageable pieces. "
        else:
            instruction_start = "Let's explore this topic together. "
        
        # Adapt to learning style
        if learning_style == "visual":
            instruction_start += "I'll use diagrams and visual representations to help you see the patterns. "
        elif learning_style == "auditory":
            instruction_start += "I'll explain this verbally and we can discuss it as we go. "
        elif learning_style == "kinesthetic":
            instruction_start += "We'll use hands-on examples and real-world applications. "
        elif learning_style == "reading":
            instruction_start += "I'll provide detailed written explanations and additional resources. "
        
        # Address knowledge gaps if relevant
        if knowledge_gaps:
            relevant_gaps = [gap for gap in knowledge_gaps if gap in learning_request.lower()]
            if relevant_gaps:
                instruction_start += f"I notice you've had some challenges with {relevant_gaps[0]} before, so I'll make sure to clarify that connection. "
        
        # Adjust for difficulty level
        if difficulty_level == "beginner":
            instruction_start += "We'll start with the fundamentals and build up gradually. "
        elif difficulty_level == "advanced":
            instruction_start += "Since you're ready for a challenge, I'll include some advanced concepts and applications. "
        
        # Add context references if available
        if context and "previous" in context.lower():
            instruction_start += "Building on what we've covered before, this connects to your earlier learning. "
        
        # Add encouragement based on profile
        motivation_factors = student_profile.get("motivation_factors", [])
        if "challenge" in motivation_factors:
            instruction_start += "This will stretch your thinking in exciting ways!"
        elif "achievement" in motivation_factors:
            instruction_start += "Mastering this will be a significant achievement!"
        elif "curiosity" in motivation_factors:
            instruction_start += "This is a fascinating topic that I think you'll find really interesting!"
        
        return instruction_start
    
    async def _analyze_understanding(self, student_response: str) -> Dict[str, Any]:
        """Analyze student response to assess understanding"""
        
        analysis = {
            "understanding_level": "partial",
            "concept_mastery": "developing",
            "confidence_level": "moderate",
            "misconceptions": [],
            "next_steps": []
        }
        
        response_lower = student_response.lower()
        
        # Assess understanding level
        if any(phrase in response_lower for phrase in ["i understand", "i get it", "that makes sense", "i see"]):
            analysis["understanding_level"] = "good"
        elif any(phrase in response_lower for phrase in ["i don't understand", "i'm confused", "this is hard"]):
            analysis["understanding_level"] = "low"
        elif any(phrase in response_lower for phrase in ["partially", "sort of", "kind of", "maybe"]):
            analysis["understanding_level"] = "partial"
        
        # Assess confidence
        if any(phrase in response_lower for phrase in ["i'm confident", "i'm sure", "definitely"]):
            analysis["confidence_level"] = "high"
        elif any(phrase in response_lower for phrase in ["i'm not sure", "i think", "maybe", "uncertain"]):
            analysis["confidence_level"] = "low"
        
        # Identify potential misconceptions
        if "always" in response_lower or "never" in response_lower:
            analysis["misconceptions"].append("absolute_thinking")
        if "because" not in response_lower and len(response_lower.split()) > 5:
            analysis["misconceptions"].append("lack_of_reasoning")
        
        # Determine next steps
        if analysis["understanding_level"] == "good":
            analysis["next_steps"] = ["practice_problems", "advanced_concepts", "real_world_applications"]
        elif analysis["understanding_level"] == "partial":
            analysis["next_steps"] = ["clarify_concepts", "more_examples", "guided_practice"]
        else:
            analysis["next_steps"] = ["review_fundamentals", "alternative_explanation", "break_down_steps"]
        
        return analysis
    
    async def _generate_assessment_feedback(
        self, 
        student_response: str, 
        understanding_analysis: Dict, 
        topic_context: List, 
        student_id: str
    ) -> str:
        """Generate personalized assessment feedback"""
        
        understanding_level = understanding_analysis["understanding_level"]
        confidence_level = understanding_analysis["confidence_level"]
        next_steps = understanding_analysis["next_steps"]
        
        # Base feedback on understanding
        if understanding_level == "good":
            feedback = "Excellent! You've grasped the key concepts well. "
        elif understanding_level == "partial":
            feedback = "You're on the right track! You understand some important parts. "
        else:
            feedback = "I can see you're working hard to understand this. Let's approach it differently. "
        
        # Address confidence level
        if confidence_level == "low" and understanding_level == "good":
            feedback += "You actually understand this better than you think! Let's build your confidence. "
        elif confidence_level == "high" and understanding_level == "low":
            feedback += "I appreciate your enthusiasm! Let's make sure your understanding matches your confidence. "
        
        # Provide next steps
        if "practice_problems" in next_steps:
            feedback += "You're ready for some practice problems to solidify your understanding. "
        elif "more_examples" in next_steps:
            feedback += "Let me give you a few more examples to help clarify this concept. "
        elif "review_fundamentals" in next_steps:
            feedback += "Let's review the fundamentals to build a stronger foundation. "
        
        # Address misconceptions
        misconceptions = understanding_analysis.get("misconceptions", [])
        if "absolute_thinking" in misconceptions:
            feedback += "Remember, most concepts in learning have exceptions and nuances. "
        if "lack_of_reasoning" in misconceptions:
            feedback += "Try to explain your reasoning - it helps deepen understanding. "
        
        return feedback
    
    async def _update_learning_progress(
        self, 
        learning_request: str, 
        instruction: str, 
        analysis: Dict, 
        student_id: str
    ):
        """Update learning progress and knowledge state"""
        try:
            subject = analysis["subject"]
            difficulty_level = analysis["difficulty_level"]
            
            # Update semantic memory with new knowledge
            await self.memory.create_semantic_memory(
                content=f"Student learned about {subject} at {difficulty_level} level: {learning_request}",
                domain=subject,
                confidence=0.7,
                user_id=student_id,
                session_id=self.current_session
            )
            
            # Update procedural memory if it's a skill
            if analysis["request_type"] == "problem_solving":
                await self.memory.create_procedural_memory(
                    content=f"Student practiced {subject} problem-solving: {learning_request}",
                    skill_domain=subject,
                    proficiency_level="developing",
                    user_id=student_id,
                    session_id=self.current_session
                )
            
            print("📚 Updated learning progress and knowledge state")
            
        except Exception as e:
            print(f"⚠️ Error updating learning progress: {e}")
    
    async def _update_knowledge_state(self, understanding_analysis: Dict, student_id: str):
        """Update knowledge state based on assessment"""
        try:
            understanding_level = understanding_analysis["understanding_level"]
            concept_mastery = understanding_analysis["concept_mastery"]
            
            # Create episodic memory of the assessment
            await self.memory.create_episodic_memory(
                content=f"Assessment: Student demonstrated {understanding_level} understanding with {concept_mastery} concept mastery",
                participants=[student_id, self.mentor_name],
                location="virtual_classroom",
                emotional_state="focused",
                user_id=student_id,
                session_id=self.current_session
            )
            
            print("✅ Updated knowledge state based on assessment")
            
        except Exception as e:
            print(f"⚠️ Error updating knowledge state: {e}")
    
    async def _log_educational_event(
        self, 
        student_input: str, 
        mentor_response: str, 
        student_id: str, 
        event_type: str,
        metadata: Dict
    ):
        """Log educational interaction events"""
        try:
            # Store as episodic memory
            educational_content = f"Educational interaction: {student_input}\nMentor response: {mentor_response}\nType: {event_type}"
            
            await self.memory.create_episodic_memory(
                content=educational_content,
                participants=[student_id, self.mentor_name],
                location="virtual_classroom",
                emotional_state="engaged",
                user_id=student_id,
                session_id=self.current_session
            )
            
        except Exception as e:
            print(f"⚠️ Error logging educational event: {e}")
    
    async def generate_personalized_curriculum(self, student_id: str, subject: str) -> Dict[str, Any]:
        """Generate personalized curriculum based on student profile"""
        try:
            student_profile = await self._get_student_profile(student_id)
            learning_history = await self._get_learning_history(student_id, subject)
            knowledge_gaps = await self._identify_knowledge_gaps(student_id, subject)
            
            curriculum = {
                "subject": subject,
                "current_level": learning_history.get("current_level", "beginner"),
                "learning_path": [],
                "estimated_duration": "4-6 weeks",
                "prerequisite_review": [],
                "advanced_topics": []
            }
            
            # Generate learning path based on level
            if curriculum["current_level"] == "beginner":
                curriculum["learning_path"] = [
                    "Introduction to fundamentals",
                    "Basic concepts and terminology",
                    "Simple applications and examples",
                    "Guided practice exercises",
                    "Review and assessment"
                ]
            elif curriculum["current_level"] == "intermediate":
                curriculum["learning_path"] = [
                    "Review of fundamentals",
                    "Intermediate concepts and relationships",
                    "Problem-solving strategies",
                    "Real-world applications",
                    "Advanced practice and synthesis"
                ]
            else:
                curriculum["learning_path"] = [
                    "Advanced theoretical concepts",
                    "Complex problem-solving",
                    "Independent research projects",
                    "Peer teaching opportunities",
                    "Mastery demonstration"
                ]
            
            # Address knowledge gaps
            if knowledge_gaps:
                curriculum["prerequisite_review"] = [f"Review {gap}" for gap in knowledge_gaps[:3]]
            
            # Suggest advanced topics
            if subject in self.subject_domains:
                advanced_skills = self.subject_domains[subject][-2:]  # Last 2 skills
                curriculum["advanced_topics"] = [f"Advanced {skill}" for skill in advanced_skills]
            
            return curriculum
            
        except Exception as e:
            print(f"⚠️ Error generating curriculum: {e}")
            return {"error": str(e)}
    
    async def get_progress_report(self, student_id: str) -> Dict[str, Any]:
        """Generate comprehensive progress report"""
        try:
            all_sessions = await self.memory.search_memories(
                query=f"student {student_id} learning session",
                user_id=student_id,
                limit=50
            )
            
            subject_progress = {}
            
            # Analyze progress by subject
            for memory in all_sessions:
                content = memory["content"].lower()
                for subject in self.subject_domains.keys():
                    if subject in content:
                        if subject not in subject_progress:
                            subject_progress[subject] = {
                                "sessions": 0,
                                "mastered_concepts": 0,
                                "struggling_areas": 0
                            }
                        subject_progress[subject]["sessions"] += 1
                        
                        if "mastered" in content:
                            subject_progress[subject]["mastered_concepts"] += 1
                        if "struggling" in content:
                            subject_progress[subject]["struggling_areas"] += 1
            
            report = {
                "total_sessions": len(all_sessions),
                "subjects_studied": list(subject_progress.keys()),
                "subject_progress": subject_progress,
                "overall_trend": "improving",
                "strengths": [],
                "areas_for_improvement": []
            }
            
            # Identify strengths and areas for improvement
            for subject, progress in subject_progress.items():
                if progress["mastered_concepts"] > progress["struggling_areas"]:
                    report["strengths"].append(subject)
                else:
                    report["areas_for_improvement"].append(subject)
            
            return report
            
        except Exception as e:
            print(f"⚠️ Error generating progress report: {e}")
            return {"error": str(e)}
    
    async def end_learning_session(self, student_id: str, session_summary: str = None):
        """End learning session with summary"""
        if self.current_session:
            summary = session_summary or "Learning session completed with educational progress"
            
            await self.memory.create_episodic_memory(
                content=f"Learning session summary: {summary}",
                participants=[student_id, self.mentor_name],
                location="virtual_classroom",
                emotional_state="accomplished",
                user_id=student_id,
                session_id=self.current_session
            )
            
            await self.memory.end_session(self.current_session)
            print(f"🏁 Ended learning session: {self.current_session}")
            self.current_session = None

async def demo_educational_mentor():
    """Demonstrate the educational mentor AI system"""
    
    print("="*70)
    print("🎓 Educational Mentor AI Demo")
    print("="*70)
    
    mentor = EducationalMentorAI("Professor Ada")
    student_id = "student_alex"
    
    try:
        # Session 1: First learning session - mathematics
        print("\n🔄 Session 1: First Mathematics Learning Session")
        print("-" * 55)
        
        greeting = await mentor.start_learning_session(student_id, "mathematics", "understand algebra basics")
        print(f"Professor Ada: {greeting}")
        
        instruction1 = await mentor.provide_instruction(
            "I don't understand how to solve equations with variables on both sides. Can you explain?", 
            student_id
        )
        print(f"Student: I don't understand how to solve equations with variables on both sides. Can you explain?")
        print(f"Professor Ada: {instruction1['instruction'][:300]}...")
        print(f"Subject: {instruction1['subject']}")
        print(f"Difficulty: {instruction1['difficulty_level']}")
        
        assessment1 = await mentor.assess_understanding(
            "I think I understand. You move the variables to one side and numbers to the other side?", 
            student_id
        )
        print(f"Student: I think I understand. You move the variables to one side and numbers to the other side?")
        print(f"Professor Ada: {assessment1['feedback']}")
        print(f"Understanding Level: {assessment1['understanding_level']}")
        
        await mentor.end_learning_session(student_id, "Algebra basics: solving equations with variables on both sides")
        
        # Store learning preferences
        await mentor.memory.create_social_memory(
            content="Student prefers step-by-step explanations in mathematics. Responds well to visual learning approaches.",
            person_id=student_id,
            relationship_type="student",
            user_id=student_id,
            session_id=None
        )
        
        # Session 2: Follow-up session - science
        print("\n🔄 Session 2: Science Learning Session")
        print("-" * 40)
        
        greeting2 = await mentor.start_learning_session(student_id, "science", "understand chemical reactions")
        print(f"Professor Ada: {greeting2}")
        
        instruction2 = await mentor.provide_instruction(
            "I need help understanding how chemical equations are balanced. Can you show me with examples?", 
            student_id
        )
        print(f"Student: I need help understanding how chemical equations are balanced. Can you show me with examples?")
        print(f"Professor Ada: {instruction2['instruction'][:300]}...")
        print(f"Learning Style Match: {instruction2['learning_style_match']}")
        
        assessment2 = await mentor.assess_understanding(
            "I see! You need to make sure the same number of each type of atom is on both sides of the equation.", 
            student_id
        )
        print(f"Student: I see! You need to make sure the same number of each type of atom is on both sides of the equation.")
        print(f"Professor Ada: {assessment2['feedback']}")
        print(f"Concept Mastery: {assessment2['concept_mastery']}")
        
        await mentor.end_learning_session(student_id, "Chemistry: balancing chemical equations with hands-on examples")
        
        # Session 3: Advanced session with curriculum planning
        print("\n🔄 Session 3: Advanced Learning with Curriculum Planning")
        print("-" * 60)
        
        greeting3 = await mentor.start_learning_session(student_id, "mathematics", "tackle more complex problems")
        print(f"Professor Ada: {greeting3}")
        
        instruction3 = await mentor.provide_instruction(
            "I want to challenge myself with quadratic equations. I'm ready for something more difficult.", 
            student_id
        )
        print(f"Student: I want to challenge myself with quadratic equations. I'm ready for something more difficult.")
        print(f"Professor Ada: {instruction3['instruction'][:300]}...")
        
        # Generate personalized curriculum
        curriculum = await mentor.generate_personalized_curriculum(student_id, "mathematics")
        print(f"\n📋 Personalized Mathematics Curriculum:")
        print(f"Current Level: {curriculum['current_level']}")
        print(f"Learning Path:")
        for i, step in enumerate(curriculum['learning_path'], 1):
            print(f"  {i}. {step}")
        
        await mentor.end_learning_session(student_id, "Advanced mathematics: introduction to quadratic equations")
        
        # Generate progress report
        print("\n📊 Student Progress Report")
        print("-" * 30)
        
        report = await mentor.get_progress_report(student_id)
        print(f"Total learning sessions: {report['total_sessions']}")
        print(f"Subjects studied: {', '.join(report['subjects_studied'])}")
        print(f"Strengths: {', '.join(report['strengths'])}")
        print(f"Areas for improvement: {', '.join(report['areas_for_improvement'])}")
        print(f"Overall trend: {report['overall_trend']}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ Educational Mentor Demo Complete!")
    print("="*70)
    print("Key features demonstrated:")
    print("• Personalized learning style adaptation")
    print("• Knowledge gap identification and addressing")
    print("• Progressive curriculum development")
    print("• Continuous assessment and feedback")
    print("• Long-term progress tracking")
    print("• Multi-subject learning support")

if __name__ == "__main__":
    asyncio.run(demo_educational_mentor())

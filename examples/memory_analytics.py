"""
Memory Analytics Example

This example demonstrates advanced memory analytics and insights for NeuronMemory:
- Memory usage pattern analysis and optimization
- Knowledge gap identification and recommendations
- Memory quality assessment and improvement
- Cross-session learning insights and trends
- Performance metrics and memory health monitoring

Shows comprehensive memory analytics capabilities and intelligent insights.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import json
from neuron_memory import NeuronMemoryAPI

class MemoryAnalyticsEngine:
    """Advanced memory analytics and insights engine"""
    
    def __init__(self):
        self.memory = NeuronMemoryAPI()
        
        # Analytics categories
        self.analytics_categories = {
            "usage_patterns": ["access_frequency", "temporal_distribution", "session_patterns"],
            "content_analysis": ["topic_distribution", "complexity_analysis", "quality_metrics"],
            "learning_insights": ["knowledge_growth", "retention_patterns", "mastery_tracking"],
            "performance_metrics": ["retrieval_efficiency", "storage_optimization", "memory_health"],
            "behavioral_patterns": ["interaction_styles", "preference_evolution", "engagement_trends"]
        }
        
        # Memory quality indicators
        self.quality_indicators = {
            "high_quality": ["detailed", "comprehensive", "accurate", "verified", "structured"],
            "medium_quality": ["partial", "basic", "general", "unverified", "brief"],
            "low_quality": ["incomplete", "vague", "contradictory", "outdated", "fragmented"]
        }
    
    async def analyze_memory_usage_patterns(self, user_id: str, days_back: int = 30) -> Dict[str, Any]:
        """Analyze memory usage patterns for insights and optimization"""
        
        print(f"🔍 Analyzing memory usage patterns for user {user_id} over {days_back} days...")
        
        # Get all memories for analysis
        all_memories = await self.memory.search_memories(
            query=f"user {user_id}",
            user_id=user_id,
            limit=1000
        )
        
        # Get session data
        session_memories = await self.memory.search_memories(
            query=f"session user {user_id}",
            user_id=user_id,
            memory_types=["episodic"],
            limit=200
        )
        
        usage_analysis = {
            "total_memories": len(all_memories),
            "memory_type_distribution": defaultdict(int),
            "temporal_patterns": {
                "daily_creation_rate": 0,
                "peak_usage_times": [],
                "session_frequency": 0
            },
            "access_patterns": {
                "most_accessed_topics": [],
                "retrieval_frequency": 0,
                "search_patterns": []
            },
            "quality_distribution": {
                "high_quality": 0,
                "medium_quality": 0,
                "low_quality": 0
            }
        }
        
        # Analyze memory type distribution
        for memory in all_memories:
            memory_type = memory.get("type", "unknown")
            usage_analysis["memory_type_distribution"][memory_type] += 1
        
        # Analyze temporal patterns
        if session_memories:
            usage_analysis["temporal_patterns"]["session_frequency"] = len(session_memories) / days_back
            usage_analysis["temporal_patterns"]["daily_creation_rate"] = len(all_memories) / days_back
        
        # Analyze content quality
        for memory in all_memories[:100]:  # Sample for performance
            quality_score = await self._assess_memory_quality(memory)
            if quality_score >= 0.7:
                usage_analysis["quality_distribution"]["high_quality"] += 1
            elif quality_score >= 0.4:
                usage_analysis["quality_distribution"]["medium_quality"] += 1
            else:
                usage_analysis["quality_distribution"]["low_quality"] += 1
        
        # Generate insights
        insights = await self._generate_usage_insights(usage_analysis, user_id)
        usage_analysis["insights"] = insights
        
        print(f"✅ Memory usage analysis complete: {usage_analysis['total_memories']} memories analyzed")
        return usage_analysis
    
    async def analyze_knowledge_gaps(self, user_id: str, domain: str = None) -> Dict[str, Any]:
        """Identify knowledge gaps and learning opportunities"""
        
        print(f"🧩 Analyzing knowledge gaps for user {user_id}" + (f" in {domain}" if domain else ""))
        
        # Get semantic and procedural memories
        knowledge_memories = await self.memory.search_memories(
            query=f"user {user_id}" + (f" {domain}" if domain else ""),
            user_id=user_id,
            memory_types=["semantic", "procedural"],
            limit=500
        )
        
        # Get episodic memories for context
        experience_memories = await self.memory.search_memories(
            query=f"user {user_id} learning struggling difficulty",
            user_id=user_id,
            memory_types=["episodic"],
            limit=200
        )
        
        gap_analysis = {
            "total_knowledge_items": len(knowledge_memories),
            "identified_gaps": [],
            "learning_opportunities": [],
            "mastery_levels": {},
            "recommended_actions": [],
            "domain_coverage": {}
        }
        
        # Analyze knowledge distribution
        domain_knowledge = defaultdict(list)
        for memory in knowledge_memories:
            content = memory["content"].lower()
            memory_domain = memory.get("domain", "general")
            domain_knowledge[memory_domain].append(memory)
        
        gap_analysis["domain_coverage"] = {
            domain: len(memories) for domain, memories in domain_knowledge.items()
        }
        
        # Identify gaps from struggling experiences
        for memory in experience_memories:
            content = memory["content"].lower()
            if any(word in content for word in ["struggling", "difficult", "confused", "don't understand"]):
                gap_content = await self._extract_knowledge_gap(memory)
                if gap_content and gap_content not in gap_analysis["identified_gaps"]:
                    gap_analysis["identified_gaps"].append(gap_content)
        
        # Generate learning opportunities
        gap_analysis["learning_opportunities"] = await self._generate_learning_opportunities(
            gap_analysis["identified_gaps"], domain_knowledge
        )
        
        # Generate recommendations
        gap_analysis["recommended_actions"] = await self._generate_gap_recommendations(gap_analysis)
        
        print(f"✅ Knowledge gap analysis complete: {len(gap_analysis['identified_gaps'])} gaps identified")
        return gap_analysis
    
    async def analyze_learning_progress(self, user_id: str, timeframe_days: int = 90) -> Dict[str, Any]:
        """Analyze learning progress and knowledge evolution"""
        
        print(f"📈 Analyzing learning progress for user {user_id} over {timeframe_days} days...")
        
        # Get learning-related memories
        learning_memories = await self.memory.search_memories(
            query=f"user {user_id} learning progress mastered",
            user_id=user_id,
            limit=300
        )
        
        # Get assessment and feedback memories
        assessment_memories = await self.memory.search_memories(
            query=f"user {user_id} assessment feedback understanding",
            user_id=user_id,
            memory_types=["episodic"],
            limit=200
        )
        
        progress_analysis = {
            "learning_trajectory": "steady",
            "mastery_progression": [],
            "skill_development": {},
            "learning_velocity": 0,
            "knowledge_retention": 0,
            "breakthrough_moments": [],
            "learning_patterns": {},
            "projected_growth": {}
        }
        
        # Analyze mastery progression
        mastery_timeline = []
        for memory in learning_memories:
            if "mastered" in memory["content"].lower() or "understands" in memory["content"].lower():
                mastery_event = {
                    "timestamp": memory.get("created_at", datetime.now().isoformat()),
                    "content": memory["content"][:100],
                    "domain": memory.get("domain", "general")
                }
                mastery_timeline.append(mastery_event)
        
        progress_analysis["mastery_progression"] = sorted(mastery_timeline, key=lambda x: x["timestamp"])
        
        # Calculate learning velocity
        if len(mastery_timeline) > 1:
            days_span = timeframe_days
            progress_analysis["learning_velocity"] = len(mastery_timeline) / days_span
        
        # Identify breakthrough moments
        for memory in assessment_memories:
            content = memory["content"].lower()
            if any(word in content for word in ["breakthrough", "finally", "aha", "suddenly understand"]):
                progress_analysis["breakthrough_moments"].append({
                    "content": memory["content"][:150],
                    "timestamp": memory.get("created_at", datetime.now().isoformat())
                })
        
        # Analyze skill development by domain
        domain_progress = defaultdict(int)
        for event in mastery_timeline:
            domain_progress[event["domain"]] += 1
        
        progress_analysis["skill_development"] = dict(domain_progress)
        
        # Generate learning insights
        progress_analysis["insights"] = await self._generate_progress_insights(progress_analysis)
        
        print(f"✅ Learning progress analysis complete: {len(mastery_timeline)} mastery events tracked")
        return progress_analysis
    
    async def analyze_memory_quality(self, user_id: str, sample_size: int = 200) -> Dict[str, Any]:
        """Analyze memory quality and suggest improvements"""
        
        print(f"⭐ Analyzing memory quality for user {user_id} (sample: {sample_size} memories)...")
        
        # Get recent memories for quality analysis
        recent_memories = await self.memory.search_memories(
            query=f"user {user_id}",
            user_id=user_id,
            limit=sample_size
        )
        
        quality_analysis = {
            "total_analyzed": len(recent_memories),
            "quality_distribution": {
                "excellent": 0,
                "good": 0,
                "fair": 0,
                "poor": 0
            },
            "quality_factors": {
                "completeness": 0,
                "accuracy": 0,
                "relevance": 0,
                "structure": 0
            },
            "improvement_suggestions": [],
            "best_practices": [],
            "memory_health_score": 0
        }
        
        total_quality_score = 0
        
        # Analyze each memory
        for memory in recent_memories:
            quality_score = await self._assess_memory_quality(memory)
            quality_details = await self._analyze_quality_factors(memory)
            
            total_quality_score += quality_score
            
            # Categorize quality
            if quality_score >= 0.8:
                quality_analysis["quality_distribution"]["excellent"] += 1
            elif quality_score >= 0.6:
                quality_analysis["quality_distribution"]["good"] += 1
            elif quality_score >= 0.4:
                quality_analysis["quality_distribution"]["fair"] += 1
            else:
                quality_analysis["quality_distribution"]["poor"] += 1
            
            # Aggregate quality factors
            for factor, score in quality_details.items():
                quality_analysis["quality_factors"][factor] += score
        
        # Calculate averages
        if recent_memories:
            for factor in quality_analysis["quality_factors"]:
                quality_analysis["quality_factors"][factor] /= len(recent_memories)
            
            quality_analysis["memory_health_score"] = total_quality_score / len(recent_memories)
        
        # Generate improvement suggestions
        quality_analysis["improvement_suggestions"] = await self._generate_quality_improvements(quality_analysis)
        quality_analysis["best_practices"] = await self._generate_best_practices(quality_analysis)
        
        print(f"✅ Memory quality analysis complete: {quality_analysis['memory_health_score']:.2f} health score")
        return quality_analysis
    
    async def analyze_cross_session_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze patterns across multiple sessions"""
        
        print(f"🔄 Analyzing cross-session patterns for user {user_id}...")
        
        # Get session data
        session_memories = await self.memory.search_memories(
            query=f"user {user_id} session",
            user_id=user_id,
            memory_types=["episodic"],
            limit=100
        )
        
        pattern_analysis = {
            "total_sessions": len(session_memories),
            "session_continuity": 0,
            "recurring_themes": [],
            "learning_threads": [],
            "relationship_development": {},
            "topic_evolution": {},
            "engagement_patterns": {}
        }
        
        # Analyze session continuity
        continuous_sessions = []
        for memory in session_memories:
            content = memory["content"].lower()
            if any(phrase in content for phrase in ["building on", "continuing", "following up"]):
                continuous_sessions.append(memory)
        
        if session_memories:
            pattern_analysis["session_continuity"] = len(continuous_sessions) / len(session_memories)
        
        # Identify recurring themes
        theme_counts = defaultdict(int)
        for memory in session_memories:
            themes = await self._extract_themes(memory["content"])
            for theme in themes:
                theme_counts[theme] += 1
        
        # Get top recurring themes
        pattern_analysis["recurring_themes"] = sorted(
            theme_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        # Analyze topic evolution
        pattern_analysis["topic_evolution"] = await self._analyze_topic_evolution(session_memories)
        
        # Generate insights
        pattern_analysis["insights"] = await self._generate_pattern_insights(pattern_analysis)
        
        print(f"✅ Cross-session pattern analysis complete: {pattern_analysis['total_sessions']} sessions analyzed")
        return pattern_analysis
    
    async def generate_comprehensive_report(self, user_id: str) -> Dict[str, Any]:
        """Generate comprehensive memory analytics report"""
        
        print(f"📊 Generating comprehensive memory analytics report for user {user_id}...")
        
        # Run all analyses in parallel for efficiency
        usage_analysis, gap_analysis, progress_analysis, quality_analysis, pattern_analysis = await asyncio.gather(
            self.analyze_memory_usage_patterns(user_id),
            self.analyze_knowledge_gaps(user_id),
            self.analyze_learning_progress(user_id),
            self.analyze_memory_quality(user_id),
            self.analyze_cross_session_patterns(user_id)
        )
        
        # Compile comprehensive report
        comprehensive_report = {
            "user_id": user_id,
            "report_generated": datetime.now().isoformat(),
            "executive_summary": {},
            "detailed_analyses": {
                "usage_patterns": usage_analysis,
                "knowledge_gaps": gap_analysis,
                "learning_progress": progress_analysis,
                "memory_quality": quality_analysis,
                "session_patterns": pattern_analysis
            },
            "recommendations": {},
            "action_items": [],
            "memory_health_dashboard": {}
        }
        
        # Generate executive summary
        comprehensive_report["executive_summary"] = await self._generate_executive_summary(
            usage_analysis, gap_analysis, progress_analysis, quality_analysis, pattern_analysis
        )
        
        # Generate unified recommendations
        comprehensive_report["recommendations"] = await self._generate_unified_recommendations(
            comprehensive_report["detailed_analyses"]
        )
        
        # Create memory health dashboard
        comprehensive_report["memory_health_dashboard"] = await self._create_health_dashboard(
            comprehensive_report["detailed_analyses"]
        )
        
        print("✅ Comprehensive memory analytics report generated successfully!")
        return comprehensive_report
    
    async def _assess_memory_quality(self, memory: Dict) -> float:
        """Assess the quality of a single memory"""
        
        content = memory.get("content", "").lower()
        quality_score = 0.5  # Base score
        
        # Check for high-quality indicators
        for indicator in self.quality_indicators["high_quality"]:
            if indicator in content:
                quality_score += 0.1
        
        # Check for medium-quality indicators
        for indicator in self.quality_indicators["medium_quality"]:
            if indicator in content:
                quality_score += 0.05
        
        # Penalize for low-quality indicators
        for indicator in self.quality_indicators["low_quality"]:
            if indicator in content:
                quality_score -= 0.1
        
        # Length-based quality assessment
        word_count = len(content.split())
        if word_count > 50:
            quality_score += 0.1
        elif word_count < 10:
            quality_score -= 0.1
        
        # Structure assessment
        if any(punct in content for punct in ['.', '!', '?']):
            quality_score += 0.05
        
        return max(0.0, min(1.0, quality_score))
    
    async def _analyze_quality_factors(self, memory: Dict) -> Dict[str, float]:
        """Analyze specific quality factors of a memory"""
        
        content = memory.get("content", "")
        
        factors = {
            "completeness": 0.5,
            "accuracy": 0.5,
            "relevance": 0.5,
            "structure": 0.5
        }
        
        # Completeness - based on content length and detail
        word_count = len(content.split())
        if word_count > 100:
            factors["completeness"] = 0.9
        elif word_count > 50:
            factors["completeness"] = 0.7
        elif word_count < 10:
            factors["completeness"] = 0.3
        
        # Structure - based on formatting and organization
        if any(marker in content for marker in ['\n', ':', ';', '.']):
            factors["structure"] += 0.2
        
        # Relevance - based on domain matching and context
        if memory.get("domain") and memory.get("domain") != "general":
            factors["relevance"] += 0.2
        
        # Cap all factors at 1.0
        for factor in factors:
            factors[factor] = min(1.0, factors[factor])
        
        return factors
    
    async def _extract_knowledge_gap(self, memory: Dict) -> Optional[str]:
        """Extract knowledge gap from struggle/difficulty memory"""
        
        content = memory["content"].lower()
        
        # Look for patterns indicating gaps
        gap_patterns = [
            ("struggling with", "with"),
            ("difficulty understanding", "understanding"),
            ("confused about", "about"),
            ("don't understand", "understand")
        ]
        
        for pattern, keyword in gap_patterns:
            if pattern in content:
                # Extract the topic after the keyword
                parts = content.split(keyword)
                if len(parts) > 1:
                    topic = parts[1].split()[0:3]  # Get next few words
                    return " ".join(topic).strip('.,!?')
        
        return None
    
    async def _generate_learning_opportunities(self, gaps: List[str], domain_knowledge: Dict) -> List[Dict]:
        """Generate learning opportunities based on identified gaps"""
        
        opportunities = []
        
        for gap in gaps[:5]:  # Top 5 gaps
            opportunity = {
                "gap": gap,
                "suggested_resources": [],
                "learning_path": [],
                "estimated_effort": "medium",
                "priority": "medium"
            }
            
            # Generate learning path
            opportunity["learning_path"] = [
                f"Review fundamentals of {gap}",
                f"Practice basic {gap} exercises",
                f"Apply {gap} to real-world scenarios",
                f"Teach {gap} to reinforce understanding"
            ]
            
            # Suggest resources
            opportunity["suggested_resources"] = [
                f"Online tutorials for {gap}",
                f"Practice exercises for {gap}",
                f"Interactive examples of {gap}"
            ]
            
            opportunities.append(opportunity)
        
        return opportunities
    
    async def _generate_gap_recommendations(self, gap_analysis: Dict) -> List[str]:
        """Generate actionable recommendations for addressing gaps"""
        
        recommendations = []
        
        gap_count = len(gap_analysis["identified_gaps"])
        
        if gap_count > 10:
            recommendations.append("Focus on addressing the top 3-5 most critical knowledge gaps first")
        elif gap_count > 5:
            recommendations.append("Create a structured learning plan to address knowledge gaps systematically")
        
        # Domain-specific recommendations
        domain_coverage = gap_analysis["domain_coverage"]
        if domain_coverage:
            max_domain = max(domain_coverage, key=domain_coverage.get)
            recommendations.append(f"Consider expanding knowledge in {max_domain} where you show strong engagement")
        
        recommendations.append("Regular review sessions to reinforce newly learned concepts")
        recommendations.append("Practice application of concepts to improve retention")
        
        return recommendations
    
    async def _extract_themes(self, content: str) -> List[str]:
        """Extract themes from memory content"""
        
        # Simple theme extraction based on keywords
        themes = []
        content_lower = content.lower()
        
        # Common themes in learning/memory contexts
        theme_keywords = {
            "learning": ["learn", "study", "understand", "knowledge"],
            "problem_solving": ["solve", "problem", "solution", "method"],
            "creativity": ["creative", "idea", "design", "art"],
            "collaboration": ["work together", "team", "collaborate", "group"],
            "analysis": ["analyze", "examine", "research", "investigate"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    async def _analyze_topic_evolution(self, session_memories: List[Dict]) -> Dict[str, Any]:
        """Analyze how topics evolve across sessions"""
        
        topic_evolution = {
            "topic_progression": [],
            "complexity_trend": "increasing",
            "breadth_vs_depth": "balanced"
        }
        
        # Simple topic tracking
        session_topics = []
        for memory in session_memories[-10:]:  # Last 10 sessions
            themes = await self._extract_themes(memory["content"])
            session_topics.append(themes)
        
        topic_evolution["topic_progression"] = session_topics
        
        return topic_evolution
    
    async def _generate_usage_insights(self, usage_analysis: Dict, user_id: str) -> List[str]:
        """Generate insights from usage analysis"""
        
        insights = []
        
        # Memory type insights
        type_dist = usage_analysis["memory_type_distribution"]
        if type_dist:
            most_common_type = max(type_dist, key=type_dist.get)
            insights.append(f"Most active memory type: {most_common_type}")
        
        # Quality insights
        quality_dist = usage_analysis["quality_distribution"]
        total_quality_memories = sum(quality_dist.values())
        if total_quality_memories > 0:
            high_quality_ratio = quality_dist["high_quality"] / total_quality_memories
            if high_quality_ratio > 0.6:
                insights.append("Memory quality is consistently high - excellent information capture")
            elif high_quality_ratio < 0.3:
                insights.append("Consider focusing on creating more detailed, structured memories")
        
        # Usage frequency insights
        daily_rate = usage_analysis["temporal_patterns"]["daily_creation_rate"]
        if daily_rate > 10:
            insights.append("High memory creation rate - very active learning")
        elif daily_rate < 1:
            insights.append("Low memory creation rate - consider more frequent learning sessions")
        
        return insights
    
    async def _generate_progress_insights(self, progress_analysis: Dict) -> List[str]:
        """Generate insights from progress analysis"""
        
        insights = []
        
        # Learning velocity insights
        velocity = progress_analysis["learning_velocity"]
        if velocity > 0.5:
            insights.append("Excellent learning velocity - mastering concepts quickly")
        elif velocity > 0.2:
            insights.append("Steady learning progress - consistent skill development")
        else:
            insights.append("Consider increasing learning intensity for faster progress")
        
        # Breakthrough insights
        breakthroughs = len(progress_analysis["breakthrough_moments"])
        if breakthroughs > 5:
            insights.append("Multiple breakthrough moments indicate deep learning engagement")
        
        # Skill development insights
        skill_dev = progress_analysis["skill_development"]
        if len(skill_dev) > 3:
            insights.append("Diverse skill development across multiple domains")
        
        return insights
    
    async def _generate_quality_improvements(self, quality_analysis: Dict) -> List[str]:
        """Generate quality improvement suggestions"""
        
        improvements = []
        
        health_score = quality_analysis["memory_health_score"]
        
        if health_score < 0.5:
            improvements.append("Focus on creating more detailed and structured memories")
            improvements.append("Include specific examples and context in memories")
        
        factors = quality_analysis["quality_factors"]
        
        if factors["completeness"] < 0.5:
            improvements.append("Provide more complete information when creating memories")
        
        if factors["structure"] < 0.5:
            improvements.append("Use better formatting and organization in memory content")
        
        if factors["relevance"] < 0.5:
            improvements.append("Ensure memories are tagged with appropriate domains")
        
        return improvements
    
    async def _generate_best_practices(self, quality_analysis: Dict) -> List[str]:
        """Generate best practices for memory creation"""
        
        practices = [
            "Include specific details and context in memory content",
            "Use clear, structured formatting with proper punctuation",
            "Tag memories with relevant domains for better organization",
            "Regular review and update of important memories",
            "Connect new memories to existing knowledge"
        ]
        
        return practices
    
    async def _generate_pattern_insights(self, pattern_analysis: Dict) -> List[str]:
        """Generate insights from pattern analysis"""
        
        insights = []
        
        continuity = pattern_analysis["session_continuity"]
        if continuity > 0.6:
            insights.append("Excellent session continuity - building well on previous learning")
        elif continuity < 0.3:
            insights.append("Consider referencing previous sessions more explicitly")
        
        themes = pattern_analysis["recurring_themes"]
        if themes:
            top_theme = themes[0][0]
            insights.append(f"Primary learning focus: {top_theme}")
        
        return insights
    
    async def _generate_executive_summary(self, *analyses) -> Dict[str, Any]:
        """Generate executive summary from all analyses"""
        
        usage_analysis, gap_analysis, progress_analysis, quality_analysis, pattern_analysis = analyses
        
        summary = {
            "overall_health": "good",
            "key_strengths": [],
            "priority_areas": [],
            "learning_trajectory": progress_analysis.get("learning_trajectory", "steady"),
            "memory_efficiency": quality_analysis.get("memory_health_score", 0.5),
            "engagement_level": "moderate"
        }
        
        # Determine overall health
        health_indicators = [
            quality_analysis.get("memory_health_score", 0.5),
            min(1.0, progress_analysis.get("learning_velocity", 0) * 2),
            pattern_analysis.get("session_continuity", 0.5)
        ]
        
        avg_health = sum(health_indicators) / len(health_indicators)
        
        if avg_health > 0.7:
            summary["overall_health"] = "excellent"
        elif avg_health > 0.5:
            summary["overall_health"] = "good"
        else:
            summary["overall_health"] = "needs_improvement"
        
        # Identify key strengths
        if quality_analysis.get("memory_health_score", 0) > 0.6:
            summary["key_strengths"].append("High-quality memory creation")
        
        if progress_analysis.get("learning_velocity", 0) > 0.3:
            summary["key_strengths"].append("Strong learning velocity")
        
        if pattern_analysis.get("session_continuity", 0) > 0.5:
            summary["key_strengths"].append("Good session continuity")
        
        # Identify priority areas
        if len(gap_analysis.get("identified_gaps", [])) > 5:
            summary["priority_areas"].append("Address knowledge gaps")
        
        if quality_analysis.get("memory_health_score", 0) < 0.5:
            summary["priority_areas"].append("Improve memory quality")
        
        return summary
    
    async def _generate_unified_recommendations(self, analyses: Dict) -> Dict[str, List[str]]:
        """Generate unified recommendations across all analyses"""
        
        recommendations = {
            "immediate_actions": [],
            "short_term_goals": [],
            "long_term_strategies": []
        }
        
        # Immediate actions
        quality_score = analyses["memory_quality"].get("memory_health_score", 0.5)
        if quality_score < 0.5:
            recommendations["immediate_actions"].append("Focus on creating higher quality memories with more detail")
        
        gaps = analyses["knowledge_gaps"].get("identified_gaps", [])
        if len(gaps) > 3:
            recommendations["immediate_actions"].append("Create learning plan for top 3 knowledge gaps")
        
        # Short-term goals
        learning_velocity = analyses["learning_progress"].get("learning_velocity", 0)
        if learning_velocity < 0.2:
            recommendations["short_term_goals"].append("Increase learning session frequency")
        
        # Long-term strategies
        recommendations["long_term_strategies"].append("Maintain consistent learning rhythm")
        recommendations["long_term_strategies"].append("Regular memory quality reviews")
        
        return recommendations
    
    async def _create_health_dashboard(self, analyses: Dict) -> Dict[str, Any]:
        """Create memory health dashboard metrics"""
        
        dashboard = {
            "overall_score": 0.0,
            "category_scores": {
                "memory_quality": 0.0,
                "learning_progress": 0.0,
                "usage_patterns": 0.0,
                "knowledge_coverage": 0.0
            },
            "health_trends": "stable",
            "alerts": []
        }
        
        # Calculate category scores
        dashboard["category_scores"]["memory_quality"] = analyses["memory_quality"].get("memory_health_score", 0.5)
        dashboard["category_scores"]["learning_progress"] = min(1.0, analyses["learning_progress"].get("learning_velocity", 0) * 2)
        dashboard["category_scores"]["usage_patterns"] = 0.7  # Default moderate score
        
        gap_count = len(analyses["knowledge_gaps"].get("identified_gaps", []))
        dashboard["category_scores"]["knowledge_coverage"] = max(0.3, 1.0 - (gap_count * 0.1))
        
        # Calculate overall score
        dashboard["overall_score"] = sum(dashboard["category_scores"].values()) / len(dashboard["category_scores"])
        
        # Generate alerts
        if dashboard["category_scores"]["memory_quality"] < 0.4:
            dashboard["alerts"].append("Memory quality below optimal - consider improvement strategies")
        
        if gap_count > 10:
            dashboard["alerts"].append("High number of knowledge gaps identified - prioritize learning")
        
        return dashboard

async def demo_memory_analytics():
    """Demonstrate the memory analytics system"""
    
    print("="*70)
    print("📊 Memory Analytics Engine Demo")
    print("="*70)
    
    analytics = MemoryAnalyticsEngine()
    user_id = "analytics_user"
    
    try:
        # First, create some sample data for analysis
        print("\n🔧 Setting up sample data for analysis...")
        
        # Create sample memories for different scenarios
        await analytics.memory.create_episodic_memory(
            content="Had a great learning session about machine learning algorithms. Struggled with understanding backpropagation at first.",
            participants=[user_id, "instructor"],
            location="online_classroom",
            emotional_state="engaged",
            user_id=user_id
        )
        
        await analytics.memory.create_semantic_memory(
            content="Machine learning is a subset of artificial intelligence focused on algorithms that improve through experience",
            domain="computer_science",
            confidence=0.8,
            user_id=user_id
        )
        
        await analytics.memory.create_procedural_memory(
            content="To implement gradient descent: 1) Calculate cost function 2) Compute gradients 3) Update parameters 4) Repeat",
            skill_domain="machine_learning",
            proficiency_level="intermediate",
            user_id=user_id
        )
        
        await analytics.memory.create_social_memory(
            content="Instructor is very patient and explains complex concepts clearly. Prefers visual examples.",
            person_id="instructor",
            relationship_type="teacher",
            user_id=user_id
        )
        
        print("✅ Sample data created successfully!")
        
        # Demo 1: Usage Pattern Analysis
        print("\n🔍 Demo 1: Memory Usage Pattern Analysis")
        print("-" * 50)
        
        usage_analysis = await analytics.analyze_memory_usage_patterns(user_id, days_back=30)
        print(f"Total memories analyzed: {usage_analysis['total_memories']}")
        print(f"Memory type distribution: {dict(usage_analysis['memory_type_distribution'])}")
        print(f"Daily creation rate: {usage_analysis['temporal_patterns']['daily_creation_rate']:.2f}")
        print(f"Quality distribution: {usage_analysis['quality_distribution']}")
        print(f"Key insights: {usage_analysis['insights'][:2]}")  # Show first 2 insights
        
        # Demo 2: Knowledge Gap Analysis
        print("\n🧩 Demo 2: Knowledge Gap Analysis")
        print("-" * 40)
        
        gap_analysis = await analytics.analyze_knowledge_gaps(user_id, domain="computer_science")
        print(f"Total knowledge items: {gap_analysis['total_knowledge_items']}")
        print(f"Identified gaps: {gap_analysis['identified_gaps']}")
        print(f"Domain coverage: {gap_analysis['domain_coverage']}")
        print(f"Learning opportunities: {len(gap_analysis['learning_opportunities'])}")
        if gap_analysis['learning_opportunities']:
            print(f"Top opportunity: {gap_analysis['learning_opportunities'][0]['gap']}")
        
        # Demo 3: Learning Progress Analysis
        print("\n📈 Demo 3: Learning Progress Analysis")
        print("-" * 45)
        
        progress_analysis = await analytics.analyze_learning_progress(user_id, timeframe_days=30)
        print(f"Learning trajectory: {progress_analysis['learning_trajectory']}")
        print(f"Learning velocity: {progress_analysis['learning_velocity']:.3f} masteries/day")
        print(f"Mastery events: {len(progress_analysis['mastery_progression'])}")
        print(f"Breakthrough moments: {len(progress_analysis['breakthrough_moments'])}")
        print(f"Skill development: {progress_analysis['skill_development']}")
        
        # Demo 4: Memory Quality Analysis
        print("\n⭐ Demo 4: Memory Quality Analysis")
        print("-" * 40)
        
        quality_analysis = await analytics.analyze_memory_quality(user_id, sample_size=50)
        print(f"Memories analyzed: {quality_analysis['total_analyzed']}")
        print(f"Quality distribution: {quality_analysis['quality_distribution']}")
        print(f"Memory health score: {quality_analysis['memory_health_score']:.2f}")
        print(f"Quality factors: {quality_analysis['quality_factors']}")
        print(f"Improvement suggestions: {quality_analysis['improvement_suggestions'][:2]}")
        
        # Demo 5: Cross-Session Pattern Analysis
        print("\n🔄 Demo 5: Cross-Session Pattern Analysis")
        print("-" * 48)
        
        pattern_analysis = await analytics.analyze_cross_session_patterns(user_id)
        print(f"Total sessions: {pattern_analysis['total_sessions']}")
        print(f"Session continuity: {pattern_analysis['session_continuity']:.2f}")
        print(f"Recurring themes: {pattern_analysis['recurring_themes'][:3]}")
        print(f"Pattern insights: {pattern_analysis['insights']}")
        
        # Demo 6: Comprehensive Report
        print("\n📊 Demo 6: Comprehensive Analytics Report")
        print("-" * 52)
        
        comprehensive_report = await analytics.generate_comprehensive_report(user_id)
        
        executive_summary = comprehensive_report["executive_summary"]
        print(f"Overall health: {executive_summary['overall_health']}")
        print(f"Learning trajectory: {executive_summary['learning_trajectory']}")
        print(f"Memory efficiency: {executive_summary['memory_efficiency']:.2f}")
        print(f"Key strengths: {executive_summary['key_strengths']}")
        print(f"Priority areas: {executive_summary['priority_areas']}")
        
        # Health Dashboard
        health_dashboard = comprehensive_report["memory_health_dashboard"]
        print(f"\n🏥 Memory Health Dashboard:")
        print(f"Overall score: {health_dashboard['overall_score']:.2f}")
        print(f"Category scores:")
        for category, score in health_dashboard["category_scores"].items():
            print(f"  {category}: {score:.2f}")
        
        if health_dashboard["alerts"]:
            print(f"Alerts: {health_dashboard['alerts']}")
        
        # Recommendations
        recommendations = comprehensive_report["recommendations"]
        print(f"\n💡 Unified Recommendations:")
        print(f"Immediate actions: {recommendations['immediate_actions']}")
        print(f"Short-term goals: {recommendations['short_term_goals']}")
        print(f"Long-term strategies: {recommendations['long_term_strategies']}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ Memory Analytics Demo Complete!")
    print("="*70)
    print("Key features demonstrated:")
    print("• Comprehensive usage pattern analysis")
    print("• Knowledge gap identification and recommendations")
    print("• Learning progress tracking and insights")
    print("• Memory quality assessment and improvement")
    print("• Cross-session pattern recognition")
    print("• Health dashboard and unified reporting")

if __name__ == "__main__":
    asyncio.run(demo_memory_analytics())

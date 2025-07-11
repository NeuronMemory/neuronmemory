"""
Advanced Retrieval Engine (ARE) for NeuronMemory

This module implements intelligent memory retrieval with multi-modal search,
contextual ranking, and relevance optimization.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from dataclasses import dataclass

from ..memory.memory_objects import BaseMemory, MemoryType, EmotionalState

from ..llm.azure_openai_client import AzureOpenAIClient
from ..config import neuron_memory_config

logger = logging.getLogger(__name__)

class SearchStrategy(str, Enum):
    """Search strategy types"""
    SEMANTIC_ONLY = "semantic_only"
    TEMPORAL_WEIGHTED = "temporal_weighted" 
    EMOTIONAL_FILTERED = "emotional_filtered"
    SOCIAL_CONTEXT = "social_context"
    HYBRID_MULTI_MODAL = "hybrid_multi_modal"

@dataclass
class SearchContext:
    """Context for memory search operations"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    current_task: Optional[str] = None
    emotional_state: Optional[EmotionalState] = None
    time_context: Optional[datetime] = None
    social_context: Optional[List[str]] = None
    domain_focus: Optional[str] = None
    urgency_level: float = 0.5

@dataclass
class RetrievalResult:
    """Result from memory retrieval"""
    memory: BaseMemory
    relevance_score: float
    similarity_score: float
    temporal_score: float
    importance_score: float
    context_score: float
    final_score: float
    explanation: str

class RetrievalEngine:
    """
    Advanced Retrieval Engine (ARE) for intelligent memory search
    
    Features:
    - Multi-modal search (semantic + temporal + emotional + social)
    - Context-aware ranking and relevance scoring
    - Adaptive search strategies based on query type
    - Diversity-aware result selection
    - Performance optimization with caching
    """
    
    def __init__(self):
        """Initialize the retrieval engine"""
        self.llm_client = AzureOpenAIClient()
        self.config = neuron_memory_config
        
        # Scoring weights for different factors
        self.scoring_weights = {
            "similarity": 0.3,
            "temporal": 0.2,
            "importance": 0.25,
            "context": 0.15,
            "emotional": 0.1
        }
        
        # Cache for recent searches
        self._search_cache: Dict[str, Tuple[List[RetrievalResult], datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)
        
    async def search(
        self,
        query: str,
        memory_store,
        context: SearchContext,
        strategy: SearchStrategy = SearchStrategy.HYBRID_MULTI_MODAL,
        limit: int = 10,
        similarity_threshold: float = 0.1,
        diversity_factor: float = 0.3
    ) -> List[RetrievalResult]:
        """
        Perform intelligent memory search
        
        Args:
            query: Search query text
            context: Search context information
            strategy: Search strategy to use
            limit: Maximum number of results
            similarity_threshold: Minimum similarity threshold
            diversity_factor: Factor for result diversification (0.0-1.0)
            
        Returns:
            List of ranked retrieval results
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, context, strategy)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result[:limit]
            
            # Determine memory types to search based on strategy and context
            memory_types = self._determine_search_scope(strategy, context)
            
            # Perform initial semantic search
            search_results = await memory_store.search_memories(
                query=query,
                memory_types=memory_types,
                limit=min(limit * 3, 50),  # Get more results for better ranking
                similarity_threshold=similarity_threshold
            )
            
            if not search_results:
                return []
            
            # Convert to retrieval results with detailed scoring
            retrieval_results = []
            for memory, similarity in search_results:
                result = await self._create_retrieval_result(
                    memory, similarity, query, context, strategy
                )
                retrieval_results.append(result)
            
            # Apply advanced ranking
            ranked_results = await self._rank_results(
                retrieval_results, query, context, strategy
            )
            
            # Apply diversity filtering if requested
            if diversity_factor > 0:
                ranked_results = await self._apply_diversity_filter(
                    ranked_results, diversity_factor
                )
            
            # Limit results
            final_results = ranked_results[:limit]
            
            # Cache the results
            self._cache_result(cache_key, ranked_results)
            
            logger.debug(f"Retrieved {len(final_results)} memories for query: {query}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in memory search: {e}")
            return []
    
    async def find_related(
        self,
        memory_id: str,
        memory_store,
        relationship_types: Optional[List[str]] = None,
        context: Optional[SearchContext] = None,
        limit: int = 5
    ) -> List[RetrievalResult]:
        """
        Find memories related to a specific memory
        
        Args:
            memory_id: ID of the source memory
            relationship_types: Types of relationships to consider
            context: Search context
            limit: Maximum number of results
            
        Returns:
            List of related memories
        """
        try:
            # Get the source memory  
            source_memory = await memory_store.retrieve_memory(memory_id)
            if not source_memory:
                return []
            
            # Use source memory content as query for finding related memories
            query = source_memory.content
            
            # Create search context if not provided
            if context is None:
                context = SearchContext(
                    user_id=source_memory.metadata.user_id,
                    session_id=source_memory.metadata.session_id
                )
            
            # Search for related memories
            results = await self.search(
                query=query,
                memory_store=memory_store,
                context=context,
                strategy=SearchStrategy.SEMANTIC_ONLY,
                limit=limit + 1,  # +1 because source memory might be included
                similarity_threshold=0.3
            )
            
            # Filter out the source memory itself
            related_results = [r for r in results if r.memory.memory_id != memory_id]
            
            return related_results[:limit]
            
        except Exception as e:
            logger.error(f"Error finding related memories: {e}")
            return []
    
    async def get_context_memories(
        self,
        context: SearchContext,
        limit: int = 5
    ) -> List[RetrievalResult]:
        """
        Get relevant memories based on current context
        
        Args:
            context: Current context information
            limit: Maximum number of results
            
        Returns:
            List of contextually relevant memories
        """
        try:
            # Build query from context
            query_parts = []
            
            if context.current_task:
                query_parts.append(context.current_task)
            
            if context.domain_focus:
                query_parts.append(context.domain_focus)
            
            if context.social_context:
                query_parts.extend(context.social_context)
            
            if not query_parts:
                # Fallback to recent memories
                return await self._get_recent_memories(context, limit)
            
            query = " ".join(query_parts)
            
            results = await self.search(
                query=query,
                context=context,
                limit=limit
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting context memories: {e}")
            return []
    
    async def _create_retrieval_result(
        self,
        memory: BaseMemory,
        similarity_score: float,
        query: str,
        context: SearchContext,
        strategy: SearchStrategy
    ) -> RetrievalResult:
        """Create a detailed retrieval result object with all scores"""
        
        # Temporal scoring
            temporal_score = self._calculate_temporal_score(memory, context)
            
        # Importance scoring
            importance_score = memory.metadata.importance_score
            
        # Contextual scoring
            context_score = await self._calculate_context_score(memory, context)
            
        # Emotional scoring (if applicable)
        emotional_score = 0.0
        if strategy == SearchStrategy.EMOTIONAL_FILTERED and context.emotional_state and memory.emotional_state:
            emotional_score = self._calculate_emotional_similarity(
                context.emotional_state, memory.emotional_state
            )
        
        # Final weighted score
            final_score = (
            similarity_score * self.scoring_weights["similarity"] +
            temporal_score * self.scoring_weights["temporal"] +
            importance_score * self.scoring_weights["importance"] +
            context_score * self.scoring_weights["context"] +
            emotional_score * self.scoring_weights["emotional"]
            )
            
            # Generate explanation
            explanation = self._generate_explanation(
                similarity_score, temporal_score, importance_score, context_score
            )
            
            return RetrievalResult(
                memory=memory,
                relevance_score=final_score,
                similarity_score=similarity_score,
                temporal_score=temporal_score,
                importance_score=importance_score,
                context_score=context_score,
                final_score=final_score,
                explanation=explanation
            )
    
    def _calculate_temporal_score(self, memory: BaseMemory, context: SearchContext) -> float:
        """
        Calculate a temporal relevance score (0.0 to 1.0)
        
        Recency is key, but access patterns also matter.
        """
        now = context.time_context or datetime.utcnow()
        last_access_hours = (now - memory.metadata.last_accessed).total_seconds() / 3600
        
        # Sigmoid function for smooth decay
        # Recent memories get a high score, older ones decay gracefully
        score = 1 / (1 + (last_access_hours / 24)**2) # 50% score after 24 hours
        
        # Boost for frequently accessed memories
        access_boost = min(0.2, (memory.metadata.access_count / 100)) # Up to 20% boost
        
        return min(1.0, score + access_boost)
    
    async def _calculate_context_score(self, memory: BaseMemory, context: SearchContext) -> float:
        """
        Calculate a contextual relevance score based on tags, domain, user, etc.
        """
            score = 0.0
            
        # User ID match (strong signal)
            if context.user_id and memory.metadata.user_id == context.user_id:
            score += 0.4
            
        # Session ID match (very strong signal)
            if context.session_id and memory.metadata.session_id == context.session_id:
            score += 0.4
            
        # Context tag overlap
        if context.current_task or context.domain_focus:
            context_text = f"{context.current_task or ''} {context.domain_focus or ''}"
            
            # Use LLM for a more nuanced comparison
            relevance_check = await self.llm_client.compare_relevance(
                text1=context_text,
                text2=" ".join(memory.metadata.context_tags)
                )
            score += 0.2 * relevance_check
            
            return min(1.0, score)
    
    def _calculate_emotional_similarity(self, emotion1: EmotionalState, emotion2: EmotionalState) -> float:
        """
        Calculate emotional similarity using cosine similarity on VAD space
        """
        v1 = np.array([emotion1.valence, emotion1.arousal, emotion1.dominance])
        v2 = np.array([emotion2.valence, emotion2.arousal, emotion2.dominance])
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        return np.dot(v1, v2) / (norm_v1 * norm_v2)
    
    async def _rank_results(
        self,
        results: List[RetrievalResult],
        query: str,
        context: SearchContext,
        strategy: SearchStrategy
    ) -> List[RetrievalResult]:
        """
        Re-rank results using a more sophisticated model or logic
        
        For this implementation, we'll sort by the final_score.
        In a more advanced system, this could involve an LLM re-ranker.
        """
        # Here we could use an LLM to re-rank the top N results
        # For now, we'll just sort by the calculated final_score
        
        # Example of a potential re-ranking prompt:
        # re_rank_prompt = f"""
        # Re-rank the following search results for the query '{query}'
        # considering the user context: {context}.
        # 
        # Results:
        # {json.dumps([r.memory.model_dump_json() for r in results[:5]], indent=2)}
        #
        # Return a JSON list of memory_ids in the new optimal order.
        # """
        
        results.sort(key=lambda r: r.final_score, reverse=True)
            return results
    
    async def _apply_diversity_filter(
        self, 
        results: List[RetrievalResult], 
        diversity_factor: float
    ) -> List[RetrievalResult]:
        """
        Apply Maximal Marginal Relevance (MMR) for diversity
        """
        if not results:
                return []
            
        # Normalize diversity factor
        lambda_param = max(0.0, min(1.0, diversity_factor))
        
        selected_results: List[RetrievalResult] = []
        remaining_results = results[:]
        
        if remaining_results:
            # First result is always the most relevant one
            selected_results.append(remaining_results.pop(0))
            
        while remaining_results:
            best_next_result = None
            max_mmr_score = -np.inf
            
            for result in remaining_results:
                similarity_to_selected = max(
                    [
                        np.dot(result.memory.embedding, sel.memory.embedding)
                        for sel in selected_results
                    ]
                ) if selected_results else 0.0
                
                mmr_score = (
                    lambda_param * result.relevance_score - 
                    (1 - lambda_param) * similarity_to_selected
                )
                
                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    best_next_result = result
            
            if best_next_result:
                selected_results.append(best_next_result)
                remaining_results.remove(best_next_result)
            else:
                break # No more results to add
                
        return selected_results
    
    def _determine_search_scope(self, strategy: SearchStrategy, context: SearchContext) -> Optional[List[MemoryType]]:
        """Determine which memory types to search based on strategy"""
        if strategy == SearchStrategy.SOCIAL_CONTEXT:
            return [MemoryType.SOCIAL, MemoryType.EPISODIC]
        if strategy == SearchStrategy.EMOTIONAL_FILTERED:
            return [MemoryType.EPISODIC]
        # Default to all long-term memory types
        return [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL, MemoryType.SOCIAL]
    
    def _generate_explanation(
        self, 
        similarity_score: float, 
        temporal_score: float, 
        importance_score: float, 
        context_score: float
    ) -> str:
        """
        Generate a human-readable explanation for the ranking
        """
        explanation_parts = []
        
        if similarity_score > 0.5:
            explanation_parts.append(f"high semantic similarity ({similarity_score:.2f})")
        
        if temporal_score > 0.7:
            explanation_parts.append(f"recent activity ({temporal_score:.2f})")
        
        if importance_score > 0.6:
            explanation_parts.append(f"high importance ({importance_score:.2f})")
        
        if context_score > 0.5:
            explanation_parts.append(f"strong contextual match ({context_score:.2f})")
        
        if not explanation_parts:
            return "General relevance match"
        
        return "Retrieved due to " + ", ".join(explanation_parts)
    
    async def _get_recent_memories(self, context: SearchContext, limit: int) -> List[RetrievalResult]:
        """Fallback to retrieve the most recently accessed memories"""
        # This requires a way to query by `last_accessed` which is not directly supported
        # by ChromaDB metadata filtering in the same way as `search`.
        # This is a placeholder for a more complex implementation.
        logger.warning("Fallback to _get_recent_memories is not fully implemented.")
            return []
    
    def _generate_cache_key(self, query: str, context: SearchContext, strategy: SearchStrategy) -> str:
        """Generate a cache key for a search query"""
        context_str = (
            f"{context.user_id}-{context.session_id}-{context.current_task}-"
            f"{context.domain_focus}"
        )
        return f"{query}|{strategy.value}|{context_str}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[RetrievalResult]]:
        """Get result from cache if available and not expired"""
        if cache_key in self._search_cache:
            results, timestamp = self._search_cache[cache_key]
            if datetime.utcnow() - timestamp < self._cache_ttl:
                logger.debug(f"Search cache hit for key: {cache_key}")
                return results
        return None
    
    def _cache_result(self, cache_key: str, results: List[RetrievalResult]):
        """Cache a search result"""
        self._search_cache[cache_key] = (results, datetime.utcnow())
        logger.debug(f"Cached search result for key: {cache_key}")
        
        # Evict old cache entries if cache is too large
        if len(self._search_cache) > self.config.retrieval_cache_size:
            oldest_key = min(
                self._search_cache, key=lambda k: self._search_cache[k][1]
            )
            del self._search_cache[oldest_key] 
            logger.debug(f"Evicted oldest cache entry: {oldest_key}") 
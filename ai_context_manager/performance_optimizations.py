"""
Performance optimizations for the AI Context Manager
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Performance optimization utilities for context management."""
    
    def __init__(self):
        self._token_cache = {}
        self._score_cache = {}
        self._last_cache_cleanup = datetime.now()
        self._cache_lock = threading.Lock()
        
    @lru_cache(maxsize=1000)
    def cached_token_estimation(self, text: str) -> int:
        """Cached token estimation to avoid repeated calculations."""
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-4")
            return len(enc.encode(text))
        except ImportError:
            return len(text.split())
    
    def batch_token_estimation(self, texts: List[str]) -> List[int]:
        """Estimate tokens for multiple texts efficiently."""
        return [self.cached_token_estimation(text) for text in texts]
    
    def cleanup_caches(self, max_age_hours: int = 1):
        """Clean up old cache entries."""
        now = datetime.now()
        if (now - self._last_cache_cleanup).total_seconds() > max_age_hours * 3600:
            with self._cache_lock:
                # Clear old entries (simplified - in production, use TTL)
                if len(self._token_cache) > 1000:
                    self._token_cache.clear()
                if len(self._score_cache) > 500:
                    self._score_cache.clear()
                self._last_cache_cleanup = now


class AsyncContextManager:
    """Async version of context manager for better performance."""
    
    def __init__(self, base_context_manager):
        self.ctx = base_context_manager
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def register_component_async(self, component):
        """Register component asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.ctx.register_component, 
            component
        )
    
    async def get_context_async(self, **kwargs) -> str:
        """Get context asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.ctx.get_context,
            **kwargs
        )
    
    async def batch_register_components(self, components: List):
        """Register multiple components in parallel."""
        tasks = [self.register_component_async(comp) for comp in components]
        return await asyncio.gather(*tasks)


class SmartCaching:
    """Intelligent caching system for frequently accessed data."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()
    
    def get(self, key: str, default=None):
        """Get cached value with TTL check."""
        with self.lock:
            if key in self.cache:
                if (datetime.now() - self.access_times[key]).total_seconds() < self.ttl_seconds:
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.access_times[key]
            return default
    
    def set(self, key: str, value):
        """Set cached value with eviction policy."""
        with self.lock:
            # Simple LRU eviction
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
            self.access_times[key] = datetime.now()
    
    def clear(self):
        """Clear all cached data."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()


class ComponentIndex:
    """Fast indexing system for component retrieval."""
    
    def __init__(self):
        self.by_type: Dict[str, Set[str]] = {}
        self.by_tag: Dict[str, Set[str]] = {}
        self.by_score_range: Dict[tuple, Set[str]] = {}
        self.score_ranges = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float('inf'))]
        
    def add_component(self, component_id: str, component_type: str, tags: List[str], score: float):
        """Add component to indices."""
        # Type index
        if component_type not in self.by_type:
            self.by_type[component_type] = set()
        self.by_type[component_type].add(component_id)
        
        # Tag index
        for tag in tags:
            if tag not in self.by_tag:
                self.by_tag[tag] = set()
            self.by_tag[tag].add(component_id)
        
        # Score range index
        for min_score, max_score in self.score_ranges:
            if min_score <= score < max_score:
                range_key = (min_score, max_score)
                if range_key not in self.by_score_range:
                    self.by_score_range[range_key] = set()
                self.by_score_range[range_key].add(component_id)
                break
    
    def remove_component(self, component_id: str, component_type: str, tags: List[str], score: float):
        """Remove component from indices."""
        # Type index
        if component_type in self.by_type:
            self.by_type[component_type].discard(component_id)
        
        # Tag index
        for tag in tags:
            if tag in self.by_tag:
                self.by_tag[tag].discard(component_id)
        
        # Score range index
        for min_score, max_score in self.score_ranges:
            if min_score <= score < max_score:
                range_key = (min_score, max_score)
                if range_key in self.by_score_range:
                    self.by_score_range[range_key].discard(component_id)
                break
    
    def get_by_type(self, component_type: str) -> Set[str]:
        """Get component IDs by type."""
        return self.by_type.get(component_type, set())
    
    def get_by_tag(self, tag: str) -> Set[str]:
        """Get component IDs by tag."""
        return self.by_tag.get(tag, set())
    
    def get_by_score_range(self, min_score: float, max_score: float) -> Set[str]:
        """Get component IDs by score range."""
        for range_min, range_max in self.score_ranges:
            if range_min <= min_score and max_score <= range_max:
                return self.by_score_range.get((range_min, range_max), set())
        return set()
    
    def get_intersection(self, type_filter: Optional[str] = None, 
                        tag_filters: Optional[List[str]] = None,
                        score_range: Optional[tuple] = None) -> Set[str]:
        """Get intersection of multiple filters."""
        result = None
        
        if type_filter:
            type_set = self.get_by_type(type_filter)
            result = type_set if result is None else result.intersection(type_set)
        
        if tag_filters:
            for tag in tag_filters:
                tag_set = self.get_by_tag(tag)
                result = tag_set if result is None else result.intersection(tag_set)
        
        if score_range:
            score_set = self.get_by_score_range(*score_range)
            result = score_set if result is None else result.intersection(score_set)
        
        return result or set()

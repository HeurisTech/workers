"""LLM Factory with instance caching to prevent API overload."""

from __future__ import annotations

import hashlib
import threading
from typing import Dict, Tuple, Any
from functools import lru_cache

from langchain_core.language_models import BaseChatModel


class LLMFactory:
    """Factory for creating and caching LLM instances to prevent API overload."""
    
    def __init__(self):
        self._cache: Dict[str, BaseChatModel] = {}
        self._lock = threading.Lock()
    
    def _get_cache_key(self, provider: str, model: str, api_key: str, **kwargs) -> str:
        """Generate a cache key based on provider, model, and API key hash."""
        # Hash the API key to avoid storing it in memory
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        
        # Include relevant kwargs in the cache key
        relevant_kwargs = {k: v for k, v in kwargs.items() if k in ['temperature', 'max_tokens']}
        kwargs_str = str(sorted(relevant_kwargs.items()))
        
        return f"{provider}:{model}:{api_key_hash}:{kwargs_str}"
    
    def get_llm(
        self, 
        provider: str, 
        model: str, 
        api_key: str, 
        temperature: float = 1.0,
        **kwargs
    ) -> BaseChatModel:
        """Get or create a cached LLM instance.
        
        Args:
            provider: Either 'anthropic' or 'openai'
            model: Model name (e.g., 'claude-3-5-sonnet-20241022', 'gpt-4o')
            api_key: API key for the provider
            temperature: Temperature setting for the model
            **kwargs: Additional arguments passed to the LLM constructor
            
        Returns:
            Cached or newly created LLM instance
        """
        cache_key = self._get_cache_key(provider, model, api_key, temperature=temperature, **kwargs)
        
        # Check if we already have this instance cached
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # Create new instance
            if provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                
                llm = ChatAnthropic(
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    **kwargs
                )
            elif provider == "openai":
                from langchain_openai import ChatOpenAI
                
                llm = ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Cache the instance
            self._cache[cache_key] = llm
            print(f"Created new {provider} LLM instance: {model} (cache size: {len(self._cache)})")
            
            return llm
    
    def get_structured_llm(
        self,
        provider: str,
        model: str,
        api_key: str,
        schema: Any,
        temperature: float = 1.0,
        method: str = "function_calling",
        **kwargs
    ) -> BaseChatModel:
        """Get a structured output LLM instance.
        
        Args:
            provider: Either 'anthropic' or 'openai'
            model: Model name
            api_key: API key for the provider
            schema: Pydantic schema for structured output
            temperature: Temperature setting
            method: Method for structured output ('function_calling' or 'json_mode')
            **kwargs: Additional arguments
            
        Returns:
            LLM instance with structured output configured
        """
        # Get the base LLM instance (cached)
        base_llm = self.get_llm(provider, model, api_key, temperature, **kwargs)
        
        # Create structured output wrapper
        # Note: We don't cache this because structured output wrappers are lightweight
        structured_llm = base_llm.with_structured_output(schema, method=method)
        
        return structured_llm
    
    def clear_cache(self) -> None:
        """Clear all cached LLM instances."""
        with self._lock:
            self._cache.clear()
            print("LLM cache cleared")
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get information about the current cache state."""
        with self._lock:
            return {
                "cached_instances": len(self._cache),
                "anthropic_instances": len([k for k in self._cache.keys() if k.startswith("anthropic:")]),
                "openai_instances": len([k for k in self._cache.keys() if k.startswith("openai:")]),
            }


# Global instance
_llm_factory = LLMFactory()


def get_llm_factory() -> LLMFactory:
    """Get the global LLM factory instance."""
    return _llm_factory


# Convenience functions for common use cases
def get_cached_llm(provider: str, model: str, api_key: str, **kwargs) -> BaseChatModel:
    """Get a cached LLM instance."""
    return _llm_factory.get_llm(provider, model, api_key, **kwargs)


def get_cached_structured_llm(provider: str, model: str, api_key: str, schema: Any, **kwargs) -> BaseChatModel:
    """Get a cached structured LLM instance."""
    return _llm_factory.get_structured_llm(provider, model, api_key, schema, **kwargs)


@lru_cache(maxsize=10)
def get_llm_for_config(provider: str, model: str, api_key_hash: str) -> BaseChatModel:
    """LRU cached LLM getter (alternative approach)."""
    # This is kept for potential future use with different caching strategies
    pass 
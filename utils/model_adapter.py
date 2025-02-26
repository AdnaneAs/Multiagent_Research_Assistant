from typing import Optional, Union, Dict, Any
import logging
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

logger = logging.getLogger(__name__)

class ModelAdapter:
    """
    Model adapter class to handle different LLM providers.
    Supports OpenAI and Ollama models.
    """
    
    @staticmethod
    def get_llm(provider: str, model_id: str, api_key: Optional[str] = None, temperature: float = 0.2) -> Any:
        """
        Get an LLM instance based on the provider and model ID.
        
        Args:
            provider: The model provider ('openai' or 'ollama')
            model_id: The specific model ID
            api_key: API key for OpenAI models
            temperature: Temperature setting for the model
            
        Returns:
            LLM instance
        """
        if provider == "openai":
            if not api_key:
                raise ValueError("API key is required for OpenAI models")
            
            return ChatOpenAI(
                model=model_id,
                temperature=temperature,
                api_key=api_key
            )
        elif provider == "ollama":
            return Ollama(
                model=model_id,
                temperature=temperature
            )
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

def get_llm_instance(provider: str = "openai", 
                    model_id: str = "gpt-3.5-turbo", 
                    api_key: Optional[str] = None) -> Union[ChatOpenAI, ChatOllama]:
    """
    Get a language model instance based on the provider and model ID.
    
    Args:
        provider: The model provider ('openai' or 'ollama')
        model_id: The model ID to use
        api_key: API key for OpenAI models (not needed for Ollama)
    
    Returns:
        A language model instance compatible with LangChain
    """
    logger.info(f"Creating LLM instance for provider: {provider}, model: {model_id}")
    
    try:
        if provider.lower() == "openai":
            if not api_key:
                raise ValueError("API key is required for OpenAI models")
            
            return ChatOpenAI(
                model=model_id,
                openai_api_key=api_key,
                temperature=0.7,
                max_tokens=1500
            )
        
        elif provider.lower() == "ollama":
            return ChatOllama(
                model=model_id,
                temperature=0.7,
                # Ollama is run locally, so we use localhost
                base_url="http://localhost:11434"
            )
        
        else:
            raise ValueError(f"Unsupported model provider: {provider}")
            
    except Exception as e:
        logger.error(f"Error creating LLM instance: {e}")
        raise
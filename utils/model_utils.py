import requests
from typing import List, Dict, Any

def get_available_ollama_models() -> List[Dict[str, Any]]:
    """
    Get a list of locally available Ollama models.
    
    Returns:
        List of dictionaries containing model information
    """
    try:
        # Try to connect to Ollama API at the default URL
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            return response.json().get("models", [])
        return []
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return []

def get_openai_model_options() -> List[Dict[str, str]]:
    """
    Get a list of available OpenAI models.
    
    Returns:
        List of dictionaries containing model information
    """
    return [
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "openai"},
        {"id": "gpt-4", "name": "GPT-4", "provider": "openai"},
        {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "provider": "openai"}
    ]
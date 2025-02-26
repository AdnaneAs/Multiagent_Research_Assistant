import logging
from typing import Dict, List, Any, Optional
import os
from dotenv import load_dotenv
from utils.model_adapter import get_llm_instance

logger = logging.getLogger("PlanningAgent")

# Load environment variables from .env file
load_dotenv()

class PlanningAgent:
    def __init__(self, provider: str = "openai", model_id: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize the planning agent with the specified model.
        
        Args:
            provider: The model provider ('openai' or 'ollama')
            model_id: The model ID to use
            api_key: API key for OpenAI models (not needed for Ollama)
        """
        logger.info(f"Initializing PlanningAgent with {provider} model: {model_id}")
        
        # Use API key from environment if not provided
        if provider == "openai" and not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            
        # Initialize the language model for planning
        self.llm = get_llm_instance(provider, model_id, api_key)
        
    def generate_plan(self, topic_keywords: str) -> Dict[str, Any]:
        
        logger.info(f"Generating research plan for topic: {topic_keywords}")
        
        prompt = f"""
        I need to research the topic described by these keywords: '{topic_keywords}'.
        
        Please provide:
        1. A clear breakdown of subtopics to explore
        2. At least 5 specific search queries that would help gather comprehensive information
        3. A short description of the expected outcome of this research
        
        Format your response as a JSON with the following structure:
        {{
            "subtopics": ["subtopic1", "subtopic2", ...],
            "search_queries": ["query1", "query2", ...],
            "expected_outcome": "description of expected outcome",
            "research_strategy": "brief description of research strategy"
        }}
        """
        logger.info("Sending prompt to LLM...")
        
        try:
            response = self.llm.invoke(prompt)
            logger.info("Received response from LLM")
            logger.debug(f"Raw LLM response: {response}")
            
            # Process the response
            try:
                # Extract content from response (format depends on LLM output)
                content = response.content if hasattr(response, 'content') else response
                # For simplicity in this example, we're returning a dict directly
                # In a real implementation, you would parse the JSON response
                
                # This is a placeholder - you would need to parse the actual JSON
                import json
                try:
                    plan = json.loads(content)
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    plan = {
                        "subtopics": ["General " + topic_keywords],
                        "search_queries": [topic_keywords, "latest research " + topic_keywords],
                        "expected_outcome": f"General overview of {topic_keywords}",
                        "research_strategy": "General search on the topic"
                    }
                
                logger.info("Plan generated successfully")
                logger.info(f"Plan details: {plan}")
                return {
                    "topic": topic_keywords,
                    "plan": plan
                }
            except Exception as e:
                logger.error(f"Error generating plan: {e}")
                # Return basic plan as fallback
                return {
                    "topic": topic_keywords,
                    "plan": {
                        "subtopics": ["General research"],
                        "search_queries": [topic_keywords],
                        "expected_outcome": "Basic information",
                        "research_strategy": "General search"
                    }
                }
        except Exception as e:
            logger.error(f"Error generating plan: {e}")
            raise

    def _extract_search_terms(self, objectives: str) -> list:
        """Extract key search terms from objectives"""
        logger.info("Extracting search terms from objectives")
        prompt = f"""
        Extract key search terms from these research objectives:
        {objectives}
        Return only the most relevant terms for academic/technical search.
        """
        
        try:
            terms = self.llm.invoke(prompt)
            search_terms = [term.strip() for term in terms.split('\n') if term.strip()]
            logger.info(f"Extracted search terms: {search_terms}")
            return search_terms
        except Exception as e:
            logger.error(f"Error extracting search terms: {e}")
            raise
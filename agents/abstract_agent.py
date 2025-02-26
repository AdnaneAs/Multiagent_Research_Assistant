import logging
from typing import Dict, List, Any, Optional
import os
from dotenv import load_dotenv
from utils.model_adapter import get_llm_instance

# Configure logging
logger = logging.getLogger("AbstractAgent")

# Load environment variables from .env file
load_dotenv()

class AbstractAgent:
    def __init__(self, provider: str = "openai", model_id: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize the abstract agent for summarizing article content
        
        Args:
            provider: The model provider ('openai' or 'ollama')
            model_id: The model ID to use
            api_key: API key for OpenAI models (not needed for Ollama)
        """
        logger.info(f"Initializing AbstractAgent with {provider} model: {model_id}")
        
        # Use API key from environment if not provided
        if provider == "openai" and not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            
        # Initialize the language model for summarization
        self.llm = get_llm_instance(
            provider=provider,
            model_id=model_id,
            api_key=api_key
        )
        logger.info("AbstractAgent initialized successfully")
        
    def generate_abstract(self, article_content: str, article_title: str = "", max_words: int = 200) -> str:
        """
        Generate an abstract/summary for an article
        
        Args:
            article_content: The content of the article
            article_title: The title of the article
            max_words: Maximum length of the summary in words
            
        Returns:
            A concise summary of the article
        """
        logger.info("Generating abstract...")
        
        # Truncate content if it's too long to fit in the context window
        # This is a simple approach - a more sophisticated one would use a smarter chunking strategy
        max_content_length = 10000  # Approximate token limit
        if len(article_content) > max_content_length:
            article_content = article_content[:max_content_length] + "..."
            
        # Create prompt for summarization
        prompt = f"""
        Article Title: {article_title}
        
        Article Content:
        {article_content}
        
        Please provide a concise academic abstract of the above article content in no more than {max_words} words.
        Focus on the main findings, methodology, and implications.
        The abstract should be informative and self-contained, allowing readers to quickly understand 
        the key points of the article without reading the full text.
        
        Abstract:
        """
        
        try:
            logger.info("Sending prompt to LLM...")
            # Generate summary using the language model
            response = self.llm.invoke(prompt)
            
            # Extract and return the abstract
            abstract = response.content if hasattr(response, 'content') else response
            logger.info(f"Abstract generated ({len(abstract)} characters)")
            logger.debug(f"Generated abstract: {abstract}")
            return abstract.strip()
            
        except Exception as e:
            logger.error(f"Error generating abstract: {e}")
            return f"Error generating abstract: {e}"
            
    def process_article_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read an article file and generate an abstract
        
        Args:
            file_path: Path to the article file
            
        Returns:
            Dictionary with the article path and its abstract
        """
        logger.info(f"Processing article file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            logger.info(f"Article content read successfully ({len(content)} characters)")
            logger.debug(f"Article content preview: {content[:200]}...")
            
            # Extract title from the file content
            title = ""
            lines = content.split('\n')
            for line in lines:
                if line.startswith("Title:"):
                    title = line.replace("Title:", "").strip()
                    break
                    
            # Get the actual article content (skip the header with title and URL)
            article_content = "\n".join(lines[3:]) if len(lines) >= 3 else content
            
            # Generate abstract
            abstract = self.generate_abstract(article_content, title)
            
            return {
                "file_path": file_path,
                "title": title,
                "abstract": abstract
            }
            
        except Exception as e:
            logger.error(f"Error processing article file {file_path}: {e}")
            return {
                "file_path": file_path,
                "title": "",
                "abstract": f"Error processing article: {e}",
                "error": str(e)
            }
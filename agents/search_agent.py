import os
import logging
from typing import Dict, List, Any
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger("SearchAgent")

class SearchAgent:
    def __init__(self):
        """Initialize the search agent with DuckDuckGo"""
        logger.info("Initializing SearchAgent with DuckDuckGo")
        self.search_engine = DDGS()
        self.max_results = 10
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
        ]
        logger.info("SearchAgent initialized successfully")
        
    def get_random_user_agent(self):
        return random.choice(self.user_agents)
        
    def is_valid_arxiv_url(self, url: str) -> bool:
        """Check if URL is from arxiv.org"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.endswith('arxiv.org')
        except:
            return False

    def search_articles(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search for articles based on the search queries from the plan
        
        Args:
            plan: The research plan with search queries
            
        Returns:
            List of dictionaries containing article information
        """
        logger.info("Starting article search...")
        logger.info(f"Search plan: {plan}")
        
        results = []
        search_queries = plan["plan"]["search_queries"]
        
        for query in search_queries:
            logger.info(f"Executing search query: {query}")
            try:
                # Focus specifically on arXiv results
                enhanced_query = f"site:arxiv.org {query}"
                
                # Get search results
                search_results = self.search_engine.text(
                    enhanced_query,
                    max_results=self.max_results // len(search_queries)  # Distribute results across queries
                )
                
                for result in search_results:
                    url = result.get("href", "")
                    # Only process arXiv URLs
                    if not self.is_valid_arxiv_url(url):
                        continue
                    
                    # Extract PDF URL with improved logic
                    pdf_url = None
                    if '/abs/' in url:
                        # Standard arXiv abstract URL format
                        pdf_url = url.replace('/abs/', '/pdf/') + '.pdf'
                    elif '/pdf/' in url:
                        # If already a PDF URL
                        pdf_url = url if url.endswith('.pdf') else url + '.pdf'
                    else:
                        # Try to extract arXiv ID and construct PDF URL
                        import re
                        arxiv_id_match = re.search(r'(\d+\.\d+)', url)
                        if arxiv_id_match:
                            arxiv_id = arxiv_id_match.group(1)
                            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                        
                    article = {
                        "title": result.get("title", ""),
                        "url": url,
                        "snippet": result.get("body", ""),
                        "source": "arxiv.org",
                        "query": query,
                        "pdf_url": pdf_url
                    }
                    
                    # Only add if URL is not already in results
                    if article["url"] and not any(r["url"] == article["url"] for r in results):
                        results.append(article)
                        logger.info(f"Found article: {article['title']}")
                        logger.debug(f"Article details: {article}")
                        
                # Be nice to the search engine
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error searching for query '{query}': {e}")
                
        logger.info(f"Search completed. Found {len(results)} unique articles")
        return results
        
    def _extract_domain(self, url: str) -> str:
        """Extract the domain from a URL"""
        try:
            return urlparse(url).netloc
        except:
            return "unknown"
            
    def fetch_article_content(self, url: str) -> Dict[str, Any]:
        """
        Fetch the content of an article from its URL
        
        Args:
            url: The URL of the article
            
        Returns:
            Dictionary with the article content
        """
        logger.info(f"Fetching content from URL: {url}")
        headers = {"User-Agent": self.get_random_user_agent()}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Extract the main content (simplified approach)
            # In a real implementation, site-specific extractors would improve this
            title = soup.title.string if soup.title else ""
            
            # Try to get article content from common article tags
            content_tags = soup.find_all(["article", "main", "div"], class_=lambda c: c and any(x in str(c).lower() for x in ["content", "article", "entry", "post"]))
            
            paragraphs = []
            if content_tags:
                for tag in content_tags:
                    for p in tag.find_all("p"):
                        if p.text.strip() and len(p.text.split()) > 10:  # Filter out short/empty paragraphs
                            paragraphs.append(p.text.strip())
            else:
                # Fallback: get all paragraphs
                for p in soup.find_all("p"):
                    if p.text.strip() and len(p.text.split()) > 10:
                        paragraphs.append(p.text.strip())
            
            content = "\n\n".join(paragraphs[:20])  # Limit to first 20 paragraphs
            
            logger.info(f"Successfully fetched content ({len(content)} characters)")
            logger.debug(f"Content preview: {content[:200]}...")
            
            return {
                "title": title,
                "url": url,
                "content": content,
                "content_length": len(content)
            }
            
        except Exception as e:
            logger.error(f"Error fetching article from {url}: {e}")
            return {
                "title": "",
                "url": url,
                "content": "",
                "content_length": 0,
                "error": str(e)
            }
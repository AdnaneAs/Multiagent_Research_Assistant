import os
import csv
import pandas as pd
import datetime
from typing import Dict, List, Any
from pathlib import Path

class IntegrationAgent:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the integration agent
        
        Args:
            data_dir: Directory to save data files
        """
        self.data_dir = data_dir
        self.ensure_data_dir()
        
    def ensure_data_dir(self):
        """Ensure the data directory exists"""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        
    def save_articles_to_csv(self, articles: List[Dict[str, Any]], topic: str) -> str:
        """
        Save the collected articles to a CSV file
        
        Args:
            articles: List of article dictionaries
            topic: Research topic
            
        Returns:
            Path to the saved CSV file
        """
        # Create filename with topic and date
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_topic = topic.replace(" ", "_").lower()
        csv_filename = f"{sanitized_topic}_{timestamp}.csv"
        csv_path = os.path.join(self.data_dir, csv_filename)
        
        # Create DataFrame from articles
        df = pd.DataFrame(articles)
        
        # Select and reorder columns
        columns = ['title', 'url', 'source', 'query', 'snippet']
        for col in columns:
            if col not in df.columns:
                df[col] = ''
                
        df = df[columns]
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def download_article_content(self, articles: List[Dict[str, Any]], article_contents: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Download and save article content to files
        
        Args:
            articles: List of article dictionaries
            article_contents: List of article content dictionaries
            
        Returns:
            Dictionary mapping URLs to file paths
        """
        url_to_filepath = {}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, content in enumerate(article_contents):
            if content.get("content_length", 0) == 0:
                # Skip articles with no content
                continue
                
            # Create a sanitized filename
            title_part = articles[i]["title"][:30].replace(" ", "_")
            title_part = ''.join(c if c.isalnum() or c == '_' else '' for c in title_part)
            filename = f"article_{i}_{timestamp}_{title_part}.txt"
            filepath = os.path.join(self.data_dir, filename)
            
            # Save the content to a file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Title: {content.get('title', '')}\n")
                f.write(f"URL: {content.get('url', '')}\n\n")
                f.write(content.get('content', ''))
                
            # Map URL to filepath
            url_to_filepath[content.get('url', '')] = filepath
            
        return url_to_filepath
        
    def update_csv_with_filepaths(self, csv_path: str, url_to_filepath: Dict[str, str]) -> str:
        """
        Update the CSV file with local file paths
        
        Args:
            csv_path: Path to the CSV file
            url_to_filepath: Dictionary mapping URLs to file paths
            
        Returns:
            Path to the updated CSV file
        """
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Add filepath column
        df['local_filepath'] = df['url'].map(url_to_filepath)
        
        # Save the updated CSV
        df.to_csv(csv_path, index=False)
        
        return csv_path
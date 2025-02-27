import os
import requests
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class IntegrationAgent:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def download_pdf(self, pdf_url: str, title: str) -> str:
        """Download PDF from arXiv and save with the article's title"""
        try:
            # Create sanitized filename
            sanitized_title = ''.join(c if c.isalnum() or c == '_' else '' for c in title)
            filename = f"{sanitized_title[:100]}.pdf"
            filepath = os.path.join(self.data_dir, filename)

            # Download PDF
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()

            # Save PDF
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Successfully downloaded PDF: {filename}")
            return filepath

        except Exception as e:
            logger.error(f"Error downloading PDF from {pdf_url}: {e}")
            return ""

    def process_articles(self, articles: List[Dict[str, Any]], topic: str) -> tuple:
        """Process articles: download PDFs and create CSV
        
        Returns:
            tuple: (csv_path, url_to_filepath) - CSV file path and mapping of URLs to local file paths
        """
        # Download PDFs for each article
        for article in articles:
            if article.get('pdf_url'):
                pdf_path = self.download_pdf(article['pdf_url'], article['title'])
                article['local_pdf_path'] = pdf_path

        # Save to CSV and get URL to filepath mapping
        csv_path, url_to_filepath = self.save_articles_to_csv(articles, topic)
        return csv_path, url_to_filepath

    def save_articles_to_csv(self, articles: List[Dict[str, Any]], topic: str) -> str:
        """Save the collected articles to a CSV file"""
        try:
            import pandas as pd
            import os
            
            # Create a sanitized filename for the CSV
            sanitized_topic = ''.join(c if c.isalnum() or c == '_' else '_' for c in topic)
            csv_filename = f"{sanitized_topic}_research_articles.csv"
            csv_path = os.path.join(self.data_dir, csv_filename)
            
            # Extract data for CSV
            data = []
            url_to_filepath = {}
            
            for article in articles:
                row = {}
                for key in ['title', 'url', 'source', 'query', 'snippet', 'pdf_url']:
                    row[key] = article.get(key, '')
                
                # Track local PDF path
                local_pdf_path = article.get('local_pdf_path', '')
                row['local_pdf_path'] = local_pdf_path
                
                # Keep track of which URL maps to which file path
                if local_pdf_path and article.get('url'):
                    url_to_filepath[article['url']] = local_pdf_path
                
                data.append(row)
            
            # Create DataFrame and save to CSV
            if data:
                df = pd.DataFrame(data)
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved {len(data)} articles to CSV: {csv_path}")
                return csv_path, url_to_filepath
            else:
                logger.warning("No articles to save to CSV")
                return None, {}
                
        except Exception as e:
            logger.error(f"Error saving articles to CSV: {e}")
            return None, {}

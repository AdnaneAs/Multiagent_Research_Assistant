import pandas as pd
from typing import Dict, List, Any
import os

class TransformationAgent:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def update_csv_with_abstracts(self, csv_path: str, abstracts: List[Dict[str, Any]]) -> str:
        """
        Update the CSV file with article abstracts
        
        Args:
            csv_path: Path to the CSV file
            abstracts: List of dictionaries containing file paths and abstracts
            
        Returns:
            Path to the updated CSV file
        """
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Create a mapping from file paths to abstracts
            file_to_abstract = {item["file_path"]: item["abstract"] for item in abstracts if "file_path" in item and "abstract" in item}
            
            # Add abstracts column based on the local_pdf_path (fix the field name)
            df['abstract'] = df['local_pdf_path'].map(file_to_abstract)
            
            # Save the updated CSV
            df.to_csv(csv_path, index=False)
            
            return csv_path
            
        except Exception as e:
            print(f"Error updating CSV with abstracts: {e}")
            return csv_path
            
    def generate_summary_report(self, csv_path: str, topic: str) -> Dict[str, Any]:
        """
        Generate a summary report of the research
        
        Args:
            csv_path: Path to the CSV file with articles and abstracts
            topic: Research topic
            
        Returns:
            Dictionary with summary statistics and information
        """
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Calculate summary statistics
            total_articles = len(df)
            articles_with_abstracts = df['abstract'].notna().sum()
            sources = df['source'].value_counts().to_dict()
            
            # Get article titles with abstracts
            articles_with_abstracts_list = []
            for _, row in df.iterrows():
                if pd.notna(row.get('abstract')):
                    articles_with_abstracts_list.append({
                        'title': row.get('title', 'Unknown Title'),
                        'source': row.get('source', 'Unknown Source'),
                        'has_abstract': True
                    })
                else:
                    articles_with_abstracts_list.append({
                        'title': row.get('title', 'Unknown Title'),
                        'source': row.get('source', 'Unknown Source'),
                        'has_abstract': False
                    })
            
            # Create summary report
            report = {
                'topic': topic,
                'total_articles': total_articles,
                'articles_with_abstracts': articles_with_abstracts,
                'sources': sources,
                'articles': articles_with_abstracts_list,
                'csv_path': csv_path
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating summary report: {e}")
            return {
                'topic': topic,
                'error': str(e),
                'csv_path': csv_path
            }

    def generate_csv_with_details(self, articles: List[Dict[str, Any]], rag_agent) -> str:
        """Generate CSV with article details and communicate with RAGAgent if information is missing"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Process only articles that have local PDF paths
        processed_articles = []
        
        for article in articles:
            # Skip articles without PDF paths
            if not article.get('local_pdf_path'):
                logger.warning(f"Skipping article without PDF path: {article.get('title', 'Unknown')}")
                continue
                
            try:
                article_copy = article.copy()
                
                if not article_copy.get('abstract'):
                    # Retrieve abstract using RAGAgent
                    logger.info(f"Retrieving abstract for: {article_copy.get('title', 'Unknown')}")
                    abstract = rag_agent.retrieve_abstract(article_copy['local_pdf_path'])
                    article_copy['abstract'] = abstract

                if not article_copy.get('authors'):
                    # Retrieve authors using RAGAgent
                    logger.info(f"Retrieving authors for: {article_copy.get('title', 'Unknown')}")
                    authors = rag_agent.retrieve_authors(article_copy['local_pdf_path'])
                    article_copy['authors'] = authors

                if not article_copy.get('link'):
                    # If article has a URL, use it as the link
                    if article_copy.get('url'):
                        article_copy['link'] = article_copy['url']
                    else:
                        # Retrieve link using RAGAgent
                        logger.info(f"Retrieving link for: {article_copy.get('title', 'Unknown')}")
                        link = rag_agent.retrieve_link(article_copy['local_pdf_path'])
                        article_copy['link'] = link
                        
                processed_articles.append(article_copy)
                
            except Exception as e:
                logger.error(f"Error processing article {article.get('title', 'Unknown')}: {e}")
                # Include article with error information
                article_copy = article.copy()
                article_copy['error'] = str(e)
                processed_articles.append(article_copy)

        # Save to CSV
        csv_path = self.save_articles_to_csv(processed_articles)
        logger.info(f"Successfully processed {len(processed_articles)} articles with details")
        return csv_path

    def save_articles_to_csv(self, articles: List[Dict[str, Any]]) -> str:
        """Save the collected articles to a CSV file"""
        # Create DataFrame from articles
        df = pd.DataFrame(articles)
        
        # Select and reorder columns
        columns = ['title', 'authors', 'link', 'abstract', 'local_pdf_path']
        for col in columns:
            if col not in df.columns:
                df[col] = ''
                
        df = df[columns]
        
        # Save to CSV
        csv_path = os.path.join(self.data_dir, 'articles_details.csv')
        df.to_csv(csv_path, index=False)
        return csv_path
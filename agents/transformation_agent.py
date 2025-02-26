import pandas as pd
from typing import Dict, List, Any
import os

class TransformationAgent:
    def __init__(self):
        """
        Initialize the transformation agent
        """
        pass
        
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
            
            # Add abstracts column based on the local_filepath
            df['abstract'] = df['local_filepath'].map(file_to_abstract)
            
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
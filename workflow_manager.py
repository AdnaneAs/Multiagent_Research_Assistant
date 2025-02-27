from langgraph.graph import StateGraph
import os
import json
import logging
from typing import Dict, List, Any, Optional, TypedDict, Sequence, Union
from agents.planning_agent import PlanningAgent
from agents.search_agent import SearchAgent
from agents.integration_agent import IntegrationAgent
from agents.abstract_agent import AbstractAgent
from agents.transformation_agent import TransformationAgent
from agents.writing_agent import WritingAgent
from agents.rag_agent import RAGAgent
import concurrent.futures
import time
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ResearchWorkflow")

class WorkflowState(TypedDict, total=False):
    topic: str
    plan: Dict[str, Any]
    articles: List[Dict[str, Any]]
    article_contents: List[str]
    csv_path: str
    url_to_filepath: Dict[str, str]
    abstracts: List[Dict[str, Any]]
    final_csv_path: str
    report: str
    latex_report: Dict[str, Any]

class ResearchWorkflow:
    def __init__(self, 
                 data_dir: str = "data", 
                 model_provider: str = "openai", 
                 model_id: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None):
        """
        Initialize the research workflow with all required agents
        
        Args:
            data_dir: Directory to store data files
            model_provider: The model provider ('openai' or 'ollama')
            model_id: The model ID to use
            api_key: API key for OpenAI models (not needed for Ollama)
        """
        logger.info(f"Initializing ResearchWorkflow with provider={model_provider}, model={model_id}")
        self.planning_agent = PlanningAgent(provider=model_provider, model_id=model_id, api_key=api_key)
        self.search_agent = SearchAgent()
        self.integration_agent = IntegrationAgent(data_dir=data_dir)
        self.abstract_agent = AbstractAgent(provider=model_provider, model_id=model_id, api_key=api_key)
        self.transformation_agent = TransformationAgent(data_dir=data_dir)
        self.writing_agent = WritingAgent(provider=model_provider, model_id=model_id, api_key=api_key)
        self.rag_agent = RAGAgent()
        self.data_dir = data_dir
        logger.info("All agents initialized successfully")
        
        # Setup state graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the workflow graph"""
        workflow = StateGraph(WorkflowState)
        
        # Define nodes for each step
        workflow.add_node("planning", self._planning_step)
        workflow.add_node("searching", self._search_step)
        workflow.add_node("integration", self._integration_step)
        workflow.add_node("abstracting", self._abstract_step)
        workflow.add_node("transformation", self._transformation_step)
        workflow.add_node("writing", self._writing_step)
        
        # Define edges
        workflow.add_edge("planning", "searching")
        workflow.add_edge("searching", "integration")
        workflow.add_edge("integration", "abstracting")
        workflow.add_edge("abstracting", "transformation")
        workflow.add_edge("transformation", "writing")
        
        # Set entry point
        workflow.set_entry_point("planning")

        # Mark writing as the end node
        workflow.set_finish_point("writing") 
        
        # Compile the graph
        return workflow.compile()
        
    def _planning_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate research plan"""
        logger.info("🎯 Starting Planning Step")
        topic = state.get("topic", "")
        logger.info(f"Generating research plan for topic: {topic}")
        
        plan = self.planning_agent.generate_plan(topic)
        logger.info(f"Generated plan: {json.dumps(plan, indent=2)}")
        
        return {"plan": plan}
        
    def _search_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Search for articles"""
        logger.info("🔍 Starting Search Step")
        plan = state.get("plan", {})
        logger.info("Searching for articles based on plan...")
        
        articles = self.search_agent.search_articles(plan)
        logger.info(f"Found {len(articles)} articles")
        
        # Fetch article contents
        logger.info("Fetching content for each article...")
        article_contents = []
        for i, article in enumerate(articles, 1):
            logger.info(f"Fetching content for article {i}/{len(articles)}: {article.get('title', '')}")
            content = self.search_agent.fetch_article_content(article["url"])
            article_contents.append(content)
            
        logger.info(f"Successfully fetched content for {len(article_contents)} articles")
        return {"articles": articles, "article_contents": article_contents}
        
    def _integration_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Save articles to CSV and download PDFs"""
        logger.info("💾 Starting Integration Step")
        articles = state.get("articles", [])
        plan = state.get("plan", {})
        topic = plan.get("topic", "research_topic")
        
        # Download PDFs and save to CSV
        logger.info("Processing articles...")
        csv_path, url_to_filepath = self.integration_agent.process_articles(articles, topic)
        logger.info(f"Articles processed and saved to: {csv_path}")
        
        return {
            "csv_path": csv_path,
            "url_to_filepath": url_to_filepath
        }
        
    def _abstract_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate abstracts for articles"""
        logger.info("📝 Starting Abstract Generation Step")
        url_to_filepath = state.get("url_to_filepath", {})
        if url_to_filepath is None:
            url_to_filepath = {}
        logger.info(f"Processing {len(url_to_filepath)} articles for abstract generation")
        
        abstracts = []
        # Process each article file
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_filepath = {
                executor.submit(self.abstract_agent.process_article_file, filepath): filepath
                for filepath in url_to_filepath.values() if filepath
            }
            
            for future in concurrent.futures.as_completed(future_to_filepath):
                filepath = future_to_filepath[future]
                try:
                    logger.info(f"Generating abstract for: {os.path.basename(filepath)}")
                    result = future.result()
                    abstracts.append(result)
                    logger.info(f"Successfully generated abstract for: {os.path.basename(filepath)}")
                except Exception as e:
                    logger.error(f"Error processing {filepath}: {e}")
                    abstracts.append({
                        "file_path": filepath,
                        "abstract": f"Error: {e}",
                        "error": str(e)
                    })
        
        logger.info(f"Completed abstract generation for {len(abstracts)} articles")            
        return {"abstracts": abstracts}
        
    def _transformation_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate CSV with details and communicate with RAGAgent if information is missing"""
        logger.info("🔄 Starting Transformation Step")
        csv_path = state.get("csv_path", "")
        articles = state.get("articles", [])
        
        # Generate CSV with details
        logger.info("Generating CSV with article details...")
        detailed_csv_path = self.transformation_agent.generate_csv_with_details(articles, self.rag_agent)
        logger.info(f"Detailed CSV saved to: {detailed_csv_path}")
        
        return {
            "final_csv_path": detailed_csv_path
        }

    def _writing_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate LaTeX academic report"""
        logger.info("📝 Starting Writing Step")
        plan = state.get("plan", {})
        csv_path = state.get("final_csv_path", "")
        
        logger.info("Writing academic report...")
        latex_report = self.writing_agent.write_report(plan, csv_path)
        
        # Save LaTeX document to file
        report_path = os.path.join(self.data_dir, "academic_report.tex")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(latex_report["latex_document"])
        logger.info(f"LaTeX report saved to: {report_path}")
        
        return {
            "latex_report": latex_report,
            "report_path": report_path
        }
        
    def run(self, topic: str) -> Dict[str, Any]:
        """
        Run the complete research workflow
        
        Args:
            topic: Research topic to investigate
            
        Returns:
            Dictionary with the final results
        """
        logger.info(f"🚀 Starting research workflow for topic: {topic}")
        # Initialize state with topic
        state = {"topic": topic}
        
        # Run the workflow
        try:
            # Use the correct method to invoke the graph
            # Instead of directly calling the graph object, use graph.invoke()
            result = self.graph.invoke(state)
            logger.info("🏁 Workflow execution completed")
            return result
        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}")
            raise
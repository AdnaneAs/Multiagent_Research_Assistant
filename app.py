import streamlit as st
import pandas as pd
import os
import time
import json
from dotenv import load_dotenv
from workflow_manager import ResearchWorkflow
import traceback
from utils.model_utils import get_available_ollama_models, get_openai_model_options
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('research_workflow.log')  # Also save to file
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

def main():
    # Header
    st.title("🔍 Multi-Agent Research Assistant")
    st.subheader("Automated article search and summarization pipeline")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model provider selection
        model_provider = st.radio(
            "Select Model Provider",
            ["OpenAI", "Ollama (Local)"],
            help="Choose between OpenAI cloud models or locally hosted Ollama models"
        )
        
        api_key = None
        model_id = None
        
        # Display model options based on provider
        if model_provider == "OpenAI":
            # OpenAI model selection
            openai_models = get_openai_model_options()
            model_names = [model["name"] for model in openai_models]
            selected_model_name = st.selectbox("Select OpenAI Model", model_names)
            
            # Find the selected model
            selected_model = next((m for m in openai_models if m["name"] == selected_model_name), openai_models[0])
            model_id = selected_model["id"]
            
            # API key input for OpenAI
            api_key = st.text_input("OpenAI API Key", type="password", 
                                  help="Enter your OpenAI API key here. It will be stored temporarily in the .env file.")
            
            # Save API key to .env if provided
            if api_key:
                with open(".env", "w") as env_file:
                    env_file.write(f"OPENAI_API_KEY={api_key}\n")
                st.success("API key saved temporarily!")
        else:
            # Ollama model selection
            try:
                # Get available Ollama models
                ollama_models = get_available_ollama_models()
                
                if not ollama_models:
                    st.warning("No Ollama models found. Make sure Ollama is running locally.")
                    st.info("Learn how to install Ollama: https://ollama.com/download")
                    model_names = ["No models found"]
                else:
                    model_names = [model["name"] for model in ollama_models]
                    
                selected_model_name = st.selectbox("Select Ollama Model", model_names) if model_names else "No models found"
                
                # Find the selected model
                if ollama_models and selected_model_name != "No models found":
                    selected_model = next((m for m in ollama_models if m["name"] == selected_model_name), ollama_models[0])
                    model_id = selected_model["name"]
                    st.success(f"Using local Ollama model: {model_id}")
                else:
                    model_id = None
                
            except Exception as e:
                st.error(f"Error connecting to Ollama: {str(e)}")
                st.info("Make sure Ollama is running locally: https://ollama.com/download")
                model_id = None
        
        st.divider()
        st.markdown("### How it works")
        st.markdown("""
        1. **Planning Agent**: Creates a research and LaTeX report plan
        2. **Search Agent**: Finds relevant articles on arXiv
        3. **Integration Agent**: Downloads PDFs and creates CSV
        4. **Abstract Agent**: Summarizes each article
        5. **Transformation Agent**: Updates CSV with abstracts and other details
        6. **Writing Agent**: Generates LaTeX report
        """)
    
    # Main area - Topic input
    topic = st.text_input("Enter your research topic keywords:", 
                        placeholder="e.g., artificial intelligence ethics, climate change solutions")
    
    # Check if model is selected
    is_valid_model = (model_provider == "OpenAI" and api_key and model_id) or \
                     (model_provider == "Ollama (Local)" and model_id is not None and model_id != "No models found")
    
    # Start research button
    start_research = st.button("Start Research", type="primary", disabled=not (topic and is_valid_model))
    
    if not is_valid_model and model_provider == "OpenAI" and not api_key:
        st.warning("Please enter your OpenAI API key to continue.")
    elif not is_valid_model and model_provider == "Ollama (Local)" and (model_id is None or model_id == "No models found"):
        st.warning("Please make sure Ollama is running and at least one model is installed.")
    
    # Initialize session state for tracking progress
    if "research_complete" not in st.session_state:
        st.session_state.research_complete = False
    if "current_step" not in st.session_state:
        st.session_state.current_step = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "error" not in st.session_state:
        st.session_state.error = None
    
    # Run research when button is clicked
    if start_research:
        st.session_state.research_complete = False
        st.session_state.current_step = "planning"
        st.session_state.results = None
        st.session_state.error = None
        
        # Create progress container
        progress_container = st.container()
        
        try:
            with progress_container:
                # Initialize workflow with selected model
                provider = "openai" if model_provider == "OpenAI" else "ollama"
                
                workflow = ResearchWorkflow(
                    data_dir="data",
                    model_provider=provider,
                    model_id=model_id,
                    api_key=api_key
                )
                
                # Create progress visualization
                steps_col = st.columns(1)[0]
                with steps_col:
                    st.info("Running complete research workflow. This may take some time...")
                    progress_bar = st.progress(0)
                
                # Actually run the workflow (this might take some time)
                with st.spinner("Running research workflow..."):
                    results = workflow.run(topic)
                    st.session_state.results = results
                    progress_bar.progress(100)
                
                st.success("Research completed successfully!")
                st.session_state.research_complete = True
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.error = str(e)
            st.code(traceback.format_exc())
    
    # Display results if research is complete
    if st.session_state.research_complete and st.session_state.results:
        results = st.session_state.results
        
        # Display research plan
        st.header("1. Research Plan")
        if "plan" in results:
            plan = results["plan"]
            topic = plan.get("topic", "")
            plan_details = plan.get("plan", {})
            
            st.subheader(f"Topic: {topic}")
            
            # Display subtopics
            if "subtopics" in plan_details:
                st.write("**Subtopics:**")
                for subtopic in plan_details["subtopics"]:
                    st.markdown(f"- {subtopic}")
            
            # Display search queries
            if "search_queries" in plan_details:
                st.write("**Search Queries:**")
                for query in plan_details["search_queries"]:
                    st.markdown(f"- {query}")
            
            # Display expected outcome
            if "expected_outcome" in plan_details:
                st.write("**Expected Outcome:**")
                st.write(plan_details["expected_outcome"])
            
            # Display LaTeX report plan
            if "latex_report_plan" in plan_details:
                st.write("**LaTeX Report Structure:**")
                latex_plan = plan_details["latex_report_plan"]
                for section, points in latex_plan.items():
                    st.write(f"**{section.replace('_', ' ').title()}:**")
                    for point in points:
                        st.write(f"- {point}")
        
        # Display search results
        st.header("2. Search Results")
        if "articles" in results:
            articles = results["articles"]
            st.write(f"Found {len(articles)} articles")
            
            # Show article list in a dataframe
            if articles:
                df_display = pd.DataFrame([{
                    "Title": article["title"],
                    "Source": article["source"],
                    "Query": article["query"]
                } for article in articles])
                
                st.dataframe(df_display, use_container_width=True)
        
        # Display final results
        st.header("3. Final Results")
        if "report" in results:
            report = results["report"]
            
            # Statistics
            st.subheader("Research Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Articles", report.get("total_articles", 0))
            col2.metric("Articles with Abstracts", report.get("articles_with_abstracts", 0))
            col3.metric("Sources", len(report.get("sources", {})))
            
            # Show abstracts if available
            if "final_csv_path" in results:
                csv_path = results["final_csv_path"]
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    
                    # Show download button for the CSV
                    csv_filename = os.path.basename(csv_path)
                    with open(csv_path, "rb") as file:
                        st.download_button(
                            label="Download CSV results",
                            data=file,
                            file_name=csv_filename,
                            mime="text/csv"
                        )
                    
                    # Display dataframe with abstracts
                    st.subheader("Articles with Abstracts")
                    st.dataframe(df["title", "authors", "link", "abstract"].dropna(subset=["abstract"]), use_container_width=True)

        # Display LaTeX report
        st.header("4. LaTeX Report")
        if "latex_report" in results:
            latex_report = results["latex_report"]
            report_path = results.get("report_path")
            
            if report_path and os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                    
                st.code(report_content, language='latex')
                
                # Download LaTeX button
                st.download_button(
                    label="Download LaTeX Report",
                    data=report_content,
                    file_name="academic_report.tex",
                    mime="text/plain"
                )

    # Display error if any
    if st.session_state.error:
        st.error(f"Error: {st.session_state.error}")
        

if __name__ == "__main__":
    main()
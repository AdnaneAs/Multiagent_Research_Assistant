import logging
from typing import Dict, List, Any, Optional
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import LongContextReorder
import chromadb
import requests
import tempfile
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger("RAGAgent")

# Load environment variables
load_dotenv()

class RAGAgent:
    def __init__(self, embedding_model: str = "nomic-embed-text"):
        """
        Initialize the RAG agent
        
        Args:
            embedding_model: The Ollama embedding model to use
        """
        logger.info("Initializing RAGAgent")
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url="http://localhost:11434"
        )
        
        # Initialize text splitters
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        
        # Initialize vector store with ChromaDB
        self.vector_store = Chroma(
            persist_directory="data/chroma_db",
            embedding_function=self.embeddings
        )
        
        # Initialize storage for parent documents
        self.store = InMemoryStore()
        
        # Initialize retriever with parent-child strategy
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter
        )
        
        logger.info("RAGAgent initialized successfully")

    def extract_article_content(self, pdf_url: str, article_id: str) -> Dict[str, Any]:
        """Extract content from PDF article"""
        try:
            # Download PDF to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                response = requests.get(pdf_url)
                response.raise_for_status()
                temp_file.write(response.content)
                temp_path = temp_file.name

            # Load PDF
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Extract and combine text
            content = "\n\n".join(page.page_content for page in pages)
            
            # Add to vector store
            self.add_to_knowledge_base(content, {"source": pdf_url, "id": article_id})
            
            return {
                "success": True,
                "content": content,
                "num_pages": len(pages)
            }
            
        except Exception as e:
            logger.error(f"Error extracting content from PDF {pdf_url}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def add_to_knowledge_base(self, content: str, metadata: Dict[str, Any]) -> None:
        """Add content to the vector store"""
        try:
            # Split and add documents
            self.retriever.add_documents([
                {"page_content": content, "metadata": metadata}
            ])
            logger.info(f"Added content to knowledge base with metadata: {metadata}")
        except Exception as e:
            logger.error(f"Error adding content to knowledge base: {e}")
            raise

    def query_knowledge_base(self, query: str, num_results: int = 5) -> List[str]:
        """Query the knowledge base for relevant content"""
        try:
            # Get relevant documents using invoke() instead of get_relevant_documents()
            docs = self.retriever.invoke(query)
            
            # Reorder for better context
            reordering = LongContextReorder()
            reordered_docs = reordering.transform_documents(docs[:num_results])
            
            # Extract and return content
            return [doc.page_content for doc in reordered_docs]
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return [f"Error retrieving content: {e}"]

    def retrieve_abstract(self, pdf_path: str) -> str:
        """Retrieve abstract from the indexed PDF"""
        try:
            query = "abstract"
            results = self.query_knowledge_base(query)
            return results[0] if results else ""
        except Exception as e:
            logger.error(f"Error retrieving abstract from {pdf_path}: {e}")
            return ""

    def retrieve_authors(self, pdf_path: str) -> str:
        """Retrieve authors from the indexed PDF"""
        try:
            query = "authors"
            results = self.query_knowledge_base(query)
            return results[0] if results else ""
        except Exception as e:
            logger.error(f"Error retrieving authors from {pdf_path}: {e}")
            return ""

    def retrieve_link(self, pdf_path: str) -> str:
        """Retrieve link from the indexed PDF"""
        try:
            query = "link"
            results = self.query_knowledge_base(query)
            return results[0] if results else ""
        except Exception as e:
            logger.error(f"Error retrieving link from {pdf_path}: {e}")
            return ""
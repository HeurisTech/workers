from langchain_core.tools import tool
from pydantic import BaseModel, Field
import json

# Correcting the import path to be relative within the 'src' directory
from ..supabase.retriever import retrieve_organizational_knowledge
from .state import KnowledgeSearchInput

@tool(args_schema=KnowledgeSearchInput)
def knowledge_base_retriever(query: str, organization_id: str) -> str:
    """
    Searches the organization's private knowledge base to answer questions.
    This tool is the best choice for questions about internal documents, project history, or specific organizational knowledge.
    """

    print(f"Tool received query: '{query}' for organization_id: '{organization_id}'")
    
    # We use 'text' search for broader results, as it proved more effective in testing.
    results = retrieve_organizational_knowledge(query, organization_id, search_type="text")
    
    if not results:
        return "No relevant information found in the knowledge base for this query."
    
    # Format the results into a single string for the LLM
    formatted_results = "\n\n".join(
        [
            f"Source: {res.get('title', 'N/A')}\nContent: {res.get('content', 'N/A')}"
            for res in results
        ]
    )
    
    return f"Found {len(results)} relevant documents:\n\n{formatted_results}"

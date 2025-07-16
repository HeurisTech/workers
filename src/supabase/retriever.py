import os
import requests
from typing import List, Dict, Any

# It's good practice to have the URL configurable, e.g., via environment variables.
# Defaulting to the standard Next.js dev port.
BASE_API_URL = os.getenv("OVEN_API_URL", "http://localhost:3000")
RETRIEVE_ENDPOINT = f"{BASE_API_URL}/api/knowledge-base/retrieve"

def retrieve_organizational_knowledge(
    query: str, organization_id: str, search_type: str = "similarity",
) -> List[Dict[str, Any]]:
    """
    Retrieves documents from the knowledge base for a specific organization
    by calling the oven-ai Next.js API endpoint.

    Args:
        query: The user's query string.
        organization_id: The ID of the organization to filter by.
        search_type: The type of search to perform ('similarity', 'text', or 'hybrid').

    Returns:
        A list of document chunks that are most relevant to the query.
        Returns an empty list if the request fails.
    """
    payload = {
        "query": query,
        "organizationId": organization_id,
        "searchType": search_type,
    }
    
    print(f"Sending request to {RETRIEVE_ENDPOINT} with payload: {payload}")

    try:
        response = requests.post(RETRIEVE_ENDPOINT, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()
        
        if data.get("success"):
            results = data.get("data", {}).get("results", [])
            print(f"Successfully retrieved {len(results)} results.")
            return results
        else:
            print(f"API request was not successful: {data.get('error', 'No error message')}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the retrieval API: {e}")
        return [] 
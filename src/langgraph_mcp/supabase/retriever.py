import os
import requests
from typing import List, Dict, Any

# It's good practice to have the URL configurable, e.g., via environment variables.
# Defaulting to the standard Next.js dev port.
BASE_API_URL = os.getenv("OVEN_API_URL", "http://localhost:3000")
RETRIEVE_ENDPOINT = f"{BASE_API_URL}/api/knowledge-base/retrieve"

def retrieve_organizational_knowledge(
    query: str, organization_id: str, search_type: str = "text",
) -> List[Dict[str, Any]]:
    """
    Retrieves documents from the knowledge base for a specific organization
    by calling the oven-ai Next.js API endpoint.

    ##TODO: This hardcoded retrieval implementation should be made configurable
    to support different knowledge base backends and API endpoints.

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

# if __name__ == "__main__":
#     import json

#     organization_id = "0f5c7483-be9c-4e69-94c9-cf3ce7e378ec"
#     query = "localization techniques"
#     # TODO: Add a dynamic search type
#     # search_types = ["text", "similarity", "hybrid"]
#     search_type = "text"
#     # for search_type in search_types:
#     #     print(f"\n--- Testing with search_type: {search_type} ---")
#     results = retrieve_organizational_knowledge(
#             query=query,
#             organization_id=organization_id,
#             search_type=search_type,
#     )
#     if results:
#             print(f"Found {len(results)} results:")
#             for i, result in enumerate(results, 1):
#                 print(f"  Result {i}:")
#                 print(f"    {json.dumps(result, indent=4)}")
#     else:
#             print("No results found or an error occurred.") 

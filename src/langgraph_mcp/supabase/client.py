import os
from supabase import create_client, Client

def get_supabase_client() -> Client:
    """
    Initializes and returns a Supabase client instance.
    
    Reads NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY from environment variables.
    
    Returns:
        A Supabase client instance.
        
    Raises:
        ValueError: If Supabase credentials are not found in environment variables.
    """
    supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    supabase_key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL and Key must be set in environment variables.")

    return create_client(supabase_url, supabase_key) 
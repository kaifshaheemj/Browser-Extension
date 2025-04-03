#For testing the fastapi endpoint, this python file is used.

import requests
import json

def query_pdf_api(url, query, api_endpoint="http://localhost:8000/process_pdf"):
    """
    Send a request to the PDF processing API
    
    Args:
        url: URL of the PDF to process
        query: User's question about the PDF
        api_endpoint: The API endpoint URL
        
    Returns:
        The API response as a dictionary
    """
    payload = {
        "content_url": url,
        "user_query": query
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(api_endpoint, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    # Example usage
    url = "https://www.w3schools.com/python/python_intro.asp"
    query = "Why python?"
    
    result = query_pdf_api(url, query)
    
    if result:
        print("API Response:")
        print(f"Status: {result['status']}")
        print(f"Answer: {result['answer']}")
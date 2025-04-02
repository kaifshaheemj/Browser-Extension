from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage

# Import the process function from your existing code
from mainfile import process

app = FastAPI(title="PDF Processing API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define request model
class QueryRequest(BaseModel):
    content_url: str
    user_query: str
    google_api_key: Optional[str] = None
    google_credentials_path: Optional[str] = None

# Define response model
class QueryResponse(BaseModel):
    answer: str
    status: str

# Global variables for API keys and credentials path
DEFAULT_GOOGLE_API_KEY = "AIzaSyAcAics3uBFyZSnRQsAdf9FJlKAwnMNuHU"
DEFAULT_GOOGLE_CREDENTIALS_PATH = "/Users/saivignesh/Documents/DGM_Project/Browser-Extension/fabled-emblem-450414-r0-a631ebb8abd9.json"

@app.post("/process_pdf", response_model=QueryResponse)
async def process_pdf(request: QueryRequest):
    """
    Process a PDF from a URL and answer a user query about its content.
    
    - **content_url**: URL of the PDF to process
    - **user_query**: User's question about the PDF content
    - **google_api_key**: Optional Google API key (will use default if not provided)
    - **google_credentials_path**: Optional path to Google credentials file (will use default if not provided)
    """
    try:
        # Use provided keys/paths or fall back to defaults
        google_api_key = request.google_api_key or DEFAULT_GOOGLE_API_KEY
        google_credentials_path = request.google_credentials_path or DEFAULT_GOOGLE_CREDENTIALS_PATH
        
        # Validate input
        if not request.content_url:
            raise HTTPException(status_code=400, detail="URL is required")
        if not request.user_query:
            raise HTTPException(status_code=400, detail="User query is required")
            
        # Process the PDF and get answer
        result = process(
            request.content_url,
            request.user_query,
            google_api_key,
            google_credentials_path
        )
        
        # Extract content from AIMessage if needed
        answer = result
        if isinstance(result, List) and result and isinstance(result[-1], AIMessage):
            answer = result[-1].content
        elif isinstance(result, AIMessage):
            answer = result.content
        
        return QueryResponse(
            answer=answer,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
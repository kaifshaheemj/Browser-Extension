from dotenv import load_dotenv
import os

print("Before loading .env:", os.environ.get("GOOGLE_API_KEY"))  # Debugging

# Load environment variables from .env file
load_dotenv()

print("After loading .env:", os.environ.get("GOOGLE_API_KEY"))  # Debugging

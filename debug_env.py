import os
from dotenv import load_dotenv

print("--- Minimal Env Debug ---")
print(f"Current Working Directory: {os.getcwd()}")
print(f"Searching for .env in: {os.getcwd()}")

# Load without override first
load_dotenv()
print(f"After load_dotenv(): LANGSMITH_API_KEY = {os.getenv('LANGSMITH_API_KEY', 'NOT FOUND')[:15]}...")

# Load with override
load_dotenv(override=True)
print(f"After load_dotenv(override=True): LANGSMITH_API_KEY = {os.getenv('LANGSMITH_API_KEY', 'NOT FOUND')[:15]}...")

print(f"LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")

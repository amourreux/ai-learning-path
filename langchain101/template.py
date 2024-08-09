import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

print('hello world')

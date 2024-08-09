import os
from dotenv import load_dotenv
# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI

# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(temperature=0.9) # temperature 0 - 1

prompt = "Türkiye de yazları ve kışları sıcak olan bir şehir?"

print(llm(prompt))
result = llm.generate([prompt]*5)
for c_name in result.generations:
    print(c_name[0].text)


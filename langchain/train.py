import os
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = os.environ['OPENAI_API_BASE']
os.environ["OPENAI_API_KEY"] = os.environ['OPENAI_API_KEY']
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

llm = AzureOpenAI(
    deployment_name="gpt-35",
    model_name="gpt-35-turbo",
)

# Run the LLM
print(llm("Tell me a joke"))
print(llm)
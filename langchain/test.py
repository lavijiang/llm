import os
from langchain import ConversationChain
from langchain.agents import load_tools,initialize_agent,AgentType
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA

_ = load_dotenv(find_dotenv()) # read local .env file

llm = AzureOpenAI(
    deployment_name="gpt-35",
    model_name="gpt-35-turbo",
)

# 加载文件夹中的所有txt类型的文件
loader = DirectoryLoader('data/', glob='**/*.txt')
# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()
# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
# 切割加载的 document
split_docs = text_splitter.split_documents(documents)
# 初始化 openai 的 embeddings 对象
embeddings = OpenAIEmbeddings(deployment="text-embedding",chunk_size=1)
# 持久化数据
docsearch = Chroma.from_documents(documents, embeddings, persist_directory="persist")
docsearch.persist()
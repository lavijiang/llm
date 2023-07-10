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
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate
)
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain

from langchain.chat_models import ChatOpenAI

_ = load_dotenv(find_dotenv()) # read local .env file

llm = AzureOpenAI(
    deployment_name="gpt-35",
    model_name="gpt-35-turbo",
)

llm = AzureChatOpenAI(
    deployment_name=os.getenv("OPENAI_GPT_DEPLOYMENT_NAME"),
    temperature=0,
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    openai_api_type=os.getenv("OPENAI_API_TYPE"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    request_timeout=10,
    streaming=True,
)

# 加载 youtube 频道
loader = YoutubeLoader.from_youtube_url('https://www.youtube.com/watch?v=pVBHO2p7TpM')
# 将数据转成 document
documents = loader.load()

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=10
)

# 分割 youtube documents
documents = text_splitter.split_documents(documents)

# 初始化 openai embeddings
embeddings = OpenAIEmbeddings(deployment=os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME"),chunk_size=1)

# 将数据存入向量存储
vector_store = Chroma.from_documents(documents, embeddings)
# 通过向量存储初始化检索器
retriever = vector_store.as_retriever()

system_template = """
Use the following context to answer the user's question.
If you don't know the answer, say you don't, don't try to make it up.
-----------
{question}
-----------
{chat_history}
"""

# 构建初始 messages 列表，这里可以理解为是 openai 传入的 messages 参数
messages = [
  SystemMessagePromptTemplate.from_template(system_template),
  HumanMessagePromptTemplate.from_template('{question}')
]

# 初始化 prompt 对象
prompt = ChatPromptTemplate.from_messages(messages)

# 初始化问答链
qa = ConversationalRetrievalChain.from_llm(llm,retriever,condense_question_prompt=prompt,return_source_documents=True,verbose=False)

chat_history = []
while True:
  question = input('问题：')
  # 开始发送问题 chat_history 为必须参数,用于存储对话历史
  result = qa({'question': question, 'chat_history': chat_history})
  chat_history.append((question, result['answer']))
  print(result['answer'])
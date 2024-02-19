from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents import AgentExecutor, create_openai_tools_agent
from IPython.display import Markdown
from dotenv import load_dotenv

load_dotenv(".env")
llm = ChatOpenAI()
loader = PyPDFLoader("sample.pdf")
pages = loader.load_and_split()
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(pages, embeddings)
retriever = db.as_retriever()



tool = create_retriever_tool(
    retriever,
    "generate_lesson_plan",
    "Searches and generates lesson plan from the pdf about how to study , what all to study before attempting the questions etc"
)

tools = [tool]
prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.invoke({"input": "Generate a detailed lesson plan on what all the student should study and how the student should study to solve the quesions provided.also write a detailed summary of more than 100 words about  the different scenarios from the document provided make sure that the lesson plan is easy to follow and also generate a detailed mcq of around 10 questions with the right answers for each questions below the mcq"})
print(result["output"])
import json
import os
import re
from langchain import BasePromptTemplate, LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain.schema.prompt import PromptValue
from tools.agents.custom_search import DeepSearch
from tools.agents.stable_diffusion import GenerateImage
from tools.agents.loader import Chat_By_Document, DocumentLoader

from langchain import OpenAI, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser,
)
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import TextLoader
import os
from transformers import pipeline
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent

from tools.agents.vits.agent import Vits
os.environ["LANGCHAIN_TRACING"] = "true"
docstore = DocstoreExplorer(Wikipedia())

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
memory = ConversationBufferMemory(memory_key="memory", return_messages=True)


os.environ['OPENAI_API_KEY'] = 'EMPTY'
os.environ['OPENAI_API_BASE'] = 'http://localhost:8000/v1'

# embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
# loader = TextLoader("documents/state_of_the_union.txt", encoding="UTF-8")
# index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])


# Set up the base template
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question 

Begin!

{chat_history}
Question: {input}
{agent_scratchpad}
"""

tool_desc = """{name}: Call this tool to interact with the {name} API. What is the {name} API useful for? {description} Parameters: {parameters} Format the arguments as a JSON object."""


template1 = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}{agent_scratchpad}"""

# Set up a prompt template
stop = ["Observation:", "Observation:\n",
        "<|im_end|>", "<|endoftext|>", "<|im_start|>"]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2,
                 request_timeout=60000, streaming=True)


class PValue(PromptValue):
    formatted: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.formatted = kwargs.get('formatted')

    def to_string(self):
        return self.formatted

    def to_messages(self):
        return [HumanMessage(content=self.formatted)]


class CustomPromptTemplate(BasePromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format_prompt(self, **kwargs) -> PromptValue:
        return PValue(formatted=self.format(**kwargs))

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")

        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += '\n' + action.log
            thoughts += f"Observation: {observation}"
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided

        kwargs["tools"] = "\n\n".join(
            [f"{tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ",".join(
            [tool.name for tool in self.tools])

        formatted = self.template.format(**kwargs)
        return formatted


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split(
                    "Final Answer:")[-1].strip()},
                log="end!",
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log="not match",
            )

        action = match.group(1).strip()
        action_input = match.group(2).strip().split('\n')[0].strip()
        for name in tool_names:
            if name in action:
                return AgentAction(tool=name, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
        return AgentFinish(
            return_values={"output": llm_output.strip()},
            log=llm_output,
        )


def GetChatRecords(input):
    history = "\n".join(
        [f"{'user:' if message.type=='human' else 'assistant:'}{message.content}" for message in memory.chat_memory.messages])
    return "\n"+history


def GenerateImageWrapper(prompt):
    prompt = prompt.strip()
    json_res = json.loads(prompt)
    prompt = json_res.get('query')
    if prompt == "":
        return ""

    # sd_flag = "stable diffusion"
    # if prompt.find(sd_flag) == -1:
    #     translator = pipeline(
    #         "translation", model="Helsinki-NLP/opus-mt-zh-en")
    #     query = translator(prompt)[0]['translation_text']
    #     q = query.split(',')
    #     set_len = len(set(q))
    #     query = prompt if set_len < (len(q) - 5) else query
    # else:
    #     query = prompt.split(sd_flag)[1]
    query = prompt
    return GenerateImage.gen(query)


store = DocumentLoader()
# store.load_dir_files("./documents")


def FileLoaderWrapper(prompt):
    prompt = prompt.strip()
    json_res = json.loads(prompt)
    prompt = json_res.get('query')
    if prompt == "":
        return ""

    store.load_file(prompt)
    return './file.text'


chat_By_Document = Chat_By_Document(store.vector_store, llm)


def ChatByDocumentWrapper(prompt):
    prompt = prompt.strip()
    json_res = json.loads(prompt)
    prompt = json_res.get('query')
    if prompt == "":
        return ""
    top = 2
    # result = chat_By_Document.run(prompt)

    list = store.vector_store.similarity_search(prompt, top)

    return '\n'.join([f"info:{item.page_content}\n" for item in list])


vits = Vits()


def GenerateAudioWrapper(prompt):
    prompt = prompt.strip()
    json_res = json.loads(prompt)
    prompt = json_res.get('query')
    if prompt == "":
        return ""

    output = vits.run(prompt)
    return output


tools = [
    Tool(
        func=GetChatRecords,
        name="GetChatRecords",
        description='GetChatRecords: What is the get chat records useful for? useful for when you need to get information about user and previous questions and obtain historical conversation records. Parameters: [{"name": "query","type": "string","description": "get historical chat records","required": True}] Format the arguments as a JSON object.'
    ),
    Tool(
        func=FileLoaderWrapper,
        name='FileLoader',
        description='FileLoader: Call this tool to interact with the file loader API. What is the file loader API useful for? useful for when you need to load files. Parameters: [{"name": "query","type": "string","description": "load the file path","required": True}] Format the arguments as a JSON object.',
    ),
    Tool(
        func=ChatByDocumentWrapper,
        name='ChatByDocument',
        description='ChatByDocument: Call this tool to interact with the chat by document API. What is the chat by document API useful for? useful for when you need to obtain relevant content from a document. Parameters: [{"name": "query","type": "string","description": "search by document content","required": True}] Format the arguments as a JSON object.',
    ),
    Tool(
        func=GenerateImageWrapper,
        name="GenerateImage",
        description='GenerateImage: Call this tool to interact with the stable diffusion API. What is the stable diffusion API useful for? useful for when you need to generate images or pictures. Parameters: [{"name": "query","type": "string","description": "In English, generate image of stable diffusion","required": True}] Format the arguments as a JSON object.'
    ),
    Tool(
        func=GenerateAudioWrapper,
        name='GenerateAudio',
        description='GenerateAudio: Call this tool to interact with the vits audio API. What is the vits audio API useful for? 当您需要生成声音或音频时非常有用. Parameters: [{"name": "query","type": "string","description": "使用中文输入, 使用 vits api 生成语音","required": True}] Format the arguments as a JSON object.',
    ),
    Tool(
        func=DeepSearch.google_search,
        name='Search',
        description='Search: Call this tool to interact with the google search API. What is the google search API useful for? useful for when you need to answer questions about current events or today Information. Parameters: [{"name": "query","type": "string","description": "search query of google","required": True}] Format the arguments as a JSON object.',
    ),
]


tool_names = [tool.name for tool in tools]
output_parser = CustomOutputParser()
prompt = CustomPromptTemplate(
    template=template1,
    tools=tools,
    input_variables=["input",  "intermediate_steps"]
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=stop,
    allowed_tools=tool_names,
    verbose=True,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory, max_iterations=4)


if __name__ == '__main__':
    while True:
        try:
            message = input('User> ').strip()
        except UnicodeDecodeError:
            print('[ERROR] Encoding error in input')
            continue
        except KeyboardInterrupt:
            exit(1)
        try:
            a = agent_executor.run('' + message)
            print(a)
        except Exception as e:
            print('[ERROR] agent_executor run error', e)
            continue

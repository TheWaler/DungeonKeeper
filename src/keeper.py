import os
from langchain import ConversationChain, LLMChain, VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import load_prompt, PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, CombinedMemory, ConversationKGMemory
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from archive import Archive, Adventure

prefix = """You are Keeper, a helpful assistant to a Dungeon Master. Have a conversation with the DM, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"
Chat:
{chat_history}
Question: {input}
Thought:{agent_scratchpad}"""

class Keeper(object):
    def __init__(self, adventure: Adventure, generation_model=None, verbose=False, archive: Archive = None):
        self.adventure = adventure
        if generation_model is None:
            generation_model = OpenAI()
        self.generation_model = generation_model
        self.verbose = verbose
        if archive is None:
            archive = Archive()
        self.archive = archive
        self.conversation = self.create_agent(generation_model=generation_model, verbose=verbose)

    def answer(self, question):
        return self.conversation.run(question)

    def create_agent(self, generation_model, verbose=False):
        tools = [
            Tool(
                name = "Adveture Book QA System",
                func=self.archive.search,
                description="useful for when you need to answer questions about something specific from the adventure book. Input should be a fully formed question."
            ),
        ]
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"]
        )
        conversation_memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="input", k=3)
        memory = CombinedMemory(memories=[conversation_memory])
        chain = LLMChain(llm=generation_model, prompt=prompt, verbose=verbose)
        agent = ZeroShotAgent(tools=tools, llm_chain=chain, verbose=verbose, max_iterations=4, early_stopping_method="generate")
        return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose, memory=memory)
    
    def _search(self, query):
        return self.archive.search(query=query, adventure=self.adventure.name)
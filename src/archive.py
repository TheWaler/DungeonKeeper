# Archive is the data store for the keeper. It is responsible for storing the
# data used by the keeper by generating a chroma vector for each adventure
# and storing it in a database.

from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from chromadb import errors
from langchain import LLMChain, OpenAI
from langchain.chains.question_answering import load_qa_chain

class Adventure(object):
    def __init__(self, name, path):
        self.name = name
        self.path = path

class Archive(object):
    def __init__(self, db_path: str = "data/chroma", model: OpenAI = None, verbose: bool = False):
        embeddings = OpenAIEmbeddings()
        self.chroma = Chroma(persist_directory=db_path, embedding_function=embeddings)
        if model is None:
            model = OpenAI()
        self.model = model
        self.verbose = verbose
        self.chain = self.__create_chain()

    def __create_chain(self):
        return load_qa_chain(llm=self.model, chain_type="stuff", verbose=self.verbose)

    def save_adventure(self, adventure: Adventure):
        if self._is_adventure_in_archive(adventure):
            return
        # If there are no datapoints, then we can just add the adventure
        texts =  MarkdownTextSplitter().split_documents(TextLoader(adventure.path).load())
        # Add the adventure name to the text
        for text in texts:
            text.metadata["adventure_name"] = adventure.name
        # Save the adventure
        self.chroma.add_documents(texts)
        self.chroma.persist()

    def _is_adventure_in_archive(self, adventure: Adventure):
        try:
            _ = self.chroma.similarity_search(adventure.name,k=1, filter={"adventure_name": adventure.name})
            return True
        except errors.NoDatapointsException:
            return False
        except errors.NoIndexException:
            return False

    def search(self, query: str, adventure: str = None, k: int = 4):
        if adventure is None:
            docs = self.chroma.similarity_search(query, k=k)
        else:
            docs = self.chroma.similarity_search(query, k=k, filter={"adventure_name": adventure})
        
        return self.chain.run(input_documents=docs, question=query)
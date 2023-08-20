import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import os
import logging
import email
from langchain.document_loaders import (WebBaseLoader, AirbyteJSONLoader,
                                        UnstructuredPDFLoader, UnstructuredMarkdownLoader,
                                        TextLoader, UnstructuredHTMLLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from typing import Any, Dict, Iterator, List, Optional, Union
import json
os.environ['OPENAI_API_KEY'] = 'EMPTY'
os.environ['OPENAI_API_BASE'] = 'http://localhost:8000/v1'


class DocumentLoader:
    vector_store: Chroma
    chunk_size = 100
    chunk_overlap = 0

    def __init__(self) -> None:
        self.vector_store = Chroma("vector_store", OpenAIEmbeddings())

    def split_document(self, data: List[Document]):
        # 分割文本，生成 document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        all_splits = text_splitter.split_documents(data)
        self.store_document(all_splits)

    def store_document(self, data: List[Document]):
        if len(data) == 0:
            return
        self.vector_store.add_documents(documents=data)

    def load_dir_files(self, directory: str):
        for root, dirs, files in os.walk(directory):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                self.traverse_all_directories(dir_path)
            for file in files:
                file_path = os.path.join(root, file)
                self.load_file(file_path)

    def load_files(self, path_list: List[str]):
        for path in path_list:
            self.load_file(path)

    def load_file(self, path:  str):
        if path.startswith("http"):
            self.load_url(path)
        else:
            path = self.get_local_path(path)
            if path.endswith(".json"):
                self.load_json(path)
            elif path.endswith(".pdf"):
                self.load_pdf(path)
            elif path.endswith(".md"):
                self.load_md(path)
            elif path.endswith(".html"):
                self.load_html(path)
            else:
                self.load_txt(path)

    def load_url(self, url: str):
        # 加载非结构化数据
        loader = WebBaseLoader(url)
        docs = loader.load()
        self.split_document(docs)

    def load_json(self, path: str):
        loader = AirbyteJSONLoader(path)
        docs = loader.load()
        self.split_document(docs)

    def load_pdf(self, path: str):
        loader = UnstructuredPDFLoader(
            path, mode="elements", strategy="fast",
        )
        docs = loader.load()
        self.split_document(docs)

    def load_md(self, path: str):
        loader = UnstructuredMarkdownLoader(
            path, mode="elements", strategy="fast",
        )
        docs = loader.load()
        self.split_document(docs)

    def load_txt(self, path: str):
        loader = TextLoader(path, encoding="UTF-8")
        docs = loader.load()
        self.split_document(docs)

    def load_html(self, path: str):
        loader = UnstructuredHTMLLoader(
            path, mode="elements", strategy="fast",
        )
        docs = loader.load()
        self.split_document(docs)

    def get_local_path(self, path: str):
        base_path = os.getcwd()
        cur_path = os.path.join(base_path, path)
        return os.path.abspath(cur_path)


class Chat_By_Document:
    chat: ConversationalRetrievalChain
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer")

    def __init__(self, vector_store: Chroma, llm) -> None:
        self.chat = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=self.memory,
            return_source_documents=True,
            return_generated_question=True)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.chat(*args, **kwds)

    def run(self, query: str):
        if query == "":
            return ""
        return self.chat({"question": query})


if __name__ == '__main__':
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, streaming=True)
    store = DocumentLoader()

    # store.load_file(
    #     "https://zh.wikipedia.org/wiki/%E5%89%8D%E7%AB%AF%E5%92%8C%E5%90%8E%E7%AB%AF")
    store.load_dir_files(
        "./documents")
    chat = Chat_By_Document(store.vector_store, llm)
    result = chat({"question": "李白生平"})
    # print(json.dumps({
    #     "answer": result['answer'],
    #     "source_documents": [d.page_content for d in result['source_documents']],
    # }))
    print(f"{result['answer']} {result['source_documents']}")

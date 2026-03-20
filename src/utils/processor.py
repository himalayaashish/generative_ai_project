import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, path: str) -> List[Document]:
        """Notebook 01: Supports PDF, Directory, and Text loading"""
        if os.path.isdir(path):
            loader = DirectoryLoader(path, glob="./*.pdf", loader_cls=PyPDFLoader)
        elif path.endswith('.pdf'):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path)
        return loader.load()

    def split_documents(self, docs: List[Document], strategy: str = "recursive") -> List[Document]:
        """Notebook 02: Splits documents into manageable chunks"""
        if strategy == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n"
            )
        return splitter.split_documents(docs)
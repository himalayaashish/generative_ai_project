# FIX: Use langchain_huggingface instead of deprecated langchain_community.embeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from src.utils.logger import get_logger
logger = get_logger(__name__)
class GPTManager:
    def __init__(self, api_key: str, model_name: str, persist_dir: str):
        # LLM for generating the final answer
        self.llm = ChatOpenAI(model_name=model_name, temperature=0, api_key=api_key)

        # Local Embeddings (Notebook 03): Efficient and cost-effective
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.persist_dir = persist_dir

        # FIX: Memory is now created once on the instance, not inside
        # create_chat_chain(). Previously, memory was re-created on every
        # API call, causing chat history to reset with each request.
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def get_vectorstore(self):
        """Initializes or loads the ChromaDB persistent store."""
        logger.info(f"Loading vectorstore from {self.persist_dir}")
        return Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

    def create_chat_chain(self, retriever):
        """Notebook 06: Implements memory and retrieval logic."""
        # FIX: Reuse self.memory so conversation history is preserved
        # across multiple /query calls within the same server session.
        logger.info("Creating chat chain with memory")
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory
        )
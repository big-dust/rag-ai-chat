import os
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.llms.gemini import Gemini
from llama_index.legacy.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 环境配置
load_dotenv()

class RAGSystem:
    def __init__(self):
        # 初始化LLM
        self.llm = Gemini(
            model_name="models/gemini-1.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            transport="rest"
        )

        # 配置代理
        if os.getenv("PROXY_URL"):
            os.environ['all_proxy'] = os.getenv("PROXY_URL")

        # 配置全局设置
        self._configure_settings()

        # 构建/加载索引
        self.index = self._build_or_load_index()
        self.query_engine = self._create_query_engine()

    def _configure_settings(self):
        """配置全局参数"""
        self.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=1,
            window_metadata_key="window",
            original_text_metadata_key="original_sentence",
        )

        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",
            device="cpu"
        )

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.node_parser = self.node_parser

    def _get_documents_hash(self):
        """计算文档目录的哈希值"""
        hasher = hashlib.sha256()
        for root, _, files in os.walk("./data"):
            for file in files:
                with open(os.path.join(root, file), "rb") as f:
                    hasher.update(f.read())
        return hasher.hexdigest()

    def _build_or_load_index(self):
        """智能构建或加载索引"""
        storage_dir = "./storage"
        hash_file = os.path.join(storage_dir, "doc_hash.sha256")

        # 检查是否需要重建索引
        rebuild = False
        if os.path.exists(storage_dir):
            current_hash = self._get_documents_hash()
            try:
                with open(hash_file, "r") as f:
                    saved_hash = f.read()
                    rebuild = current_hash != saved_hash
            except FileNotFoundError:
                rebuild = True
        else:
            rebuild = True

        if rebuild:
            print("检测到文档变更，正在重建索引...")
            os.makedirs(storage_dir, exist_ok=True)

            # 构建新索引
            documents = SimpleDirectoryReader(input_dir="./data").load_data()
            index = VectorStoreIndex.from_documents(documents)

            # 保存索引和哈希
            index.storage_context.persist(persist_dir=storage_dir)
            with open(hash_file, "w") as f:
                f.write(self._get_documents_hash())
            print(f"索引已更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("加载已有索引...")
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(storage_context)

        return index

    def _create_query_engine(self):
        """创建查询引擎"""
        post_processor = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )

        return self.index.as_query_engine(
            similarity_top_k=3,
            node_postprocessors=[post_processor]
        )

    def query(self, question: str) -> str:
        """执行查询"""
        response = self.query_engine.query(question)
        return str(response).strip()

if __name__ == "__main__":
    # 检查环境配置
    if not os.getenv("GEMINI_API_KEY"):
        print("错误：请先在.env文件中配置GEMINI_API_KEY")
        exit(1)

    # 初始化系统
    print("初始化RAG系统...")
    rag = RAGSystem()

    # 交互界面
    print("\nRAG系统已就绪！输入 'exit' 退出")
    while True:
        try:
            user_input = input("\n您：")
            if user_input.lower() in ["exit", "quit"]:
                break

            response = rag.query(user_input)
            print(f"\nAI：{response}")

        except Exception as e:
            print(f"发生错误：{str(e)}")
            print("请检查：1.网络连接 2.API密钥 3.文档路径")

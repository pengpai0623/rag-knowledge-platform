"""
Day13：Milvus 生产级向量库迁移实战
基于 Milvus 重构向量存储层，保留原有 RAG 全链路逻辑
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# ========== 配置项 ==========
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DOC_PATH = os.path.join(BASE_DIR, "docs", "rag", "向量检索核心原理.md")

# Milvus 连接配置
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "rag_knowledge_base"
TOP_K = 3

# 模型配置（完全沿用原有配置）
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL")
DOUBAO_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3"
EMBEDDING_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v3"

# 初始化 Embedding 模型
embeddings = OpenAIEmbeddings(
    api_key=QWEN_API_KEY,
    base_url=EMBEDDING_URL,
    model=EMBEDDING_MODEL,
    check_embedding_ctx_length=False,
)

# 初始化大语言模型
llm = ChatOpenAI(
    api_key=DOUBAO_API_KEY,
    base_url=DOUBAO_ENDPOINT,
    model=DOUBAO_MODEL,
    temperature=0,
    streaming=False
)

# RAG Prompt 模板
RAG_PROMPT_TEMPLATE = """
你是专业的知识库问答助手，必须严格基于以下提供的上下文内容回答用户问题。
【回答规则】
1. 答案只能来自下方的上下文内容，绝对不可以使用自身知识编造
2. 如果上下文里没有相关信息，请直接回答："抱歉，知识库中没有找到相关内容，无法为您解答。"
3. 回答要准确、简洁、条理清晰，保留关键术语
4. 不要提及"根据上下文"、"从资料中"这类表述
【上下文】
{context}
【用户问题】
{question}
【答案】
"""
from langchain_core.prompts import PromptTemplate
RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def init_vector_store(rebuild: bool = False) -> Milvus:
    """
    初始化 Milvus 向量库
    :param rebuild: True 则删除旧集合重建
    """
    if rebuild:
        try:
            Milvus.drop_collection(
                collection_name=COLLECTION_NAME,
                uri=MILVUS_URI
            )
            print("旧集合已删除")
        except Exception:
            pass

    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        uri=MILVUS_URI,
        # 距离度量：余弦相似度
        metric_type="COSINE",
        # 索引类型：生产首选 HNSW
        index_params={
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        },
        # 检索参数
        search_params={
            "params": {"ef": 50}
        },
        # 自动创建集合
        auto_id=True,
        drop_old=False
    )
    print(f"Milvus 向量库初始化完成，集合：{COLLECTION_NAME}")
    return vector_store


def load_and_split_docs() -> list:
    """加载文档并分片（与 Chroma 版完全一致）"""
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"文档分片完成，共 {len(splits)} 个分片")
    return splits


def batch_ingest(splits: list, vector_store: Milvus):
    """批量入库（接口与 Chroma 完全一致）"""
    ids = vector_store.add_documents(splits)
    print(f"批量入库完成，共写入 {len(ids)} 条向量")
    return ids


def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store: Milvus):
    """
    构建 LCEL RAG 链（与 Chroma 版完全一致）

    数据流向
    【输入】 纯字符串："什么是ANN？"
    │
    ▼
    [ 检索打包 (RunnableLambda) ]
    ├─ 查向量库 → 拿到 List[Document]
    └─ 打包成字典 → { "context": "拼好的文本", "question": "什么是ANN？", "source_documents": [文档列表] }
    │
    ▼
    [ 并行分发 (RunnableParallel) ]
    ├─▶ 分支A：提取 context和question → 填Prompt → 调大模型 → 解析 → 得到 "result"(字符串回答)
    └─▶ 分支B：直接提取 source_documents → 得到 [原始文档列表]
    │
    ▼
    【输出】 合并字典：{ "result": "大模型的回答...", "source_documents": [文档1, 文档2...] }
    """
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": TOP_K, "score_threshold": 0.5}
    )

    def _retrieve_context(query: str):
        docs = retriever.invoke(query)
        return {
            "context": format_docs(docs),
            "question": query,
            "source_documents": docs
        }

    retrieve_step = RunnableLambda(_retrieve_context)
    generate_step = (
        {"context": itemgetter("context"), "question": itemgetter("question")}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    full_chain = retrieve_step | RunnableParallel(
        result=generate_step,
        source_documents=itemgetter("source_documents")
    )
    return full_chain


def ask_question(qa_chain, query: str, show_source: bool = True):
    print(f"\n{'='*60}")
    print(f"用户提问：{query}")
    print("-" * 60)
    result = qa_chain.invoke(query)
    print(f"模型回答：\n{result['result']}")

    if show_source and result.get("source_documents"):
        print("\n" + "-" * 60)
        print(f"引用来源（共 {len(result['source_documents'])} 条）：")
        for idx, doc in enumerate(result["source_documents"], 1):
            print(f"\n[{idx}] 来源：{doc.metadata.get('source', '未知')}")
            print(f"    内容摘要：{doc.page_content[:80]}...")
    print("=" * 60)


if __name__ == "__main__":
    # 1. 初始化 Milvus（首次运行 rebuild=True 重建集合）
    vector_store = init_vector_store(rebuild=True)

    # 2. 文档分片 + 批量入库
    splits = load_and_split_docs()
    batch_ingest(splits, vector_store)

    # 3. 构建 RAG 链并测试
    rag_chain = build_rag_chain(vector_store)

    print("\n" + "#" * 60)
    print("# Milvus RAG 问答测试")
    print("#" * 60)
    ask_question(rag_chain, "什么是近似最近邻（ANN）")
    ask_question(rag_chain, "HNSW 和 IVF_FLAT 有什么区别")
    ask_question(rag_chain, "如何训练一个大语言模型")

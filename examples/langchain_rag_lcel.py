import os
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# ========== 配置项 (保持不变) ==========
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
PERSIST_DIR = os.path.join(BASE_DIR, "vector_db", "chroma_rag_db")
COLLECTION_NAME = "rag_knowledge_base"
TOP_K = 3

DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL")
DOUBAO_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3"
EMBEDDING_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" 
EMBEDDING_MODEL = "text-embedding-v3" 

# ========== 初始化模型 ==========
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=EMBEDDING_URL,
    model=EMBEDDING_MODEL,
    check_embedding_ctx_length=False,
)

llm = ChatOpenAI(
    api_key=DOUBAO_API_KEY,
    base_url=DOUBAO_ENDPOINT,
    model=DOUBAO_MODEL,
    temperature=0,
    streaming=False
)

# ========== 自定义 Prompt ==========
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
RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def load_vector_store() -> Chroma:
    """加载已持久化的本地向量库"""
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"向量库加载完成，当前集合向量数：{vector_store._collection.count()}")
    return vector_store

# 辅助函数：将检索到的 Document 列表拼装成一段纯文本
def format_docs(docs):
    # 手写stuff，更灵活；当然langchain也有create_stuff_documents_chain可使用
    return "\n\n".join(doc.page_content for doc in docs)

def build_lcel_qa_chain(vector_store: Chroma, k: int = TOP_K):
    """
    构建 LCEL 版本的 RAG 链
    数据流：

    输入: {"question": "什么是ANN"}
            │
        并行├──> [RunnableParallel] ──> 分支A ──> retriever查文档 ──> format_docs拼接 ──> Prompt填充 ──> llm生成 ──> StrOutputParser解析
            │
            └──> [RunnableParallel] ──> 分支B ──> retriever查文档 
            
    最终输出: {
        "answer": "近似最近邻（ANN）...",
        "source_documents": [Document(...), Document(...)]
    }
    """
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": 0.5}
    )
    
    # 核心的 RAG 生成链 (输入 context 和 question，输出字符串答案)
    rag_generation_chain = (
        RAG_PROMPT         # {"context": "文档内容", "question": "用户问题"}填充prompt模版，组装上下文
        | llm              # 将prompt发送给大模型，大模型返回一个 AIMessage 对象（包含内容、Token 数等元数据）
        | StrOutputParser() # 把 AIMessage 解析成纯字符串
    )
    
    # 外层包装：并行执行【获取答案 --answer】和【获取原始文档 --source_documents】
    # 输入: {"question": "用户的问题"}
    # 输出: {"answer": "...", "source_documents": [...]}
    full_chain = RunnableParallel(
        answer={
            "context": lambda x: format_docs(retriever.invoke(x["question"])), # 动态检索并格式化, stuff拼装的上下文
            "question": lambda x: x["question"]                                # 透传问题
        } | rag_generation_chain,
        
        # 单独再跑一次检索返回文档 (注意：LangChain 底层有缓存，不会请求两次向量库)
        # TopK召回的结果[Document(...), Document(...)]
        source_documents=lambda x: retriever.invoke(x["question"]) 
    )
    
    return full_chain

def ask_question_lcel(chain, query: str, show_source: bool = True):
    """执行 LCEL 问答并格式化输出"""
    print(f"\n{'='*60}")
    print(f"用户提问：{query}")
    print("-" * 60)

    # 传参必须是字典格式
    result = chain.invoke({"question": query})
    
    print(f"模型回答：\n{result['answer']}")

    if show_source and result.get("source_documents"):
        print("\n" + "-" * 60)
        print(f"引用来源（共 {len(result['source_documents'])} 条）：")
        for idx, doc in enumerate(result["source_documents"], 1):
            print(f"\n[{idx}] 来源：{doc.metadata.get('source', '未知')}")
            print(f"    内容摘要：{doc.page_content[:80]}...")
    print("=" * 60)

if __name__ == "__main__":
    vector_store = load_vector_store()
    
    # 构建 LCEL 链
    lcel_chain = build_lcel_qa_chain(vector_store)
    
    # 测试
    test_query = "什么是近似最近邻（ANN）"
    ask_question_lcel(lcel_chain, test_query)
    
    # 测试兜底
    ask_question_lcel(lcel_chain, "如何训练一个大语言模型？")
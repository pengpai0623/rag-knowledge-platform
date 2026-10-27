"""
Day12：RetrievalQA 全链路 RAG 问答实战
功能：Stuff/Map-Reduce 双策略对比 + 自定义Prompt模板 + 来源引用返回
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ========== 配置项 ==========
# 获取当前脚本 (langchain_rag.py) 所在的目录，即 examples 文件夹
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录（项目根目录），并拼接出 vector_db 的路径
BASE_DIR = os.path.dirname(CURRENT_DIR)
PERSIST_DIR = os.path.join(BASE_DIR, "vector_db", "chroma_rag_db")
COLLECTION_NAME = "rag_knowledge_base"
TOP_K = 3
LLM_MODEL = "gpt-3.5-turbo"

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL")
DOUBAO_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3"
EMBEDDING_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" 
EMBEDDING_MODEL = "text-embedding-v3" 
CHAT_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
CHAT_MODEL = "doubao-lite-240615"

# 初始化 Embedding 模型
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=EMBEDDING_URL,
    model=EMBEDDING_MODEL,
    check_embedding_ctx_length=False,
)

# 初始化大语言模型
llm = ChatOpenAI(
    api_key=DOUBAO_API_KEY,  # 替换为豆包的 ARK API Key
    base_url=DOUBAO_ENDPOINT,  # 火山方舟兼容端点
    model=DOUBAO_MODEL,  # 填写你的豆包模型 ID（如 doubao-pro-32k 或 endpoint ID）
    temperature=0,
    streaming=False
)

# Map-Reduce（分治策略） 专属
# 1. Map 阶段：针对每个切片提取关键信息
map_prompt_template = """你是一个信息提取助手。请根据以下文本片段提取与问题相关的所有关键事实和细节。
如果片段与问题无关，请直接返回“无相关内容”。
【文本片段】
{context}
【问题】
{question}
【提取结果】"""

MAP_PROMPT = PromptTemplate(
    template=map_prompt_template,
    input_variables=["context", "question"]
)

# 2. Combine 阶段：汇总所有提取结果并生成最终答案
combine_prompt_template = """你是一名严谨的知识库问答助手，必须严格基于以下提取的信息回答用户问题。
【回答规则】
1. 只能使用下方的提取内容作答，不得使用外部自身知识
2. 如果提取结果中均无相关信息，请直接回答：“抱歉，知识库中没有找到相关内容，无法为您解答。”
3. 回答要准确、简洁，不要提及“根据提取信息”等过渡词

【提取内容】
{context}

【用户问题】
{question}

【答案】"""

COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template,
    input_variables=["context", "question"]
)

# Stuff（默认策略） & Refine（迭代精炼策略）
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


def build_qa_chain(
    vector_store: Chroma,
    chain_type: str = "stuff",
    return_source_documents: bool = True,
    k: int = TOP_K
) -> RetrievalQA:
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": 0.5}
    )
    
    if chain_type == "map_reduce":
        # 传入映射与合并两种专属模板，避免触发 prompt 字段校验异常
        chain_type_kwargs = {
            "question_prompt": MAP_PROMPT,     # Map 阶段：提取
            "combine_prompt": COMBINE_PROMPT,  # Reduce 阶段：汇总
            "combine_document_variable_name": "context"     # 告诉 Reduce 阶段把结果喂给 {context}
        }
    else:
        chain_type_kwargs = {"prompt": RAG_PROMPT}
        
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=return_source_documents,
        chain_type_kwargs=chain_type_kwargs,
        verbose=False
    )
    return qa_chain

def ask_question(qa_chain: RetrievalQA, query: str, show_source: bool = True):
    """执行问答并格式化输出"""
    print(f"\n{'='*60}")
    print(f"用户提问：{query}")
    print(f"链类型：{getattr(qa_chain, '_chain_type', '未知')}")
    print("-" * 60)

    result = qa_chain.invoke(query)
    print(f"模型回答：\n{result['result']}")

    if show_source and result.get("source_documents"):
        print("\n" + "-" * 60)
        print(f"引用来源（共 {len(result['source_documents'])} 条）：") # 最终送入大模型的分片数量
        for idx, doc in enumerate(result["source_documents"], 1):
            print(f"\n[{idx}] 来源：{doc.metadata.get('source', '未知')}")
            print(f"    内容摘要：{doc.page_content[:80]}...")
    print("=" * 60)


if __name__ == "__main__":
    vector_store = load_vector_store()
    test_query = "什么是近似最近邻（ANN）"

    # 1. Stuff 模式问答（默认推荐）
    print("\n" + "#" * 60)
    print("# 模式1：Stuff 策略（默认，一次拼装全部上下文）")
    print("#" * 60)
    qa_stuff = build_qa_chain(vector_store, chain_type="stuff")
    ask_question(qa_stuff, test_query)

    # 2. Map-Reduce 模式问答
    print("\n" + "#" * 60)
    print("# 模式2：Map-Reduce 策略（分治汇总，适合长文档）")
    print("#" * 60)
    qa_mr = build_qa_chain(vector_store, chain_type="map_reduce")
    ask_question(qa_mr, test_query)

    # 3. 测试知识库外的问题（验证兜底效果）
    print("\n" + "#" * 60)
    print("# 测试：知识库外的问题，验证幻觉抑制")
    print("#" * 60)
    ask_question(qa_stuff, "如何训练一个大语言模型？")
    # docs_with_scores = vector_store.similarity_search_with_score(
    #     "如何训练一个大语言模型？", k=3
    # )
    # print("\n--- 真实得分 ---")
    # for doc, score in docs_with_scores:
    #     print(f"相似度：{1 - score:.4f} | 内容：{doc.page_content[:30]}...")
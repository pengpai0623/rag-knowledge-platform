import errno
import os
import threading
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.error_codes import (
    ERR_KB_DIM_MISMATCH,
    ERR_KB_INDEX_BROKEN,
    ERR_KB_NOT_FOUND,
    ERR_KB_STORAGE,
    ERR_KB_STORAGE_FULL,
    ERR_KB_VERSION_MISMATCH,
)
from app.core.exceptions import KbNotFoundError, KbStorageError, RAGBaseError
from app.services.model_factory import ModelFactory
from utils.logger import get_logger

# 向量库持久化目录：默认与 examples/ 保持同一份库，避免出现两套库
# __file__ = app/services/index_service.py -> 上三级就是项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Docker 部署时用 CHROMA_PERSIST_DIR 指向挂载卷（docker-compose 里
# ./chroma_data:/app/chroma_data 对应 CHROMA_PERSIST_DIR=/app/chroma_data），
# 容器重启/删除重建后向量数据仍在宿主机；本地开发不设该变量，走默认路径。
# （model_factory 在本模块顶部被导入时已执行 load_dotenv，写在 .env 里也生效）
PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR") or os.path.join(
    PROJECT_ROOT, "vector_db", "chroma_rag_db"
)

# 巡检接口扫描 metadata 时的分页大小：只累积统计量、不保留原始行，避免大库一次性拉全量
_INSPECT_PAGE = 1000

logger = get_logger("app.services.index_service")

# knowledge_base_id 生成状态：毫秒时间戳 + 进程内严格递增。
# 为什么不用 uuid/随机数：这个 ID 要写进 MySQL 当主键或业务 ID，单调递增才便于排查和排序。
# 为什么不用数据库自增：MySQL 还没接，且上传链路不该依赖 DB 可用性。
# 单进程部署下唯一性由「时间戳不倒退 + 锁内 +1」保证；多副本部署会撞，届时换 DB/Redis 序列。
_ID_LOCK = threading.Lock()
_last_kb_id = 0


def _identity_fields(meta: dict) -> dict:
    """从分片 metadata 里取「代表整库身份」的字段。

    一个 collection 只有一份文件，所以任意一条分片就能代表整库；上传判定、
    列表、巡检都读这一份，避免各写一遍导致字段漂移（比如某处漏了 kb_id）。

    Chroma 只接受 str/int/float/bool，写进去是 int，读出来仍是 int；
    MySQL 未接入时 kb_id 允许缺省，这里统一成 None，前端不用判 undefined。
    """
    kb_id = meta.get("knowledge_base_id")
    return {
        "source": meta.get("source"),
        "file_sha256": meta.get("file_sha256"),
        "knowledge_base_id": (
            int(kb_id) if isinstance(kb_id, (int, float)) and not isinstance(kb_id, bool) else None
        ),
        "parser_version": meta.get("parser_version"),
        "clean_policy": meta.get("clean_policy"),
        "cleaner_version": meta.get("cleaner_version"),
        "splitter_version": meta.get("splitter_version"),
    }


class IndexService:
    """向量库（Chroma）访问层。

    异常策略（三类，按异常类型分，不按调用位置分）：
    1. 拿不到客户端（依赖缺失 / 目录无权 / 库文件损坏） → _persistent_client 统一 2002；
    2. collection 名不存在（正常业务状态）                → 2001，由 has_collection 或
       _collection 捕获 chromadb.errors.NotFoundError 给出，含并发删除的窗口期；
    3. 其它操作期故障                                    → _translate_chroma 只翻译
       「调用方有补救动作」的几类（2003 维度 / 2004 索引 / 2005 磁盘满 / 2006 版本），
       识别不了的一律裸抛 → 全局兜底 500（固定文案 + 堆栈 + rid）。
    """

    _embed_model = None

    @classmethod
    def get_embedding(cls):
        """单例加载 Embedding 模型。

        必须与库内已有向量同源：现有 vector_db/chroma_rag_db 是 text-embedding-v3（1024 维），
        换成别的后端（如 BGE-small-zh 512 维）会导致写入维度报错、检索结果失真。
        """
        if cls._embed_model is None:
            cls._embed_model = ModelFactory.get_embedding()
        return cls._embed_model

    @classmethod
    def get_vector_store(cls, knowledge_base: str) -> Chroma:
        """获取指定知识库的向量库实例（新建集合时统一用 cosine 距离，与既有库一致）"""
        return Chroma(
            collection_name=knowledge_base,
            embedding_function=cls.get_embedding(),
            persist_directory=PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def add_documents(
        cls,
        knowledge_base: str,
        docs: List[Document],
        knowledge_base_id: Optional[int] = None,
    ) -> int:
        """文档批量入库，返回分片数量。

        knowledge_base_id 在这里统一注入、不在路由层注入：这是全项目唯一的写入口，
        放在这里才能保证「只要进了库，分片一定带着业务 ID」，将来接 MySQL 或换
        其它调用方都不会漏打。
        """
        if knowledge_base_id is not None:
            for doc in docs:
                doc.metadata["knowledge_base_id"] = knowledge_base_id

        start = time.perf_counter()
        with cls._guard(f"写入知识库 {knowledge_base}"):
            vector_store = cls.get_vector_store(knowledge_base)
            vector_store.add_documents(docs)
        # 逐批调 embedding，是上传链路里最慢的一步；单独记耗时才能和解析耗时区分瓶颈
        logger.info(
            "向量写入完成 | kb=%s 分片=%d | %.1fms",
            knowledge_base, len(docs), (time.perf_counter() - start) * 1000,
        )
        return len(docs)

    @classmethod
    def next_knowledge_base_id(cls) -> int:
        """生成一个新的 knowledge_base_id（毫秒时间戳，进程内严格递增）。

        只在「库不存在」时生成；库已存在一律复用库里的值，所以同一份文件重复上传
        不会堆出一串指向同一个 collection 的 ID。
        """
        global _last_kb_id
        with _ID_LOCK:
            _last_kb_id = max(int(time.time() * 1000), _last_kb_id + 1)
            return _last_kb_id

    # ------------------------------------------------------- 异常翻译（唯一出口）
    @classmethod
    def _persistent_client(cls):
        """打开向量库客户端。这是「存储不可用」（2002）唯一的产生点。

        import 失败（依赖缺失 / protobuf 不匹配）与打开失败（目录无权、库文件损坏、
        被其它进程占用）对调用方是同一件事：服务端存储用不了。明细只进日志，
        响应体给固定文案，避免把 protobuf 版本号、绝对路径带出去。
        """
        try:
            import chromadb
            return chromadb.PersistentClient(path=PERSIST_DIR)
        except Exception as e:
            logger.error("向量库不可用：%s", PERSIST_DIR, exc_info=True)
            wrapped = cls._translate_chroma(e, "打开向量库") or KbStorageError(
                "向量库暂时不可用，请稍后重试", code=ERR_KB_STORAGE
            )
            raise wrapped from e

    @staticmethod
    def _translate_chroma(e: Exception, where: str) -> Optional[RAGBaseError]:
        """把可识别的向量库故障翻成业务异常；识别不了返回 None（由调用方裸抛）。

        只翻译「调用方有补救动作」的那几类：
        - 维度不匹配   → 换回原 embedding 模型，或清库重灌
        - 版本不兼容   → 清库重灌
        - 索引/内部错  → 清库重灌
        - 磁盘满       → 清磁盘后重试
        InvalidUUIDError / InvalidArgumentError / DuplicateIDError 属于我们传参写错或
        并发写重，是 bug，保持裸抛——用业务码会把 bug 掩盖成「预期内的业务失败」。
        """
        # 磁盘满没有独立类型，且可能被 chromadb 包成 InternalError，所以先看 message
        text = str(e).lower()
        if (isinstance(e, OSError) and e.errno == errno.ENOSPC) or \
                "disk is full" in text or "no space left" in text:
            return KbStorageError(
                f"磁盘空间不足，写入失败：{where}", code=ERR_KB_STORAGE_FULL
            )

        try:
            from chromadb import errors as chroma_errors
        except ImportError:
            return None  # chromadb 本身导入失败 → 由调用方兜成 2002

        if isinstance(e, chroma_errors.InvalidDimensionException):
            return KbStorageError(
                "向量维度与库内已有数据不一致，请使用原 embedding 模型或清空知识库后重灌",
                code=ERR_KB_DIM_MISMATCH,
            )
        if isinstance(e, chroma_errors.VersionMismatchError):
            return KbStorageError(
                "向量库版本与当前服务不兼容，需要清空知识库后重灌",
                code=ERR_KB_VERSION_MISMATCH,
            )
        if isinstance(e, chroma_errors.InternalError):
            return KbStorageError(
                f"向量索引异常，建议重建知识库：{where}", code=ERR_KB_INDEX_BROKEN
            )
        return None

    @classmethod
    @contextmanager
    def _guard(cls, where: str):
        """向量库操作的统一翻译层。

        未识别异常**不在这里重复打日志**：它会走到 handlers 的兜底 handler，
        那里已经记录了 `未捕获异常 | 类型: 消息` + 完整堆栈 + rid。
        """
        try:
            yield
        except RAGBaseError:
            raise  # 已经翻译过（2001/2002/...），不要二次包装
        except Exception as e:
            wrapped = cls._translate_chroma(e, where)
            if wrapped is None:
                raise
            raise wrapped from e

    @classmethod
    def _collection(cls, client, knowledge_base: str):
        """按名取 collection：不存在 → 2001，其余交给 _translate_chroma / 裸抛。

        用它替代散落的 get_collection：has_collection 返回 True 之后、真正取用时
        collection 可能已被并发删除（窗口期），那时的 NotFoundError 语义仍是「不存在」，
        不该变成 500。
        """
        try:
            return client.get_collection(knowledge_base)
        except Exception as e:
            from chromadb import errors as chroma_errors
            if isinstance(e, chroma_errors.NotFoundError):
                raise KbNotFoundError(
                    f"知识库不存在：{knowledge_base}", code=ERR_KB_NOT_FOUND
                ) from e
            wrapped = cls._translate_chroma(e, knowledge_base)
            if wrapped is None:
                raise
            raise wrapped from e

    # ------------------------------------------------------------ 查询
    @classmethod
    def list_knowledge_bases(cls) -> List[dict]:
        """列出所有知识库。

        约定：一个 Chroma collection 就是一个知识库，collection 名由文件内容决定
        （`doc_<sha16>`），接口里对外就叫 knowledge_base，与 get_vector_store() 一致。

        只读 collection 的统计信息与一条分片的 metadata，不加载 Embedding 模型：
        - source / knowledge_base_id：这个库现在装的是哪份文件、属于哪个业务知识库
        - doc_count  ：恒为 1（有分片）或 0（空库）。一个库只放一份文件，保留该字段
                       是为了不破坏既有调用方
        - chunk_count：该库的向量分片总数（collection.count()）

        单个库读失败不再「跳过并当成 0」：那是运维入口，宁可整页报错，
        也不要给出「库里少了一个」的错觉。
        """
        if not os.path.exists(PERSIST_DIR):
            return []

        client = cls._persistent_client()
        collections = []
        with cls._guard("列出知识库"):
            for item in client.list_collections():
                # chromadb 1.x list_collections() 返回 Collection 对象，旧版返回名称字符串
                name = getattr(item, "name", item)
                if not isinstance(name, str):
                    continue
                collection = client.get_collection(name)
                chunk_count = collection.count()
                # 一个库一份文件：读一条就够，不必全量扫 metadata
                meta = cls._first_metadata(collection) if chunk_count else {}
                collections.append({
                    "name": name,
                    **_identity_fields(meta),
                    "doc_count": 1 if chunk_count else 0,
                    "chunk_count": chunk_count,
                })
        return collections

    @classmethod
    def collection_profile(cls, knowledge_base: str) -> Optional[dict]:
        """读一个 collection 的身份信息，不存在返回 None。

        返回 source / file_sha256 / knowledge_base_id / parser_version / clean_policy /
        cleaner_version / splitter_version / chunk_count。上传流程只靠这一次读取就能判定
        created / skipped / replaced，不用再单独查一次「有没有同内容分片」。

        取一条分片就够：一个 collection 只放一份文件。
        """
        if not cls.has_collection(knowledge_base):
            return None
        client = cls._persistent_client()
        with cls._guard(f"读取 {knowledge_base} 的身份信息"):
            collection = cls._collection(client, knowledge_base)
            chunk_count = collection.count()
            meta = cls._first_metadata(collection) if chunk_count else {}
        return {
            **_identity_fields(meta),
            "chunk_count": chunk_count,
        }

    @staticmethod
    def decide_upload_status(
        profile: Optional[dict],
        *,
        parser_version: str,
        clean_policy: str,
        cleaner_version: str,
        splitter_version: str,
    ) -> Tuple[str, str]:
        """判定本次上传该 created / skipped / replaced，返回 (状态, 原因)。

        collection 名由内容 sha 决定，所以「库存在」本身就等价于「同一份内容」；
        剩下只需要比四个策略维度（解析 / 清洗策略 / 清洗实现代数 / 切分）：

        - 库不存在 / 空库   → created（空库是上次写入失败的残留，必须重灌，
                                      否则会永久 skipped 而库里没有数据）
        - 四项全一致        → skipped（不解析、不 embed，直接返回库里的 kb_id）
        - 任一不一致        → replaced（复用 kb_id，整库重建）

        clean_policy 与 cleaner_version 必须分开比：前者是用户选的档位（full/safe/raw），
        后者是清洗代码的代数。只改 cleaner 实现不改档位名时，只看 clean_policy 会误判
        成「同策略」而跳过，新清洗结果永远进不了库。

        缺字段的老数据（v3 之前的解析、没有 splitter_version / cleaner_version）
        一律判 replaced：宁可多算一次 embedding，也不要静默沿用旧策略的切片。
        """
        if not profile or not profile.get("chunk_count"):
            return "created", "首次入库"

        changes = []
        for label, actual, expected in (
            ("parser_version", profile.get("parser_version"), parser_version),
            ("clean_policy", profile.get("clean_policy"), clean_policy),
            ("cleaner_version", profile.get("cleaner_version"), cleaner_version),
            ("splitter_version", profile.get("splitter_version"), splitter_version),
        ):
            if actual != expected:
                changes.append(f"{label} {actual or '<未标记>'} → {expected}")
        if changes:
            return "replaced", "；".join(changes)
        return "skipped", "同内容同策略已入库"

    @classmethod
    def collections_with_source(cls, source: str, exclude: str = "") -> List[dict]:
        """列出与 source 同文件名、但不是 exclude 的其它库（历史版本库提示）。

        内容一改 sha 就变，库名和 kb_id 都跟着变，旧库没人认领就成了孤儿。
        这里**只做提示、不做删除**：按文件名自动删会误伤——两个部门各有一份
        `报告.pdf` 时，删掉的是别人的库。真正的清理由业务侧调 DELETE 接口完成
        （MySQL 接入后可以按 doc_key + revision 做对账）。
        """
        if not source or not os.path.exists(PERSIST_DIR):
            return []

        client = cls._persistent_client()
        found: List[dict] = []
        with cls._guard(f"查找与 {source} 同名的其它知识库"):
            for item in client.list_collections():
                name = getattr(item, "name", item)
                if not isinstance(name, str) or name == exclude:
                    continue
                collection = client.get_collection(name)
                chunk_count = collection.count()
                if not chunk_count:
                    continue
                meta = cls._first_metadata(collection)
                if (meta.get("source") or "") != source:
                    continue
                found.append({
                    "name": name,
                    "knowledge_base_id": _identity_fields(meta)["knowledge_base_id"],
                    "chunk_count": chunk_count,
                })
        return found

    @classmethod
    def update_source(cls, knowledge_base: str, source: str) -> int:
        """把库内所有分片的 `source` 刷成新的文件名，返回受影响分片数。

        用途：同一份内容改名重传。上传判定只比「内容 + 策略」，源文件名不参与，
        所以改名重传会走 skipped（不解析、不重算向量）；但如果不顺手刷 source，
        引用展示和「按来源删除」会一直停在第一次入库时的名字。

        只改 metadata 里这一个字段：文件没变、向量没变、kb_id 与各版本号都不动，
        所以不是「重灌」，也不该触发 replaced。

        Chroma 的 update 传完整 metadata（读出来改一个键再整体写回），
        不依赖它对「部分字段是替换还是合并」的实现细节。
        """
        if not source or not cls.has_collection(knowledge_base):
            return 0
        client = cls._persistent_client()
        with cls._guard(f"刷新 {knowledge_base} 的来源名"):
            collection = cls._collection(client, knowledge_base)
            page = collection.get(include=["metadatas"])
            ids = page.get("ids") or []
            metas = page.get("metadatas") or []
            if not ids:
                return 0
            new_metas = []
            for meta in metas:
                item = dict(meta or {})
                item["source"] = source
                new_metas.append(item)
            collection.update(ids=ids, metadatas=new_metas)
        logger.info("来源名已刷新 | kb=%s 分片=%d source=%s", knowledge_base, len(ids), source)
        return len(ids)

    @staticmethod
    def _first_metadata(collection) -> dict:
        """取一条分片的 metadata 代表整库（一个 collection 只有一份文件）。

        list_knowledge_bases / collection_profile / collections_with_source 都读它：
        各自拼一遍的话，迟早出现「列表里有 kb_id、巡检里没有」这种漂移。
        """
        page = collection.get(limit=1, include=["metadatas"])
        metas = page.get("metadatas") or []
        return (metas[0] if metas else None) or {}

    @classmethod
    def inspect_collections(cls, knowledge_base: str = "") -> List[dict]:
        """巡检向量库：列出 collection 及其内部的来源（source）分布。

        与 list_knowledge_bases 的分工：
        - list_knowledge_bases 是业务视图（库名 / 文档数 / 分片数）；
        - inspect_collections 是排障视图（collection 级 metadata + 每个 source 的分片数
          与解析版本），用来回答「这个库里到底有哪些文档、还有没有 v1 老数据残留」。

        只读 metadata、不加载 Embedding；按 _INSPECT_PAGE 条一页滚动扫描，只累积统计量，
        内存占用与分片总数无关（代价是耗时随分片数线性增长，属排障接口的预期成本）。
        parser_version 缺失的老数据显示为 "<未标记>"，source 缺失显示为 "<未知来源>"。
        """
        if not os.path.exists(PERSIST_DIR):
            return []

        client = cls._persistent_client()
        result: List[dict] = []
        with cls._guard("巡检向量库"):
            for item in client.list_collections():
                # chromadb 1.x 返回 Collection 对象，旧版返回名称字符串
                name = getattr(item, "name", item)
                if not isinstance(name, str):
                    continue
                if knowledge_base and name != knowledge_base:
                    continue
                collection = client.get_collection(name)

                stats: dict = {}
                offset = 0
                while True:
                    page = collection.get(
                        include=["metadatas"], limit=_INSPECT_PAGE, offset=offset
                    )
                    ids = page.get("ids") or []
                    if not ids:
                        break
                    for meta in page.get("metadatas") or []:
                        meta = meta or {}
                        source = meta.get("source") or "<未知来源>"
                        entry = stats.setdefault(
                            source,
                            {"source": source, "chunk_count": 0, "parser_versions": set()},
                        )
                        entry["chunk_count"] += 1
                        entry["parser_versions"].add(meta.get("parser_version") or "<未标记>")
                    offset += len(ids)
                    if len(ids) < _INSPECT_PAGE:
                        break

                result.append({
                    "name": name,
                    "chunk_count": collection.count(),
                    "doc_count": len(stats),
                    "metadata": {str(k): v for k, v in (collection.metadata or {}).items()},
                    "sources": [
                        {
                            "source": entry["source"],
                            "chunk_count": entry["chunk_count"],
                            "parser_versions": sorted(entry["parser_versions"]),
                        }
                        for entry in sorted(
                            stats.values(), key=lambda s: (-s["chunk_count"], s["source"])
                        )
                    ],
                })
        return result

    @classmethod
    def has_collection(cls, knowledge_base: str) -> bool:
        """知识库（collection）是否存在。只查元数据，不加载 Embedding 模型。

        返回 False 只有一种含义：collection 列表里确实没有它（含首次部署、目录不存在）。
        向量库读不出来时抛业务异常，不再吞成 False——否则故障会伪装成
        2001「知识库不存在」，前端和日志都看不出真相。
        """
        if not os.path.exists(PERSIST_DIR):
            return False
        client = cls._persistent_client()
        with cls._guard("读取知识库列表"):
            names = [getattr(c, "name", c) for c in client.list_collections()]
        return knowledge_base in names

    @classmethod
    def delete_collection(cls, knowledge_base: str) -> dict:
        """删除整个 collection，返回 {deleted, knowledge_base_id, chunk_count}。

        为什么要把 kb_id 带回来：MySQL 接入后，「删向量库」必须连带删掉业务侧的
        knowledge_base 记录，否则业务表里会留下指向空库的悬空行。调用方拿这个
        kb_id 去删记录，顺序必须是「先删 Chroma、再删 MySQL」——
        Chroma 删除是幂等的，重试无害；反过来先删 MySQL 记录、Chroma 却失败了，
        就会留下查不到归属的孤儿向量，检索还能命中、业务侧却不知道它属于谁。

        chunk_count 用于响应里如实上报「删掉了多少分片」（原来只能报 collection 个数）。
        """
        if not cls.has_collection(knowledge_base):
            return {"deleted": False, "knowledge_base_id": None, "chunk_count": 0}

        client = cls._persistent_client()
        with cls._guard(f"删除知识库 {knowledge_base}"):
            collection = cls._collection(client, knowledge_base)
            chunk_count = collection.count()
            meta = cls._first_metadata(collection) if chunk_count else {}
            kb_id = _identity_fields(meta)["knowledge_base_id"]
            client.delete_collection(knowledge_base)
        # 破坏性操作留痕：事后追「这个库是谁什么时候清掉的」只能靠这行，
        # kb_id 一并记下，MySQL 接入后据此对账（业务表里不该再有这个 ID）
        logger.warning(
            "整库删除 | kb=%s kb_id=%s 分片=%d", knowledge_base, kb_id, chunk_count
        )
        return {"deleted": True, "knowledge_base_id": kb_id, "chunk_count": chunk_count}

    @classmethod
    def delete_by_source(cls, knowledge_base: str, source: str) -> int:
        """按来源文件名删除该文档的全部向量分片，返回实际删除数量。

        - 知识库不存在 → KbNotFoundError（code 2001）
        - 存储层故障   → KbStorageError（2002/2003/2004/2005/2006），识别不了的落 500
        - 匹配不到分片 → 返回 0，由调用方如实上报，不谎报成功

        先按 source 取 id 再按 id 删：返回值就是「真正命中的分片数」，
        不依赖 collection.count() 前后差（那是全库记录数，并发下会串）。
        """
        if not cls.has_collection(knowledge_base):
            raise KbNotFoundError(f"知识库不存在：{knowledge_base}", code=ERR_KB_NOT_FOUND)

        client = cls._persistent_client()
        with cls._guard(f"删除 {knowledge_base}/{source} 的分片"):
            collection = cls._collection(client, knowledge_base)
            # include=[] 只取 id：不需要 document/metadata，大文档能省一大段序列化
            ids = list(collection.get(where={"source": {"$eq": source}}, include=[])["ids"] or [])
            if ids:
                collection.delete(ids=ids)
        # 同样留痕：deleted=0 是「名字对不上」的常见现象，有这行才不用猜
        logger.info("按来源删除完成 | kb=%s source=%s 分片=%d", knowledge_base, source, len(ids))
        return len(ids)

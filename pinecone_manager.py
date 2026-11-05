"""Менеджер чтения/записи в векторную базу Pinecone.

Умная память: перед записью считается косинусное сходство нового вектора
с уже сохранёнными. Результат записи -- словарь с полем ``action``:

* ``inserted`` -- информация новая (сходство ниже порога), запись добавлена;
* ``updated``  -- похожая запись уже есть (сходство >= порога), обновлён
  существующий слот, дубликат не создаётся;
* ``skipped``  -- практически точный повтор (сходство >= SKIP-порога), запись
  не меняется.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

logger = logging.getLogger("pinecone_manager")

# ---------------------------------------------------------------------------
# Глобальные настройки порогов косинусного сходства (можно задать в .env).
# Чем ближе значение к 1, тем больше тексты похожи по смыслу.
# ---------------------------------------------------------------------------
# >= этого значения запись считается дубликатом/вариацией -> 'updated'
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
# >= этого значения запись -- почти точный повтор -> 'skipped'
SIMILARITY_SKIP_THRESHOLD = float(os.getenv("SIMILARITY_SKIP_THRESHOLD", "0.97"))

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class PineconeManager:
    """Класс для управления операциями с векторной базой данных Pinecone."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        dimension: Optional[int] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        skip_threshold: Optional[float] = None,
        create_index_if_missing: bool = True,
        index: Any = None,
        openai_client: Any = None,
    ):
        """
        Args:
            api_key: ключ Pinecone (иначе PINECONE_API_KEY из .env)
            index_name: имя индекса (иначе PINECONE_INDEX_NAME)
            openai_api_key: ключ OpenAI (иначе OPENAI_API_KEY)
            openai_base_url: базовый URL API, например прокси (иначе OPENAI_BASE_URL)
            embedding_model: модель эмбеддингов (иначе OPENAI_EMBEDDING_MODEL)
            dimension: размерность векторов (по умолчанию определяется по модели)
            cloud / region: где создавать serverless-индекс
            similarity_threshold / skip_threshold: пороги, см. модуль
            create_index_if_missing: создать индекс, если его ещё нет
            index / openai_client: готовые объекты (для тестов без сети)
        """
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        self.embedding_model = embedding_model or os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.dimension = dimension or EMBEDDING_DIMENSIONS.get(self.embedding_model, 1536)
        self.cloud = cloud or os.getenv("PINECONE_CLOUD", "aws")
        self.region = region or os.getenv("PINECONE_REGION", "us-east-1")
        self.similarity_threshold = (
            SIMILARITY_THRESHOLD if similarity_threshold is None else similarity_threshold
        )
        self.skip_threshold = SIMILARITY_SKIP_THRESHOLD if skip_threshold is None else skip_threshold

        # --- OpenAI (эмбеддинги), с поддержкой OPENAI_BASE_URL ---
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL") or None
        if openai_client is not None:
            self.openai_client = openai_client
        else:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY не найден. Укажите в параметрах или в .env.")
            kwargs: Dict[str, Any] = {"api_key": self.openai_api_key}
            if self.openai_base_url:
                kwargs["base_url"] = self.openai_base_url
            self.openai_client = OpenAI(**kwargs)

        # --- Pinecone ---
        if index is not None:
            self.pc = None
            self.index = index
            return

        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY не найден. Укажите в параметрах или в .env.")
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME не найден. Укажите в параметрах или в .env.")

        self.pc = Pinecone(api_key=self.api_key)
        if create_index_if_missing:
            self.ensure_index()
        self.index = self.pc.Index(self.index_name)

    # ------------------------------------------------------------------
    # Индекс
    # ------------------------------------------------------------------
    def ensure_index(self) -> None:
        """Создаёт serverless-индекс (метрика cosine), если его нет."""
        existing = {i["name"] if isinstance(i, dict) else i.name for i in self.pc.list_indexes()}
        if self.index_name in existing:
            return
        logger.info("Индекс %s не найден, создаю...", self.index_name)
        self.pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=self.cloud, region=self.region),
        )
        while not self.pc.describe_index(self.index_name).status["ready"]:
            time.sleep(1)

    def delete_index(self) -> None:
        """Полностью удаляет индекс (после этого менеджером пользоваться нельзя)."""
        self.pc.delete_index(self.index_name)

    # ------------------------------------------------------------------
    # Эмбеддинги
    # ------------------------------------------------------------------
    def create_embedding(self, text: str) -> List[float]:
        """Создаёт эмбеддинг текста через OpenAI-совместимый API."""
        resp = self.openai_client.embeddings.create(model=self.embedding_model, input=text)
        return list(resp.data[0].embedding)

    # ------------------------------------------------------------------
    # Проверка сходства (ядро «умной памяти»)
    # ------------------------------------------------------------------
    def _check_similarity(
        self,
        vector: List[float],
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Ищет самый похожий сохранённый вектор (top-1).

        Индекс создан с метрикой cosine, поэтому ``score`` от Pinecone --
        это и есть косинусное сходство. Возвращает {'id', 'score', 'metadata'}
        или None, если в базе ничего нет.
        """
        res = self.index.query(
            vector=vector,
            top_k=1,
            include_metadata=True,
            namespace=namespace,
            filter=filter,
        )
        matches = res["matches"] if isinstance(res, dict) else res.matches
        if not matches:
            return None
        m = matches[0]
        get = (lambda k: m[k]) if isinstance(m, dict) else (lambda k: getattr(m, k))
        return {"id": get("id"), "score": float(get("score")), "metadata": get("metadata") or {}}

    # ------------------------------------------------------------------
    # Запись
    # ------------------------------------------------------------------
    def upsert_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        check_similarity: bool = True,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Запись вектора с проверкой косинусного сходства.

        Returns:
            {'action': 'inserted'|'updated'|'skipped',
             'similarity_score': float|None, 'existing_id': str|None, 'id': str}
        """
        result: Dict[str, Any] = {
            "action": "inserted",
            "similarity_score": None,
            "existing_id": None,
            "id": vector_id,
        }
        metadata = metadata or {}

        if check_similarity:
            similar = self._check_similarity(vector, namespace=namespace, filter=filter)
            if similar is not None:
                result["similarity_score"] = similar["score"]
                if similar["score"] >= self.skip_threshold:
                    # почти точный повтор -- ничего не пишем
                    result.update(action="skipped", existing_id=similar["id"], id=similar["id"])
                    return result
                if similar["score"] >= self.similarity_threshold:
                    # вариация -- обновляем существующий слот, а не плодим дубль
                    result.update(action="updated", existing_id=similar["id"], id=similar["id"])
                    self.index.upsert(
                        vectors=[{"id": similar["id"], "values": vector, "metadata": metadata}],
                        namespace=namespace,
                    )
                    return result

        self.index.upsert(
            vectors=[{"id": vector_id, "values": vector, "metadata": metadata}],
            namespace=namespace,
        )
        return result

    def upsert_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        namespace: str = "",
        check_similarity: bool = True,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Запись документа: текст -> эмбеддинг -> upsert (с проверкой дубликатов)."""
        vector = self.create_embedding(text)
        meta = dict(metadata or {})
        meta["text"] = text
        result = self.upsert_vector(
            doc_id or str(uuid.uuid4()),
            vector,
            meta,
            namespace=namespace,
            check_similarity=check_similarity,
            filter=filter,
        )
        logger.info(
            "action: %s | similarity: %s | id: %s",
            result["action"],
            None if result["similarity_score"] is None else round(result["similarity_score"], 4),
            result["id"],
        )
        return result

    def upsert_documents(self, documents: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Запись нескольких документов: [{'text':..., 'metadata':..., 'id':...}, ...]."""
        return [
            self.upsert_document(d["text"], d.get("metadata"), d.get("id"), **kwargs)
            for d in documents
        ]

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------
    def query_by_vector(
        self,
        vector: List[float],
        top_k: int = 5,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        res = self.index.query(
            vector=vector, top_k=top_k, include_metadata=True, namespace=namespace, filter=filter
        )
        matches = res["matches"] if isinstance(res, dict) else res.matches
        out = []
        for m in matches:
            g = (lambda k, m=m: m[k]) if isinstance(m, dict) else (lambda k, m=m: getattr(m, k))
            out.append({"id": g("id"), "score": float(g("score")), "metadata": g("metadata") or {}})
        return out

    def query_by_text(self, text: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        return self.query_by_vector(self.create_embedding(text), top_k=top_k, **kwargs)

    def fetch_vectors(self, ids: List[str], namespace: str = "") -> Dict[str, Any]:
        return self.index.fetch(ids=ids, namespace=namespace)

    # ------------------------------------------------------------------
    # Обновление / удаление / статистика
    # ------------------------------------------------------------------
    def update_metadata(self, vector_id: str, metadata: Dict[str, Any], namespace: str = "") -> None:
        self.index.update(id=vector_id, set_metadata=metadata, namespace=namespace)

    def delete(self, ids: List[str], namespace: str = "") -> None:
        self.index.delete(ids=ids, namespace=namespace)

    def delete_by_filter(self, filter: Dict[str, Any], namespace: str = "") -> None:
        self.index.delete(filter=filter, namespace=namespace)

    def delete_all(self, namespace: str = "") -> None:
        """Удаляет все векторы в namespace (если namespace ещё пуст -- не ошибка)."""
        try:
            self.index.delete(delete_all=True, namespace=namespace)
        except Exception as e:  # у пустого namespace Pinecone отвечает 404
            if "404" not in str(e) and "not found" not in str(e).lower():
                raise

    def describe_index_stats(self) -> Any:
        return self.index.describe_index_stats()


# ---------------------------------------------------------------------------
# Ручной тест: python pinecone_manager.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    mgr = PineconeManager()
    print(f"[OK] Подключение к индексу '{mgr.index_name}' установлено")
    print(f"     base_url={mgr.openai_base_url or 'по умолчанию'}, "
          f"model={mgr.embedding_model}, порог={mgr.similarity_threshold}")
    print("     статистика:", mgr.describe_index_stats())

    NS = "manual-test"
    mgr.delete_all(namespace=NS)

    phrases = [
        "Хочу на Марс",           # ожидаем inserted
        "Я полечу на Марс",       # ожидаем updated / skipped
        "Я люблю пиццу",          # ожидаем inserted
    ]
    for p in phrases:
        r = mgr.upsert_document(p, namespace=NS)
        time.sleep(5)  # Pinecone serverless -- eventual consistency
        print(f"{p!r:28} -> action: {r['action']}, similarity: {r['similarity_score']}")

    print("Поиск по тексту 'космос':")
    for m in mgr.query_by_text("космос", top_k=3, namespace=NS):
        print(f"  {m['score']:.3f}  {m['metadata'].get('text')}")

    mgr.delete_all(namespace=NS)
    print("[OK] Тест завершён, тестовые данные удалены")

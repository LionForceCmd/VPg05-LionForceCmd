"""Офлайн-проверка логики дедупликации (без Pinecone/OpenAI): python test_offline.py"""
import math
from types import SimpleNamespace

from pinecone_manager import PineconeManager


class FakeIndex:
    """Мини-индекс с метрикой cosine."""
    def __init__(self):
        self.data = {}

    def upsert(self, vectors, namespace=""):
        for v in vectors:
            self.data[v["id"]] = v

    def query(self, vector, top_k, include_metadata, namespace="", filter=None):
        def cos(a, b):
            return sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))
        rows = [(cos(vector, v["values"]), v) for v in self.data.values()
                if not filter or all(v["metadata"].get(k) == x for k, x in filter.items())]
        rows.sort(key=lambda r: -r[0])
        return {"matches": [{"id": v["id"], "score": s, "metadata": v["metadata"]}
                            for s, v in rows[:top_k]]}


CONCEPTS = ["марс", "полет", "пицца", "привет"]


class FakeOpenAI:
    """Эмбеддинг = наличие понятий в тексте (так 'хочу на марс' ~ 'я полечу на марс')."""
    class embeddings:
        @staticmethod
        def create(model, input):
            t = input.lower().replace("полечу", "полет").replace("лететь", "полет")
            vec = [1.0 if c in t else 0.0 for c in CONCEPTS] + [0.01]
            vec.append(0.5 if "хочу" in t else 0.0)  # лёгкое отличие формулировки
            return SimpleNamespace(data=[SimpleNamespace(embedding=vec)])


m = PineconeManager(index=FakeIndex(), openai_client=FakeOpenAI(), similarity_threshold=0.6)
steps = [("Привет", "inserted"), ("Хочу на Марс", "inserted"),
         ("Я полечу на Марс", "updated"), ("Я полечу на Марс", "skipped"),
         ("Я люблю пиццу", "inserted")]
for text, expected in steps:
    r = m.upsert_document(text)
    print(f"{text!r:22} action: {r['action']:9} similarity: {r['similarity_score']}")
    assert r["action"] == expected, (text, r)
assert len(m.index.data) == 3
print("OK: дубликаты фильтруются, записей в базе:", len(m.index.data))

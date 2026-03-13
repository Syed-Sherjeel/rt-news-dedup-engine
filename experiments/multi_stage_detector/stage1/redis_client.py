import redis
import numpy as np
from redis.commands.search.field import VectorField, NumericField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from constants import *
from models import Article

class RedisNewsStore:
    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT):
        self.client = redis.Redis(host=host, port=port, decode_responses=False)
        self._ensure_index()

    def _ensure_index(self):
        try:
            self.client.ft(INDEX_NAME).info()
            print(f"[redis] index '{INDEX_NAME}' already exists")
        except Exception:
            schema = (
                TextField("$.source",       as_name="source"),
                NumericField("$.published_at", as_name="published_at"),
                VectorField(
                    "$.vector",
                    "HNSW",
                    {
                        "TYPE":            "FLOAT32",
                        "DIM":             VECTOR_DIM,
                        "DISTANCE_METRIC": "COSINE",
                        "M":               16,       # HNSW graph connectivity
                        "EF_CONSTRUCTION": 200,      # build-time accuracy vs speed
                    },
                    as_name="vector",
                ),
            )
            self.client.ft(INDEX_NAME).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[KEY_PREFIX], index_type=IndexType.JSON
                ),
            )
            print(f"[redis] index '{INDEX_NAME}' created")

    def store(self, article: Article, vector: np.ndarray):
        key = f"{KEY_PREFIX}{article.article_id}"
        payload = {
            "article_id":   article.article_id,
            "text":         article.text,
            "source":       article.source,
            "published_at": article.published_at,
            "vector":       vector.tolist(),
        }
        pipe = self.client.pipeline()
        pipe.json().set(key, "$", payload)
        pipe.expire(key, TTL_SECONDS)
        pipe.execute()

        
    def search(self, vector: np.ndarray, k: int = TOP_K) -> list[dict]:
            """
            ANN search across all articles in Redis (no time window filter).
            Returns list of {article_id, score, text, source}.
            """
            q = (
                Query(f"*=>[KNN {k} @vector $vec AS score]")
                .sort_by("score")
                .return_fields("article_id", "text", "source", "score")
                .paging(0, k)
                .dialect(2)
            )
            params = {"vec": vector.tobytes()}
            results = self.client.ft(INDEX_NAME).search(q, query_params=params)

            hits = []
            for doc in results.docs:
                score = 1.0 - float(doc.score)
                hits.append({
                    "article_id": doc.article_id,
                    "text":       doc.text,
                    "source":     doc.source,
                    "score":      score,
                })
            return hits

REDIS_HOST       = "localhost"
REDIS_PORT       = 6379
INDEX_NAME       = "news_index"
KEY_PREFIX       = "article:"
TTL_SECONDS      = 3600          # 1 hour window
VECTOR_DIM       = 384
TOP_K            = 5

THRESHOLD_LOW    = 0.75          # below → not duplicate
THRESHOLD_HIGH   = 0.92          # above → duplicate (high confidence)
                                 # between → Stage 2 candidate
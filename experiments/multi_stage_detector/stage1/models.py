import uuid

from dataclasses import dataclass, field



@dataclass
class Article:
    text: str
    source: str
    article_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    published_at: int = field(default_factory=lambda: int(time.time()))

@dataclass
class SearchResult:
    article_id: str
    score: float
    text: str
    source: str
    verdict: str                 # NOT_DUPLICATE | DUPLICATE | STAGE2_CANDIDATE
from dataclasses import dataclass
from typing import Optional


@dataclass
class Post:
    url: str
    date: str
    title: str
    snippet: str
    gallery: Optional[str] = None

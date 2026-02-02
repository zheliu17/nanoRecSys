from pydantic import BaseModel
from typing import List, Optional


class RecommendRequest(BaseModel):
    user_id: int
    k: int = 10
    explain: bool = False
    include_history: bool = False


class RecommendResponse(BaseModel):
    movie_ids: List[int]
    scores: List[float]
    explanations: Optional[List[str]] = None
    debug_timing: Optional[dict] = None
    history: Optional[List[int]] = None

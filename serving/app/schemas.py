# Copyright (c) 2026 Zhe Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: int = Field(
        ..., description="ID of the user to get recommendations for", example=123
    )  # type: ignore
    k: int = Field(10, description="Number of recommendations to return", example=10)  # type: ignore
    explain: bool = Field(
        False, description="Whether to include explanations for each recommendation"
    )
    include_history: bool = Field(
        False,
        description="Whether to include the user's interaction history in the response",
    )


class RecommendResponse(BaseModel):
    movie_ids: List[int] = Field(..., description="List of recommended movie IDs")
    scores: List[float] = Field(
        ..., description="Corresponding relevance scores for each recommended item"
    )
    explanations: list[str] | None = Field(
        None, description="Optional human-readable explanations for recommendations"
    )
    debug_timing: dict | None = Field(
        None, description="Optional debug timing information (for development)"
    )
    history: list[int] | None = Field(
        None,
        description="Optional user history included when `include_history` is true",
    )

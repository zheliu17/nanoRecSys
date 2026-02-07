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

from typing import List, Optional

from pydantic import BaseModel


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

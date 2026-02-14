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

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .schemas import RecommendRequest, RecommendResponse
from .service import RecommendationService

logger = logging.getLogger(__name__)

service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    logger.info("Initializing Recommendation Service...")
    service = RecommendationService()
    yield
    logger.info("Shutting down...")


app = FastAPI(title="NanoRecSys Serving", lifespan=lifespan)


@app.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Get movie recommendations",
    description=(
        "Return top-k movie recommendations for a user. "
        "Set `explain=true` to include human-readable explanations, "
        "and `include_history=true` to include the user's recent history in the response."
    ),
)
def recommend(request: RecommendRequest):
    """Get recommendations for a user.

    - `user_id`: numeric id of the user
    - `k`: number of items to return
    - `explain`: include explanations per item
    - `include_history`: include user's history in the response
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = service.get_recommendations(
            user_id=request.user_id,
            k=request.k,
            explain=request.explain,
            include_history=request.include_history,
        )
        return result
    except Exception as e:
        logger.exception("Error processing recommendation request")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}

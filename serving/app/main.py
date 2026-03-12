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
from fastapi.responses import ORJSONResponse

from .exceptions import (
    ArtifactMissingError,
    InvalidRecommendationRequestError,
    ServiceNotReadyError,
    UserNotFoundError,
)
from .schemas import RecommendRequest, RecommendResponse
from .service import RecommendationService

logger = logging.getLogger(__name__)

service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    import concurrent.futures

    loop = asyncio.get_running_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))

    global service
    logger.info("Initializing Recommendation Service...")
    service = RecommendationService()
    await service.post_init()
    logger.info("Recommendation Service initialized. ready=%s", service.ready)
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="NanoRecSys Serving",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)


@app.post(
    "/recommend",
    response_model=RecommendResponse,
    response_model_exclude_none=True,
    summary="Get movie recommendations",
    description=(
        "Return top-k movie recommendations for a user. "
        "Set `explain=true` to include human-readable explanations, "
        "and `include_history=true` to include the user's recent history in the response."
    ),
)
async def recommend(request: RecommendRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await service.get_recommendations(
            user_id=request.user_id,
            k=request.k,
            explain=request.explain,
            include_history=request.include_history,
        )
        return result
    except InvalidRecommendationRequestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except UserNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ServiceNotReadyError, ArtifactMissingError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logger.exception("Error processing recommendation request")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
def health():
    # Backward-compatible alias
    return {"status": "ok"}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    status = service.get_status()
    if not status["ready"]:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "errors": status["errors"],
                "warnings": status["warnings"],
            },
        )
    return status


@app.get("/debug/status")
def debug_status():
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return service.get_status()

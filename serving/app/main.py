import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
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


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
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

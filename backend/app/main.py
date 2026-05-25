"""FastAPI entry point for the CognitiveScreen platform."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router

app = FastAPI(
    title="Temple CognitiveScreen",
    description=(
        "Multi-disorder EEG classification platform.\n\n"
        "HAMD-Net hybrid attention model · MNE preprocessing · "
        "FHIR R4 reports · live EEG streaming."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "service": "Temple CognitiveScreen",
        "docs": "/docs",
        "endpoints": [
            "/api/health",
            "/api/model/info",
            "/api/inference/upload",
            "/api/inference/synthetic",
            "/api/inference/{id}",
            "/api/inference/{id}/fhir",
            "/api/datasets",
            "/api/train/start",
            "/api/train/status",
            "/api/train/stream",
            "/api/ws/live (WebSocket)",
        ],
    }

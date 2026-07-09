from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.db.database import get_db
from app.core.config import settings
from app.api.v1.predictions import router as predictions_router
from app.api.v1.auth import router as auth_router

app = FastAPI(
    title="Fraud & Credit Risk Intelligence Platform",
    description="Real-time fraud scoring API with explainable ML and GenAI investigation reports.",
    version="0.1.0",
)

# Allows the React frontend (running on a different port) to call this API.
# In production this would be locked to the actual deployed frontend domain,
# not left wide open with allow_origins=["*"].
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions_router)
app.include_router(auth_router)


@app.get("/")
async def root():
    return {"message": "Fraud Detection Platform API", "environment": settings.ENVIRONMENT}


@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Verifies the API process is up AND can reach Postgres.
    A production health check that only checks 'is the process alive'
    is nearly useless — you want to know if its dependencies are healthy too.
    """
    result = await db.execute(text("SELECT 1"))
    db_ok = result.scalar() == 1
    return {"status": "ok" if db_ok else "degraded", "database": db_ok}
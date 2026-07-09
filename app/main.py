from fastapi import FastAPI, Depends
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

app.include_router(predictions_router)
app.include_router(auth_router)


@app.get("/")
async def root():
    return {"message": "Fraud Detection Platform API", "environment": settings.ENVIRONMENT}


@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("SELECT 1"))
    db_ok = result.scalar() == 1
    return {"status": "ok" if db_ok else "degraded", "database": db_ok}
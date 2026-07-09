from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

# Async engine: uses asyncpg driver so FastAPI can await DB calls
# instead of blocking a worker thread on every query.
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class all ORM models inherit from."""
    pass


async def get_db():
    """
    FastAPI dependency. Yields a DB session per request and guarantees
    it's closed afterward, even if the request raises an exception.
    """
    async with AsyncSessionLocal() as session:
        yield session

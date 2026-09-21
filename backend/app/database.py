from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

# SQLite connection args with busy timeout
connect_args = {"timeout": 30} if "sqlite" in settings.database_url else {}

engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
    connect_args=connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """Create tables on startup and enable WAL mode for SQLite concurrency."""
    async with engine.begin() as conn:
        if "sqlite" in settings.database_url:
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
            await conn.exec_driver_sql("PRAGMA busy_timeout=30000;")
            await conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
        await conn.run_sync(Base.metadata.create_all)

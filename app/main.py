from fastapi import FastAPI

from app.api.router import router

app = FastAPI(
    title="Structured Output Benchmark",
    description="다양한 structured output 프레임워크 벤치마크 서버",
    version="0.1.0",
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}

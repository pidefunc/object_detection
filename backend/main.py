import logging
from fastapi import FastAPI
from backend.routers import detection
import uvicorn

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

app.include_router(detection.router)

@app.get("/")
def read_root():
    return {"message": "Object Detection API"}

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        ssl_keyfile="backend/key.pem",  # 使用生成的 key.pem 的实际路径
        ssl_certfile="backend/cert.pem"  # 使用生成的 cert.pem 的实际路径
    )

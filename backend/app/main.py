import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.qa import router as qa_router
from app.api.summary import router as summary_router
from app.api.videos import router as videos_router

os.environ["SB_DISABLE_K2"] = "1"  # disable speechbrain k2

app = FastAPI(
    title="Video Understanding API",
    description="Meeting & Lecture video Q&A system",
    version="0.1.0",
)

# CORS: Cho phép frontend  (localhost:3000) gọi API này (localhost:8000)
# CORS = Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js mặc định chạy port 3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos_router)
app.include_router(qa_router)
app.include_router(summary_router)


@app.get("/")
def root():
    return {"message": "Video Understanding API is running"}


@app.get("/health")
def health_check():  # endpoint để frontend kiểm tra backend còn sống hay không
    return {"status": "ok"}

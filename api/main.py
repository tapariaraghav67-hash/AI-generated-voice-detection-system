from fastapi import FastAPI
from api.routes.detect import router as detect_router

app = FastAPI(
    title="AI-Generated Voice Detection API",
    version="1.0"
)

# Register routes
app.include_router(detect_router)

# -------------------------
# Health check (IMPORTANT)
# -------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "AI-Generated Voice Detection API is running"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

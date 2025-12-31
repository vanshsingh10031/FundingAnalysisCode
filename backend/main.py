from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.rag_engine import analyze_funding

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(title="AI Funding Intelligence API")

# -------------------------------------------------
# CORS (allow frontend access)
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Serve Frontend
# -------------------------------------------------
app.mount("/static", StaticFiles(directory="Frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("Frontend/intropage.html")


# @app.get("/2.html")
# def serve_frontend2():
#     return FileResponse("Frontend/2.html")

# -------------------------------------------------
# Request Model (FUNDING)
# -------------------------------------------------
class FundingRequest(BaseModel):
    startup_description: str
    stage: str
    sector: str
    geography: str
    funding_goal: str

# -------------------------------------------------
# Funding Analysis Endpoint
# -------------------------------------------------
@app.post("/analyze")
def analyze(request: FundingRequest):
    return analyze_funding(
        startup_description=request.startup_description,
        stage=request.stage,
        sector=request.sector,
        geography=request.geography,
        funding_goal=request.funding_goal
    )

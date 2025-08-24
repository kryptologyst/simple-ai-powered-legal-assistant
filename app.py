import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import gradio as gr

from services.legal_generator import generate_legal_document, stream_legal_document
from fastapi.responses import StreamingResponse
from legal_assistant import interface as gradio_interface


# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("legal-assistant")


app = FastAPI(title="AI Legal Assistant API")

# CORS (adjust origins for your environment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LegalRequest(BaseModel):
    doc_type: str = Field(..., description="Document type, e.g., rental agreement, employment contract, partnership, nda")
    party1: str = Field(..., min_length=1)
    party2: str = Field(..., min_length=1)
    duration: Optional[str] = Field("", description="Months, for rental")
    salary: Optional[str] = Field("", description="Annual, for employment")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    num_predict: Optional[int] = Field(None, ge=1, le=8192)


class LegalResponse(BaseModel):
    response: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/legal/", response_model=LegalResponse)
async def legal(req: LegalRequest) -> LegalResponse:
    try:
        text = await generate_legal_document(
            doc_type=req.doc_type,
            party1=req.party1,
            party2=req.party2,
            duration=req.duration or "",
            salary=req.salary or "",
            temperature=req.temperature,
            top_p=req.top_p,
            num_predict=req.num_predict,
        )
        return LegalResponse(response=text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}")


@app.post("/legal/stream")
async def legal_stream(req: LegalRequest):
    async def generator():
        try:
            async for chunk in stream_legal_document(
                doc_type=req.doc_type,
                party1=req.party1,
                party2=req.party2,
                duration=req.duration or "",
                salary=req.salary or "",
                temperature=req.temperature,
                top_p=req.top_p,
                num_predict=req.num_predict,
            ):
                yield chunk
        except Exception as e:
            yield f"\n[STREAM ERROR] {e}"

    return StreamingResponse(generator(), media_type="text/plain")


# Mount Gradio UI
app = gr.mount_gradio_app(app, gradio_interface, path="/ui")

# Run with: uvicorn app:app --reload

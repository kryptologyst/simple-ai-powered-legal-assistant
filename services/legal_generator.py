from typing import Dict, Optional, AsyncGenerator, Generator

from config import MODEL_NAME, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_NUM_PREDICT
from .ollama_client import generate, stream_generate, stream_generate_sync, OllamaError

# Centralized legal templates with required fields metadata
LEGAL_TEMPLATES: Dict[str, Dict[str, str]] = {
    "rental agreement": {
        "template": (
            "Generate a comprehensive residential rental agreement between {party1} (tenant) "
            "and {party2} (landlord) for a term of {duration} months. Include: premises, term, rent, "
            "security deposit, utilities, maintenance, tenant obligations, landlord obligations, default, "
            "termination, governing law, and signature blocks. Write in clear legal language."
        ),
    },
    "employment contract": {
        "template": (
            "Draft an employment contract between {party2} (employer) and {party1} (employee) with an annual "
            "salary of {salary}. Include: position, duties, compensation, benefits, working hours, probation, "
            "confidentiality, IP assignment, non-compete (if appropriate), termination, severance, and governing law."
        ),
    },
    "business partnership agreement": {
        "template": (
            "Draft a business partnership agreement between {party1} and {party2}. Include: contributions, ownership percentages, "
            "management and decision-making, profit/loss allocation, withdrawals/distributions, dispute resolution, "
            "admission/withdrawal of partners, dissolution, and governing law."
        ),
    },
    "nda": {
        "template": (
            "Generate a mutual non-disclosure agreement between {party1} and {party2} to protect confidential information. "
            "Include: definitions, obligations, exclusions, term, permitted disclosures, remedies, and governing law."
        ),
    },
}

# Helper to normalize doc types from UI/API
DOC_ALIASES = {
    "rental agreement": "rental agreement",
    "rental": "rental agreement",
    "lease": "rental agreement",
    "employment contract": "employment contract",
    "employment": "employment contract",
    "job": "employment contract",
    "business partnership agreement": "business partnership agreement",
    "partnership": "business partnership agreement",
    "bpa": "business partnership agreement",
    "nda": "nda",
    "non-disclosure": "nda",
}


def normalize_doc_type(doc_type: str) -> Optional[str]:
    key = doc_type.strip().lower()
    return DOC_ALIASES.get(key) or (key if key in LEGAL_TEMPLATES else None)


async def generate_legal_document(
    *,
    doc_type: str,
    party1: str,
    party2: str,
    duration: Optional[str] = "",
    salary: Optional[str] = "",
    model: Optional[str] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
    top_p: Optional[float] = DEFAULT_TOP_P,
    num_predict: Optional[int] = DEFAULT_NUM_PREDICT,
) -> str:
    canonical = normalize_doc_type(doc_type)
    if not canonical:
        raise ValueError(
            "Invalid document type. Choose from rental agreement, employment contract, business partnership agreement, or NDA."
        )

    template = LEGAL_TEMPLATES[canonical]["template"]
    prompt = template.format(party1=party1, party2=party2, duration=duration or "",
                             salary=salary or "")

    # Add a compliance and formatting instruction footer
    prompt += (
        "\n\nConstraints: Use clear headings and bullet points where helpful. Avoid hallucinating facts. "
        "Add a final section: 'Important Notice: This document is AI-generated and must be reviewed by a qualified attorney.'"
    )

    response = await generate(
        prompt,
        model=model or MODEL_NAME,
        temperature=temperature,
        top_p=top_p,
        num_predict=num_predict,
        stream=False,
    )
    return response


def build_prompt(doc_type: str, party1: str, party2: str, duration: Optional[str] = "", salary: Optional[str] = "") -> str:
    canonical = normalize_doc_type(doc_type)
    if not canonical:
        raise ValueError(
            "Invalid document type. Choose from rental agreement, employment contract, business partnership agreement, or NDA."
        )
    template = LEGAL_TEMPLATES[canonical]["template"]
    prompt = template.format(party1=party1, party2=party2, duration=duration or "",
                             salary=salary or "")
    prompt += (
        "\n\nConstraints: Use clear headings and bullet points where helpful. Avoid hallucinating facts. "
        "Add a final section: 'Important Notice: This document is AI-generated and must be reviewed by a qualified attorney.'"
    )
    return prompt


async def stream_legal_document(
    *,
    doc_type: str,
    party1: str,
    party2: str,
    duration: Optional[str] = "",
    salary: Optional[str] = "",
    model: Optional[str] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
    top_p: Optional[float] = DEFAULT_TOP_P,
    num_predict: Optional[int] = DEFAULT_NUM_PREDICT,
) -> AsyncGenerator[str, None]:
    prompt = build_prompt(doc_type, party1, party2, duration, salary)
    async for chunk in stream_generate(
        prompt,
        model=model or MODEL_NAME,
        temperature=temperature,
        top_p=top_p,
        num_predict=num_predict,
    ):
        yield chunk


def stream_legal_document_sync(
    *,
    doc_type: str,
    party1: str,
    party2: str,
    duration: Optional[str] = "",
    salary: Optional[str] = "",
    model: Optional[str] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
    top_p: Optional[float] = DEFAULT_TOP_P,
    num_predict: Optional[int] = DEFAULT_NUM_PREDICT,
) -> Generator[str, None, None]:
    prompt = build_prompt(doc_type, party1, party2, duration, salary)
    yield from stream_generate_sync(
        prompt,
        model=model or MODEL_NAME,
        temperature=temperature,
        top_p=top_p,
        num_predict=num_predict,
    )

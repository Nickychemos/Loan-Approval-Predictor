"""Alternative-data feature extraction with LlamaIndex.

Turns a borrower's messy SMS / statement text into STRUCTURED FEATURES for the
loan model (active loans, salary regularity, bounced payments, ...). The LLM is
SWAPPABLE: Gemini (free hosted API) if GOOGLE_API_KEY is set, else local Ollama
(free + private — preferred for real PII). LlamaIndex is a feature extractor,
NOT the decision-maker: it outputs numbers; the model still decides.

Demo:
    GOOGLE_API_KEY=<key> python -m loan_predictor.altdata     # Gemini
    OLLAMA_MODEL=llama3.2:3b python -m loan_predictor.altdata  # local Ollama
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load GOOGLE_API_KEY etc. from the repo-root .env (gitignored), regardless of CWD.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class AltDataFeatures(BaseModel):
    """Alternative-data features pulled from a borrower's SMS / statements."""
    active_loans: int = Field(0, description="Number of distinct active/outstanding loans across all lenders")
    lenders: list[str] = Field(default_factory=list, description="Lender / loan-app names mentioned, e.g. Branch, Tala, KCB")
    has_regular_salary: bool = Field(False, description="Do regular salary/income credits appear?")
    monthly_salary: Optional[float] = Field(None, description="Approx monthly salary (KES) if stated, else null")
    avg_monthly_inflow: Optional[float] = Field(None, description="Approx total money received per month if inferable, else null")
    bounced_or_late_payments: int = Field(0, description="Count of failed / late / overdue / bounced payment messages")
    insufficient_funds_alerts: int = Field(0, description="Count of insufficient-funds / failed-due-to-balance messages")


def get_llm():
    """Return the LLM: Gemini if GOOGLE_API_KEY is set, else local Ollama."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        from llama_index.llms.google_genai import GoogleGenAI
        # gemini-2.5-flash has free-tier quota (2.0-flash returned limit:0 on a new project).
        return GoogleGenAI(model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                           api_key=key, temperature=0)
    from llama_index.llms.ollama import Ollama
    return Ollama(model=os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
                  temperature=0, request_timeout=120)


PROMPT = (
    "You are a credit analyst reading a borrower's mobile-money / bank SMS messages. "
    "Extract the structured features. Use ONLY information present in the messages; "
    "if something is unknown, use 0, false, an empty list, or null. Count a lender only once.\n\n"
    "MESSAGES:\n{messages}"
)


def extract_features(messages: str) -> AltDataFeatures:
    """Run the LLM extraction on raw text and return validated features."""
    sllm = get_llm().as_structured_llm(AltDataFeatures)
    resp = sllm.complete(PROMPT.format(messages=messages))
    obj = getattr(resp, "raw", None)
    return obj if isinstance(obj, AltDataFeatures) else AltDataFeatures.model_validate_json(resp.text)


def read_pdf_text(path: str) -> str:
    """Extract plain text from a PDF (e.g. an M-Pesa / bank statement)."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def extract_from_pdf(path: str) -> AltDataFeatures:
    """Read a statement PDF and extract the same structured features."""
    return extract_features(read_pdf_text(path))


SAMPLE_SMS = """\
Tala: Your loan of KES 5,000 has been disbursed to your M-PESA. Repay KES 5,750 by 30/06/2026.
Branch: Reminder - your loan repayment of KES 3,200 is due tomorrow.
MPESA: Confirmed. You have received KES 45,000 from EMPLOYER LTD. SALARY JUNE.
MPESA: Failed. You have insufficient funds in your M-PESA account to send KES 2,000.
KCB: Your loan account is OVERDUE. Please pay KES 8,000 immediately to avoid penalties.
MPESA: Confirmed. You have received KES 1,200 from JOHN DOE.
MPESA: Confirmed. You have received KES 45,000 from EMPLOYER LTD. SALARY MAY.
"""


if __name__ == "__main__":
    import sys
    import json
    which = "Gemini" if (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")) else "Ollama (local)"
    if len(sys.argv) > 1 and sys.argv[1].endswith(".pdf"):
        path = sys.argv[1]
        print(f"Extracting from PDF '{path}' using: {which}\n")
        feats = extract_from_pdf(path)
    else:
        print(f"Extracting from sample SMS using: {which}\n")
        print(SAMPLE_SMS)
        feats = extract_features(SAMPLE_SMS)
    print("--- extracted features ---")
    print(json.dumps(feats.model_dump(), indent=2))

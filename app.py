import os
import re
import json
import time
import uuid
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import requests
import fitz
import ollama
import pytesseract
import chromadb

from PIL import Image
from bs4 import BeautifulSoup
from ddgs import DDGS
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------- CONFIG ----------------

MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral:latest")
DB_URL = "sqlite:///subsidiary_audit.db"
VECTOR_PATH = "./vector_db"

MAX_PDF_PAGES = 80
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

app = FastAPI(title="GenAI Subsidiary Extraction API")

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

chroma_client = chromadb.PersistentClient(path=VECTOR_PATH)
collection = chroma_client.get_or_create_collection("subsidiary_sources")
class ChatRequest(BaseModel):
    job_id: str
    company_name: str
    question: str

# ---------------- DATABASE ----------------

class Job(Base):
    __tablename__ = "jobs"

    job_id = Column(String, primary_key=True)
    status = Column(String, default="pending")
    input_file = Column(String)
    output_file = Column(String)
    total_companies = Column(Integer, default=0)
    completed_companies = Column(Integer, default=0)
    remarks = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class ExtractionResult(Base):
    __tablename__ = "extraction_results"

    id = Column(String, primary_key=True)
    job_id = Column(String)
    parent_company = Column(String)
    subsidiary_name = Column(String)
    incorporated_location = Column(String)
    holding_percentage = Column(String)
    source_url = Column(Text)
    confidence = Column(String)
    remarks = Column(Text)
    review_status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)


class SourceAudit(Base):
    __tablename__ = "source_audit"

    id = Column(String, primary_key=True)
    job_id = Column(String)
    company_name = Column(String)
    source_url = Column(Text)
    source_title = Column(Text)
    source_type = Column(String)
    rank_score = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(engine)


# ---------------- MODELS ----------------

class ReviewUpdate(BaseModel):
    subsidiary_name: str
    incorporated_location: str
    holding_percentage: str
    confidence: str
    remarks: str
    review_status: str


# ---------------- HELPERS ----------------

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_domain(website: str) -> str:
    if not website or str(website).lower() == "nan":
        return ""
    return (
        str(website)
        .replace("https://", "")
        .replace("http://", "")
        .replace("www.", "")
        .split("/")[0]
        .strip()
    )


def get_company_column(df: pd.DataFrame) -> str:
    cols = [
        "Parent Company Name",
        "Company name",
        "Company Name",
        "company",
        "Company",
        "Parent Company Enterprise",
        "parent_company",
    ]
    for col in cols:
        if col in df.columns:
            return col
    return df.columns[0]


def get_country_column(df: pd.DataFrame):
    for col in ["Parent Company Country", "Country", "country", "Location", "location"]:
        if col in df.columns:
            return col
    return None


def get_website_column(df: pd.DataFrame):
    for col in ["Website", "website", "Domain", "domain", "URL", "url"]:
        if col in df.columns:
            return col
    return None


# ---------------- SOURCE RANKING ----------------

def rank_source(url: str, title: str, snippet: str) -> int:
    text = f"{url} {title} {snippet}".lower()
    score = 0

    if "annual report" in text:
        score += 40
    if "subsidiar" in text:
        score += 30
    if "group structure" in text:
        score += 20
    if "investor" in text or "financial statement" in text:
        score += 20
    if url.lower().endswith(".pdf"):
        score += 20
    if any(x in text for x in ["linkedin", "facebook", "twitter", "bloomberg"]):
        score -= 30

    return score


def search_web(company_name: str, website: str = "") -> List[Dict[str, str]]:
    domain = normalize_domain(website)

    queries = [
        f'"{company_name}" annual report pdf subsidiaries',
        f'"{company_name}" annual report subsidiaries holding percentage',
        f'"{company_name}" subsidiaries ownership percentage',
        f'"{company_name}" group structure subsidiaries',
    ]

    if domain:
        queries.insert(0, f'site:{domain} "{company_name}" annual report subsidiaries')

    results = []

    with DDGS() as ddgs:
        for query in queries:
            for r in ddgs.text(query, max_results=5):
                item = {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                item["rank_score"] = rank_source(
                    item["url"], item["title"], item["snippet"]
                )
                results.append(item)
            time.sleep(1)

    unique = {}
    for item in results:
        if item["url"]:
            unique[item["url"]] = item

    ranked = sorted(unique.values(), key=lambda x: x["rank_score"], reverse=True)
    return ranked[:10]


# ---------------- EXTRACTION + OCR ----------------

def ocr_pdf_page(page) -> str:
    pix = page.get_pixmap(dpi=200)
    img = Image.open(BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def extract_pdf_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)

        if response.status_code != 200:
            return ""

        doc = fitz.open(stream=BytesIO(response.content), filetype="pdf")
        text_parts = []

        for page_num in range(min(len(doc), MAX_PDF_PAGES)):
            page = doc[page_num]
            text = page.get_text("text")

            if len(clean_text(text)) < 50:
                text = ocr_pdf_page(page)

            if text:
                text_parts.append(text)

        doc.close()
        return clean_text("\n".join(text_parts))

    except Exception:
        return ""


def fetch_page_text(url: str) -> str:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=25,
            stream=True,
        )

        if response.status_code != 200:
            return ""

        content_type = response.headers.get("Content-Type", "").lower()

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            return extract_pdf_text_from_url(url)

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        return clean_text(soup.get_text(" "))

    except Exception:
        return ""


# ---------------- VECTOR DB ----------------

def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]

        if len(clean_text(chunk)) > 100:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def store_chunks(
    job_id: str,
    company_name: str,
    source_url: str,
    source_title: str,
    source_type: str,
    text: str,
):
    chunks = chunk_text(text)

    for idx, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        embedding = embed_model.encode(chunk).tolist()

        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[
                {
                    "job_id": job_id,
                    "company_name": company_name,
                    "source_url": source_url,
                    "source_title": source_title,
                    "source_type": source_type,
                    "chunk_index": idx,
                }
            ],
        )


def retrieve_top_chunks(company_name: str, query: str, top_k: int = 30) -> List[Dict]:
    query_embedding = embed_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"company_name": company_name},
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    return [
        {
            "text": docs[i],
            "metadata": metas[i],
        }
        for i in range(len(docs))
    ]


def rerank_chunks(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    if not chunks:
        return []

    pairs = [[query, c["text"]] for c in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    return [item[0] for item in ranked[:top_k]]


def build_rag_context(best_chunks: List[Dict]) -> str:
    parts = []

    for idx, item in enumerate(best_chunks, start=1):
        meta = item["metadata"]
        parts.append(
            f"""
SOURCE {idx}
Title: {meta.get("source_title", "")}
URL: {meta.get("source_url", "")}
Content:
{item["text"]}
"""
        )

    return "\n\n".join(parts)


# ---------------- JSON + LLM ----------------

def extract_json_from_text(text: str):
    text = text.strip().replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    array_match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
    if array_match:
        return json.loads(array_match.group(0))

    object_match = re.search(r"\{.*\}", text, re.DOTALL)
    if object_match:
        obj = json.loads(object_match.group(0))
        return obj.get("subsidiaries", [])

    raise ValueError("Could not parse valid JSON")


def extract_subsidiaries(company_name, country, website, context):
    if not context.strip():
        return []

    prompt = f"""
You are a strict JSON extraction engine.

Extract subsidiaries of the parent company.

Parent company: {company_name}
Country: {country}
Website: {website}

Rules:
1. Return ONLY valid JSON array.
2. Extract only subsidiaries owned or controlled by the parent company.
3. Do not include customers, vendors, branches, products, or partners.
4. Holding percentage must be used only if explicitly found.
5. If missing, use "Not available".
6. Every row must include source_url.
7. Do not hallucinate.
8. If no subsidiary is found, return [].

JSON format:
[
  {{
    "parent_company": "{company_name}",
    "subsidiary_name": "",
    "incorporated_location": "",
    "holding_percentage": "",
    "source_url": "",
    "confidence": "High",
    "remarks": ""
  }}
]

Context:
{context}
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON. No markdown. No explanation.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={"temperature": 0, "num_predict": 4096},
    )

    raw = response["message"]["content"]
    data = extract_json_from_text(raw)

    if isinstance(data, dict):
        data = data.get("subsidiaries", [])

    return data if isinstance(data, list) else []


def validate_rows(company_name: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []

    for item in rows:
        subsidiary = clean_text(item.get("subsidiary_name", ""))

        if not subsidiary:
            continue

        if subsidiary.lower() == company_name.lower():
            continue

        source_url = clean_text(item.get("source_url", ""))

        confidence = item.get("confidence", "Medium")
        if not source_url:
            confidence = "Low"

        cleaned.append(
            {
                "parent_company": company_name,
                "subsidiary_name": subsidiary,
                "incorporated_location": clean_text(
                    item.get("incorporated_location", "Not available")
                ) or "Not available",
                "holding_percentage": clean_text(
                    item.get("holding_percentage", "Not available")
                ) or "Not available",
                "source_url": source_url,
                "confidence": clean_text(confidence),
                "remarks": clean_text(item.get("remarks", "")),
            }
        )

    if not cleaned:
        cleaned.append(
            {
                "parent_company": company_name,
                "subsidiary_name": "No subsidiaries found",
                "incorporated_location": "Not available",
                "holding_percentage": "Not available",
                "source_url": "",
                "confidence": "Low",
                "remarks": "No valid subsidiary information found.",
            }
        )

    return cleaned


# ---------------- BACKGROUND JOB ----------------

def process_job(job_id: str, input_path: str):
    db = SessionLocal()

    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        job.status = "running"
        db.commit()

        df = pd.read_excel(input_path)
        df.columns = [str(c).strip() for c in df.columns]

        company_col = get_company_column(df)
        country_col = get_country_column(df)
        website_col = get_website_column(df)

        job.total_companies = len(df)
        db.commit()

        final_rows = []

        for _, row in df.iterrows():
            company_name = str(row.get(company_col, "")).strip()
            country = str(row.get(country_col, "")).strip() if country_col else ""
            website = str(row.get(website_col, "")).strip() if website_col else ""

            if not company_name or company_name.lower() == "nan":
                continue

            search_results = search_web(company_name, website)

            for result in search_results:
                url = result["url"]
                text = fetch_page_text(url)

                source_type = "pdf" if url.lower().endswith(".pdf") else "html"

                audit = SourceAudit(
                    id=str(uuid.uuid4()),
                    job_id=job_id,
                    company_name=company_name,
                    source_url=url,
                    source_title=result.get("title", ""),
                    source_type=source_type,
                    rank_score=result.get("rank_score", 0),
                )
                db.add(audit)
                db.commit()

                if len(text) > 100:
                    store_chunks(
                        job_id=job_id,
                        company_name=company_name,
                        source_url=url,
                        source_title=result.get("title", ""),
                        source_type=source_type,
                        text=text,
                    )

            query = (
                f"{company_name} subsidiaries ownership percentage "
                f"holding percentage incorporated location annual report"
            )

            candidates = retrieve_top_chunks(company_name, query, top_k=30)
            best_chunks = rerank_chunks(query, candidates, top_k=5)
            rag_context = build_rag_context(best_chunks)

            try:
                rows = extract_subsidiaries(company_name, country, website, rag_context)
                rows = validate_rows(company_name, rows)
            except Exception as e:
                rows = [
                    {
                        "parent_company": company_name,
                        "subsidiary_name": "Error",
                        "incorporated_location": "Not available",
                        "holding_percentage": "Not available",
                        "source_url": "",
                        "confidence": "Low",
                        "remarks": str(e),
                    }
                ]

            for r in rows:
                result = ExtractionResult(
                    id=str(uuid.uuid4()),
                    job_id=job_id,
                    parent_company=r["parent_company"],
                    subsidiary_name=r["subsidiary_name"],
                    incorporated_location=r["incorporated_location"],
                    holding_percentage=r["holding_percentage"],
                    source_url=r["source_url"],
                    confidence=r["confidence"],
                    remarks=r["remarks"],
                )
                db.add(result)
                final_rows.append(r)

            job.completed_companies += 1
            db.commit()

        output_path = f"output_{job_id}.xlsx"
        pd.DataFrame(final_rows).drop_duplicates().to_excel(output_path, index=False)

        job.status = "completed"
        job.output_file = output_path
        job.remarks = "Completed successfully"
        db.commit()

    except Exception as e:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        job.status = "failed"
        job.remarks = str(e)
        db.commit()

    finally:
        db.close()


# ---------------- API ROUTES ----------------

@app.post("/upload")
async def upload_excel(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    input_path = f"input_{job_id}_{file.filename}"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    db = SessionLocal()
    job = Job(
        job_id=job_id,
        status="pending",
        input_file=input_path,
        remarks="Job created",
    )
    db.add(job)
    db.commit()
    db.close()

    background_tasks.add_task(process_job, job_id, input_path)

    return {
        "job_id": job_id,
        "status": "started",
        "message": "Processing started",
    }


@app.get("/job/{job_id}")
def get_job_status(job_id: str):
    db = SessionLocal()
    job = db.query(Job).filter(Job.job_id == job_id).first()
    db.close()

    if not job:
        return {"error": "Job not found"}

    return {
        "job_id": job.job_id,
        "status": job.status,
        "total_companies": job.total_companies,
        "completed_companies": job.completed_companies,
        "remarks": job.remarks,
        "output_file": job.output_file,
    }


@app.get("/results/{job_id}")
def get_results(job_id: str):
    db = SessionLocal()
    rows = db.query(ExtractionResult).filter(ExtractionResult.job_id == job_id).all()
    db.close()

    return [
        {
            "id": r.id,
            "parent_company": r.parent_company,
            "subsidiary_name": r.subsidiary_name,
            "incorporated_location": r.incorporated_location,
            "holding_percentage": r.holding_percentage,
            "source_url": r.source_url,
            "confidence": r.confidence,
            "remarks": r.remarks,
            "review_status": r.review_status,
        }
        for r in rows
    ]


@app.get("/sources/{job_id}")
def get_sources(job_id: str):
    db = SessionLocal()
    rows = db.query(SourceAudit).filter(SourceAudit.job_id == job_id).all()
    db.close()

    return [
        {
            "company_name": r.company_name,
            "source_url": r.source_url,
            "source_title": r.source_title,
            "source_type": r.source_type,
            "rank_score": r.rank_score,
        }
        for r in rows
    ]


@app.put("/review/{result_id}")
def manual_review_update(result_id: str, payload: ReviewUpdate):
    db = SessionLocal()
    row = db.query(ExtractionResult).filter(ExtractionResult.id == result_id).first()

    if not row:
        db.close()
        return {"error": "Result not found"}

    row.subsidiary_name = payload.subsidiary_name
    row.incorporated_location = payload.incorporated_location
    row.holding_percentage = payload.holding_percentage
    row.confidence = payload.confidence
    row.remarks = payload.remarks
    row.review_status = payload.review_status

    db.commit()
    db.close()

    return {"message": "Review updated successfully"}


@app.get("/download/{job_id}")
def download_result(job_id: str):
    db = SessionLocal()
    job = db.query(Job).filter(Job.job_id == job_id).first()
    db.close()

    if not job or not job.output_file:
        return {"error": "Output file not available"}

    return FileResponse(
        job.output_file,
        filename="subsidiary_extraction_result.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.post("/chat")
def chat_with_company_data(payload: ChatRequest):
    db = SessionLocal()

    try:
        result_rows = (
            db.query(ExtractionResult)
            .filter(ExtractionResult.job_id == payload.job_id)
            .filter(ExtractionResult.parent_company.ilike(f"%{payload.company_name}%"))
            .all()
        )

        source_rows = (
            db.query(SourceAudit)
            .filter(SourceAudit.job_id == payload.job_id)
            .filter(SourceAudit.company_name.ilike(f"%{payload.company_name}%"))
            .all()
        )

        query = (
            f"{payload.company_name} {payload.question} "
            f"subsidiaries holding percentage incorporated location source evidence"
        )

        vector_chunks = retrieve_top_chunks(
            company_name=payload.company_name,
            query=query,
            top_k=20,
        )

        best_chunks = rerank_chunks(
            query=query,
            chunks=vector_chunks,
            top_k=5,
        )

        rag_context = build_rag_context(best_chunks)

        structured_result_context = []

        for r in result_rows:
            structured_result_context.append(
                f"""
Parent Company: {r.parent_company}
Subsidiary: {r.subsidiary_name}
Incorporated Location: {r.incorporated_location}
Holding Percentage: {r.holding_percentage}
Source URL: {r.source_url}
Confidence: {r.confidence}
Remarks: {r.remarks}
Review Status: {r.review_status}
"""
            )

        source_context = []

        for s in source_rows[:10]:
            source_context.append(
                f"""
Source Title: {s.source_title}
Source URL: {s.source_url}
Source Type: {s.source_type}
Rank Score: {s.rank_score}
"""
            )

        prompt = f"""
You are a company subsidiary QA assistant.

Answer the user's question using only the given context.

Rules:
1. Do not hallucinate.
2. If data is not available, say "Not available in processed sources".
3. Mention source URLs where useful.
4. Keep answer clear and business-friendly.
5. Use extracted result data first, then source chunks.

Company:
{payload.company_name}

User Question:
{payload.question}

Extracted Result Context:
{chr(10).join(structured_result_context)}

Source Audit Context:
{chr(10).join(source_context)}

Relevant Source Chunks:
{rag_context}
"""

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a reliable QA assistant for subsidiary extraction data.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0,
                "num_predict": 2048,
            },
        )

        answer = response["message"]["content"].strip()

        sources = [
            {
                "source_title": s.source_title,
                "source_url": s.source_url,
                "source_type": s.source_type,
                "rank_score": s.rank_score,
            }
            for s in source_rows[:10]
        ]

        return {
            "answer": answer,
            "sources": sources,
        }

    finally:
        db.close()
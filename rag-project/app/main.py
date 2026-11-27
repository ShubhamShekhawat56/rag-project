"""FastAPI app: JSON API + static single-page frontend."""
import io
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import config, rag, vectorstore
from .providers import ProviderError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rag.api")
ROOT = Path(__file__).resolve().parent.parent
NS_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")


@asynccontextmanager
async def lifespan(_):
    missing = config.missing_keys()
    if missing:
        log.warning("Missing environment variables: %s", ", ".join(missing))
    else:
        try:
            vectorstore.ensure_collection()
        except Exception as e:  # keep serving so /api/health can explain the problem
            log.error("Could not initialise Qdrant collection: %s", e)
    yield


app = FastAPI(title="Mini RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.exception_handler(ProviderError)
async def provider_error(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=502)


def _ns(namespace: str) -> str:
    cleaned = (namespace or "").strip()
    if not NS_RE.match(cleaned):
        raise HTTPException(400, "Invalid session id.")
    return cleaned


def _require_config():
    missing = config.missing_keys()
    if missing:
        raise HTTPException(503, "Server is missing configuration: " + ", ".join(missing))


class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=config.MAX_QUESTION_CHARS)
    namespace: str

    @field_validator("question")
    @classmethod
    def _non_blank_question(cls, v: str) -> str:
        s = v.strip()
        if len(s) < 3:
            raise ValueError("Question must be at least 3 characters.")
        return s


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    missing = config.missing_keys()
    return {
        "status": "ok",
        "configured": not missing,
        "missing": missing,
        "provider": config.EMBED_PROVIDER,
        "embed_model": config.EMBED_MODEL,
        "rerank_model": config.RERANK_MODEL,
        "llm_model": config.LLM_MODEL,
        "chunk_size": config.CHUNK_SIZE_TOKENS,
        "chunk_overlap": config.CHUNK_OVERLAP_TOKENS,
    }


@app.get("/api/sample")
def sample():
    sample_path = ROOT / "data" / "sample.md"
    if not sample_path.exists():
        raise HTTPException(404, "Sample document not found.")
    return {"title": "Northwind Labs Handbook (sample)", "text": sample_path.read_text("utf-8")}


def _read_upload(f: UploadFile) -> str:
    raw = f.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 5 MB.")
    if not raw:
        raise HTTPException(400, "Uploaded file is empty.")

    name = Path(f.filename or "").name.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(raw))
            return "\n\n".join((p.extract_text() or "").strip() for p in reader.pages if (p.extract_text() or "").strip())
        except Exception as e:
            raise HTTPException(422, "Could not read this PDF.") from e
    if name.endswith((".txt", ".md")):
        return raw.decode("utf-8", errors="ignore")
    raise HTTPException(415, "Upload a .txt, .md or .pdf file, or paste text instead.")


@app.post("/api/ingest")
def ingest(
    namespace: str = Form(...),
    title: str = Form(""),
    text: str = Form(""),
    file: UploadFile | None = File(None),
):
    _require_config()
    ns = _ns(namespace)
    content, source = text, None
    if file is not None and file.filename:
        content, source = _read_upload(file), Path(file.filename).name
    content = (content or "").strip()
    if not content:
        raise HTTPException(400, "Paste some text or upload a file first.")
    if len(content) > config.MAX_DOC_CHARS:
        raise HTTPException(413, f"Text is longer than {config.MAX_DOC_CHARS:,} characters.")

    clean_title = (title or "").strip()[:120] or source or "Pasted text"
    clean_source = (
        re.sub(r"[^a-z0-9._-]+", "-", (source or clean_title).lower()).strip("-")[:60]
        or "pasted-text"
    )
    try:
        return rag.ingest(content, clean_title, clean_source, ns)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except ProviderError:
        raise
    except Exception as e:
        log.exception("Ingest failed for namespace=%s source=%s", ns, clean_source)
        raise HTTPException(500, "Could not index document right now. Try again.") from e


@app.post("/api/query")
def query(body: QueryIn):
    _require_config()
    ns = _ns(body.namespace)
    try:
        return rag.answer(body.question, ns)
    except ProviderError:
        raise
    except Exception as e:
        log.exception("Query failed for namespace=%s", ns)
        raise HTTPException(500, "Could not process question right now. Try again.") from e


@app.get("/api/documents")
def documents(namespace: str = Query(...)):
    _require_config()
    ns = _ns(namespace)
    try:
        return {"documents": vectorstore.list_sources(ns)}
    except Exception as e:
        log.exception("List documents failed for namespace=%s", ns)
        raise HTTPException(500, "Could not fetch documents.") from e


@app.delete("/api/documents")
def clear(namespace: str = Query(...)):
    _require_config()
    ns = _ns(namespace)
    try:
        vectorstore.clear_namespace(ns)
        return {"cleared": True}
    except Exception as e:
        log.exception("Clear documents failed for namespace=%s", ns)
        raise HTTPException(500, "Could not clear documents.") from e
import base64
import os
import uuid
from functools import lru_cache

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.engine.nlp import GrammarEngine, _build_symspell, two_stage_correct
from app.engine.vision import TrOCREngine, segment_lines
from app.export.exporters import export_class_to_xlsx, export_document_to_docx, export_document_to_pdf
from app.models import Document, Node
from app.schemas import DocumentOut, DocumentWithNodes, NodeOut, NodeUpdate, ProcessImageResponse

router = APIRouter(prefix="/api", tags=["htr"])


# --- Model singletons -------------------------------------------------------
# Loaded lazily (first request) and cached, so the app starts instantly and
# the (large) model weights are only pulled into memory once, shared across
# requests. `lru_cache` on a zero-arg function is a simple process-wide singleton.


@lru_cache(maxsize=1)
def get_trocr_engine() -> TrOCREngine:
    return TrOCREngine()


@lru_cache(maxsize=1)
def get_grammar_engine() -> GrammarEngine:
    return GrammarEngine()


@lru_cache(maxsize=1)
def get_symspell():
    return _build_symspell()


# --- Helpers -----------------------------------------------------------------


def _decode_image(raw_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image data")
    return img


async def _run_pipeline(
    db: AsyncSession, page_bgr: np.ndarray, filename: str, student_id: str, writer_name: str | None, class_id: str | None
) -> tuple[Document, list[Node]]:
    document = Document(
        filename=filename,
        student_id=student_id,
        writer_name=writer_name,
        class_id=class_id,
        status="processing",
    )
    db.add(document)
    await db.flush()  # get document.id before children are inserted

    trocr = get_trocr_engine()
    grammar = get_grammar_engine()
    sym_spell = get_symspell()

    line_crops = segment_lines(page_bgr)

    nodes: list[Node] = []
    for i, crop in enumerate(line_crops):
        raw_text, confidence = trocr.transcribe(crop.image)
        corrected_text = two_stage_correct(raw_text, sym_spell, grammar)

        node = Node(
            document_id=document.id,
            raw_text=raw_text,
            corrected_text=corrected_text,
            bounding_box=crop.bounding_box,
            canvas_position={"x": 80, "y": 60 + i * 110},  # simple vertical auto-layout
            confidence_score=confidence,
            is_corrected=(raw_text.strip() != corrected_text.strip()),
        )
        db.add(node)
        nodes.append(node)

    document.status = "ready"
    await db.commit()
    for n in nodes:
        await db.refresh(n)
    await db.refresh(document)

    return document, nodes


# --- Endpoints ----------------------------------------------------------------


@router.post("/process-image", response_model=ProcessImageResponse)
async def process_image(
    student_id: str = Form(...),
    writer_name: str | None = Form(None),
    class_id: str | None = Form(None),
    file: UploadFile | None = File(None),
    image_base64: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Accepts either a multipart file upload OR a Base64 camera capture.
    Runs OpenCV segmentation -> TrOCR -> two-stage correction, persists a
    Document + its Nodes, and returns them."""
    if file is not None:
        raw_bytes = await file.read()
        filename = file.filename or f"upload-{uuid.uuid4().hex}.png"
    elif image_base64:
        header_stripped = image_base64.split(",")[-1]
        raw_bytes = base64.b64decode(header_stripped)
        filename = f"capture-{uuid.uuid4().hex}.png"
    else:
        raise HTTPException(status_code=400, detail="Provide either `file` or `image_base64`")

    os.makedirs(settings.upload_dir, exist_ok=True)
    page_bgr = _decode_image(raw_bytes)
    with open(os.path.join(settings.upload_dir, filename), "wb") as f:
        f.write(raw_bytes)

    document, nodes = await _run_pipeline(db, page_bgr, filename, student_id, writer_name, class_id)

    low_conf = sum(1 for n in nodes if n.confidence_score < settings.low_confidence_threshold)

    return ProcessImageResponse(
        document=DocumentOut.model_validate(document),
        nodes=[NodeOut.model_validate(n) for n in nodes],
        low_confidence_count=low_conf,
    )


@router.get("/nodes", response_model=list[NodeOut])
async def get_nodes(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Node).where(Node.document_id == document_id).order_by(Node.created_at))
    return result.scalars().all()


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(class_id: str | None = None, db: AsyncSession = Depends(get_db)):
    stmt = select(Document).order_by(Document.created_at.desc())
    if class_id:
        stmt = stmt.where(Document.class_id == class_id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/documents/{document_id}", response_model=DocumentWithNodes)
async def get_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    result = await db.execute(select(Node).where(Node.document_id == document_id).order_by(Node.created_at))
    doc_out = DocumentWithNodes.model_validate(doc)
    doc_out.nodes = [NodeOut.model_validate(n) for n in result.scalars().all()]
    return doc_out


@router.put("/nodes/{node_id}", response_model=NodeOut)
async def update_node(node_id: uuid.UUID, payload: NodeUpdate, db: AsyncSession = Depends(get_db)):
    node = await db.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    if payload.corrected_text is not None:
        node.corrected_text = payload.corrected_text
        node.is_corrected = node.raw_text.strip() != node.corrected_text.strip()
    if payload.canvas_position is not None:
        node.canvas_position = payload.canvas_position.model_dump()

    await db.commit()
    await db.refresh(node)

    # Mark the parent document reviewed once a human has touched any node
    doc = await db.get(Document, node.document_id)
    if doc and doc.status != "reviewed":
        doc.status = "reviewed"
        await db.commit()

    return node


@router.get("/export/{document_id}")
async def export_document(document_id: uuid.UUID, fmt: str = "docx", db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    result = await db.execute(select(Node).where(Node.document_id == document_id).order_by(Node.created_at))
    nodes = result.scalars().all()

    if fmt == "docx":
        path = export_document_to_docx(doc, nodes)
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif fmt == "pdf":
        path = export_document_to_pdf(doc, nodes)
        media_type = "application/pdf"
    elif fmt == "xlsx":
        path = export_class_to_xlsx([doc], {doc.id: nodes})
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        raise HTTPException(status_code=400, detail="fmt must be one of: docx, pdf, xlsx")

    return FileResponse(path, media_type=media_type, filename=os.path.basename(path))


@router.get("/export/class/{class_id}")
async def export_class(class_id: str, db: AsyncSession = Depends(get_db)):
    """Exports every student's document in a class to one multi-sheet
    workbook, one tab per student."""
    result = await db.execute(select(Document).where(Document.class_id == class_id))
    docs = result.scalars().all()
    if not docs:
        raise HTTPException(status_code=404, detail="No documents found for this class_id")

    nodes_by_doc: dict[uuid.UUID, list[Node]] = {}
    for doc in docs:
        res = await db.execute(select(Node).where(Node.document_id == doc.id).order_by(Node.created_at))
        nodes_by_doc[doc.id] = res.scalars().all()

    path = export_class_to_xlsx(docs, nodes_by_doc)
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=os.path.basename(path),
    )

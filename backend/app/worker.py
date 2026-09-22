import os
import asyncio
from celery import Celery
import cv2
import numpy as np

from app.config import settings
from app.database import AsyncSessionLocal
from app.models import ProcessingJob, ProcessingStatus, DocumentImage, Node, OcrResult
from app.engine.vision import TrOCREngine, segment_lines
from app.engine.nlp import GrammarEngine, two_stage_correct, _build_symspell

# Initialize Celery
celery_app = Celery(
    "htr_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Load engines globally in the worker
_trocr = None
_grammar = None
_symspell = None

@celery_app.on_after_configure.connect
def setup_engines(sender, **kwargs):
    global _trocr, _grammar, _symspell
    print("Loading AI Engines in Celery worker...")
    _trocr = TrOCREngine()
    _grammar = GrammarEngine()
    _symspell = _build_symspell()
    print("Engines loaded.")

async def _process_document_async(job_id: str):
    global _trocr, _grammar, _symspell
    
    # Initialize engines lazily if not set (for testing without full celery beat)
    if _trocr is None:
        _trocr = TrOCREngine()
    if _grammar is None:
        _grammar = GrammarEngine()
    if _symspell is None:
        _symspell = _build_symspell()

    async with AsyncSessionLocal() as db:
        # Get Job
        job = await db.get(ProcessingJob, job_id)
        if not job:
            return
            
        try:
            job.status = ProcessingStatus.PROCESSING
            await db.commit()
            
            # Get Document
            doc = await db.get(DocumentImage, job.document_id)
            if not doc:
                raise ValueError(f"Document {job.document_id} not found")
                
            # Read Image
            if not os.path.exists(doc.file_path):
                raise FileNotFoundError(f"Image not found at {doc.file_path}")
                
            page_bgr = cv2.imread(doc.file_path)
            if page_bgr is None:
                raise ValueError("Failed to decode image")
                
            # Vision Segmentation
            line_crops = segment_lines(page_bgr)
            
            # OCR + NLP Processing
            results = []
            for i, crop in enumerate(line_crops):
                raw_text, conf = _trocr.transcribe(crop.image)
                
                # Assign confidence level
                level = "LOW"
                if conf > 0.90:
                    level = "HIGH"
                elif conf > settings.low_confidence_threshold:
                    level = "MEDIUM"
                    
                corrected_text = two_stage_correct(raw_text, _symspell, _grammar)
                
                if "[UNCLEAR]" not in raw_text and conf < 0.2:
                    # Just an extreme fallback if the model completely fails
                    raw_text = "[UNCLEAR] " + raw_text
                    corrected_text = "[UNCLEAR] " + corrected_text
                
                # Create Node (Bounding Box + Text)
                node = Node(
                    document_id=doc.id,
                    student_id=doc.student_id,
                    line_number=i + 1,
                    bounding_box=crop.bounding_box,
                    raw_text=raw_text,
                    corrected_text=corrected_text,
                    confidence=conf,
                    confidence_level=level
                )
                db.add(node)
                # Flush to get Node ID
                await db.flush()
                
                # Create OcrResult for versioning
                ocr_result = OcrResult(
                    node_id=node.id,
                    engine_version="TrOCR-Base + T5",
                    raw_text=raw_text,
                    confidence_score=conf
                )
                db.add(ocr_result)
                
            job.status = ProcessingStatus.COMPLETED
            job.error_message = None
            await db.commit()
            
        except Exception as e:
            import traceback
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e) + "\n" + traceback.format_exc()
            await db.commit()

@celery_app.task(name="process_document")
def process_document_task(job_id: str):
    """Celery task entrypoint."""
    asyncio.run(_process_document_async(job_id))

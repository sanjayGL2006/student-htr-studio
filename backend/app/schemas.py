import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class CanvasPosition(BaseModel):
    x: float
    y: float


class NodeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    raw_text: str
    corrected_text: str
    bounding_box: BoundingBox
    canvas_position: CanvasPosition
    confidence_score: float
    is_corrected: bool
    created_at: datetime


class NodeUpdate(BaseModel):
    corrected_text: str | None = None
    canvas_position: CanvasPosition | None = None


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    student_id: str
    writer_name: str | None
    class_id: str | None
    status: str
    created_at: datetime


class DocumentWithNodes(DocumentOut):
    nodes: list[NodeOut] = []


class ProcessImageResponse(BaseModel):
    document: DocumentOut
    nodes: list[NodeOut]
    low_confidence_count: int

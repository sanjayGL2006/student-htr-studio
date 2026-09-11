import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, JSON, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    """One scanned/captured page, belonging to a single student submission."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)

    # writer_id is the canonical tag used to keep every line scoped to one student
    student_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    writer_name: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # optional grouping for a classroom batch upload / session
    class_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="processing")
    # processing | ready | reviewed

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    nodes: Mapped[list["Node"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Node.created_at"
    )


class Node(Base):
    """One recognized line/segment on a document, rendered as a React Flow node."""

    __tablename__ = "nodes"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )

    raw_text: Mapped[str] = mapped_column(String, nullable=False, default="")
    corrected_text: Mapped[str] = mapped_column(String, nullable=False, default="")

    # {x, y, w, h} of the crop in the source page image
    bounding_box: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # {x, y} canvas position for React Flow — independent of bounding_box so a
    # reviewer can rearrange nodes without touching the underlying OCR geometry
    canvas_position: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_corrected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    document: Mapped["Document"] = relationship(back_populates="nodes")


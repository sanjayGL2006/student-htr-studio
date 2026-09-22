import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(50), default="teacher")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Class(Base):
    __tablename__ = "classes"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255))
    teacher_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    teacher: Mapped["User"] = relationship()


class Student(Base):
    __tablename__ = "students"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    student_id_code: Mapped[str] = mapped_column(String(100), index=True)  # e.g., Roll Number
    name: Mapped[str] = mapped_column(String(255))
    class_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("classes.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    enrolled_class: Mapped["Class"] = relationship()


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    student_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("students.id", ondelete="SET NULL"), nullable=True)
    class_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("classes.id", ondelete="SET NULL"), nullable=True)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="Uploaded")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    nodes: Mapped[list["Node"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Node.line_number"
    )
    student: Mapped["Student"] = relationship()


class DocumentImage(Base):
    __tablename__ = "document_images"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    document_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("documents.id", ondelete="CASCADE"))
    image_type: Mapped[str] = mapped_column(String(50)) # e.g., 'original', 'preprocessed'
    file_path: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Node(Base):
    __tablename__ = "nodes"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    document_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    student_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("students.id", ondelete="SET NULL"), nullable=True)
    line_number: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    corrected_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confidence_level: Mapped[str] = mapped_column(String(20), nullable=False, default="High")
    bounding_box: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    canvas_position: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    document: Mapped["Document"] = relationship(back_populates="nodes")


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    document_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("documents.id", ondelete="CASCADE"))
    task_id: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class OcrResult(Base):
    __tablename__ = "ocr_results"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    node_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("nodes.id", ondelete="CASCADE"))
    raw_ocr: Mapped[Text] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class CorrectionHistory(Base):
    __tablename__ = "correction_history"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    node_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("nodes.id", ondelete="CASCADE"))
    old_text: Mapped[str] = mapped_column(Text)
    new_text: Mapped[str] = mapped_column(Text)
    changed_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Export(Base):
    __tablename__ = "exports"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    document_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    class_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("classes.id", ondelete="SET NULL"), nullable=True)
    format: Mapped[str] = mapped_column(String(20))
    file_path: Mapped[str] = mapped_column(String(512))
    created_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String(255))
    target_type: Mapped[str] = mapped_column(String(50)) # e.g. Document, Node
    target_id: Mapped[str] = mapped_column(String(255))
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


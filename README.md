# Classroom Multi-Writer Handwriting Recognition & Node Canvas System

A classroom-scale HTR pipeline: students' scanned/photographed pages are
segmented into lines, transcribed with TrOCR, spell/grammar-corrected, and
shown as editable nodes on a React Flow canvas — exportable per student or
as a whole-class workbook.

## Architecture

```
backend/   FastAPI + SQLAlchemy (async) + PostgreSQL
  app/
    engine/vision.py   OpenCV deskew + line segmentation, TrOCR inference
    engine/nlp.py      SymSpell lexical pass + T5 grammar correction
    api/routes.py      REST endpoints
    export/exporters.py  docx / pdf / multi-sheet xlsx export
    models.py          documents, nodes tables
    schemas.py         Pydantic request/response models
    main.py            app entrypoint

frontend/  React + Vite + TypeScript + Tailwind + React Flow
  src/
    components/IngestionPanel.tsx   webcam + drag-drop upload
    components/CanvasView.tsx       React Flow board, editable nodes
    components/LineNode.tsx         custom node (raw/corrected/confidence)
    components/RosterView.tsx       class roster grid
    components/ExportBar.tsx        per-student / per-class export links
```

## Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set DATABASE_URL to a running PostgreSQL instance

# create the database (adjust to your local Postgres setup)
createdb htr_db

uvicorn app.main:app --reload --port 8000
```

The first request that touches `/api/process-image` will download
`microsoft/trocr-base-handwritten` and `vennify/t5-base-grammar-correction`
from the Hugging Face Hub (a few hundred MB each) and cache them locally —
this needs outbound internet access to `huggingface.co` the first time only.

## Frontend setup

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
```

Set `VITE_API_BASE_URL` in a `frontend/.env` file if the backend isn't at
`http://localhost:8000/api`.

## API summary

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/process-image` | Upload/capture a page, run the full pipeline |
| GET | `/api/nodes?document_id=` | Fetch a document's recognized lines |
| GET | `/api/documents?class_id=` | List submissions (optionally by class) |
| GET | `/api/documents/{id}` | One document with its nodes |
| PUT | `/api/nodes/{id}` | Edit corrected text or canvas position |
| GET | `/api/export/{id}?fmt=docx\|pdf\|xlsx` | Export one student |
| GET | `/api/export/class/{class_id}` | Multi-sheet workbook, one tab per student |

## Verification performed in this build

Given the sandbox this was built in only allows outbound network access to
package registries (PyPI, npm) and not to `huggingface.co`, the TrOCR/T5
model weights could not be downloaded here, so live end-to-end inference on
a real handwritten photo was **not** run in this environment. What was
verified directly, standalone, without network access:

- **OpenCV segmentation** (`engine/vision.py`): ran `segment_lines()` against
  a synthetic multi-line page image — correctly detected and bounded all 4
  lines with sensible padding.
- **Two-stage text correction** (`engine/nlp.py`): ran the SymSpell lexical
  pass standalone against sample noisy strings — correctly resolved common
  OCR-style typos (`qick`→`quick`, `lazee`→`lazy`).
- **Export pipeline** (`export/exporters.py`): generated real `.docx`,
  `.pdf`, and a multi-sheet `.xlsx` (one tab per student, correctly named
  and separated) from mock node data — all three files opened correctly and
  contained expected content.
- **Backend code**: all modules pass `python -m py_compile` with no syntax
  errors. Full app boot (`uvicorn app.main:app`) was **not** run here because
  it requires a live PostgreSQL connection and the torch/transformers model
  downloads; do this as your first step after installing `requirements.txt`.
- **Frontend**: `npm run build` completes cleanly (TypeScript + Vite,
  zero errors).

**Before relying on this in a real classroom**, run the backend against a
real PostgreSQL instance and a handful of actual scanned pages, and check:
confidence scores are correlating sensibly with actual OCR quality, the
grammar model isn't over-correcting proper nouns/names (the `raw_text` vs
`corrected_text` diff in each node exists specifically so a teacher can
catch and revert this), and line segmentation handles your students'
specific handwriting scale/density — the dilation kernel size
(`(35, 5)` in `vision.py`) may need tuning for very small or very large
handwriting.

## Known limitations / documented upgrade paths

- **Segmentation is rectangular**, not polygon-based. Students whose lines
  slope or curve mid-line may get imperfect crops. `segment_lines_craft()`
  in `vision.py` is a stubbed-in upgrade path — swap the call site in
  `api/routes.py` and `pip install craft-text-detector` if you need this.
- **Confidence score is a proxy**, not a calibrated probability: it's the
  mean max-softmax across TrOCR's generated tokens. Useful for ranking
  "which lines need a human look" but not as an absolute quality metric.
- **Per-student or dataset retraining.** While the base model uses a generalized encoder, you can now fine-tune the model to specific handwriting styles or datasets using the provided training script. This allows the system to better handle unusual handwriting or domain-specific text.

## Fine-Tuning TrOCR

A dedicated script (`backend/train_trocr.py`) is provided to fine-tune the TrOCR model on custom data.

```bash
cd backend

# 1. Train on a Hugging Face dataset (e.g., IAM lines)
python train_trocr.py --dataset_name Teklia/iam-lines --epochs 5

# 2. Train on a local dataset (e.g., Kaggle Handwritten Names)
python train_trocr.py --local_data_dir "D:\classroom-htr-system\dataset\archive" --epochs 5

# 3. Fast verification with synthetic data
python train_trocr.py --synthetic --epochs 2
```

To use your fine-tuned model in the application, update the backend environment variable:
`TROCR_MODEL_ID=./models/trocr-custom`
- **SQLite is not supported** for this schema as-is: `nodes.bounding_box`
  and `nodes.canvas_position` use PostgreSQL's `JSONB` column type. If you
  want a lighter local-only setup (consistent with other SPVM³ projects),
  swap `JSONB` for a generic `JSON` type in `models.py` and switch the
  connection strings to `sqlite+aiosqlite:///./htr.db`.

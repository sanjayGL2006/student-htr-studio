import { useCallback, useRef, useState } from "react";
import Webcam from "react-webcam";
import { uploadCapture, uploadFile } from "../api/client";
import { ProcessImageResponse } from "../types";

interface Props {
  onProcessed: (result: ProcessImageResponse) => void;
}

export default function IngestionPanel({ onProcessed }: Props) {
  const [studentId, setStudentId] = useState("");
  const [writerName, setWriterName] = useState("");
  const [classId, setClassId] = useState("");
  const [cameraOn, setCameraOn] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const webcamRef = useRef<Webcam>(null);

  const canSubmit = studentId.trim().length > 0;

  const handleResult = useCallback(
    (result: ProcessImageResponse) => {
      onProcessed(result);
      setBusy(false);
    },
    [onProcessed]
  );

  const handleFile = useCallback(
    async (file: File) => {
      if (!canSubmit) {
        setError("Enter a student ID before uploading.");
        return;
      }
      setError(null);
      setBusy(true);
      try {
        const result = await uploadFile(file, studentId, writerName, classId);
        handleResult(result);
      } catch (err) {
        setError("Upload failed. Check the backend is running and the image is readable.");
        setBusy(false);
      }
    },
    [canSubmit, studentId, writerName, classId, handleResult]
  );

  const handleCapture = useCallback(async () => {
    if (!canSubmit) {
      setError("Enter a student ID before capturing.");
      return;
    }
    const shot = webcamRef.current?.getScreenshot();
    if (!shot) {
      setError("Could not read from the camera.");
      return;
    }
    setError(null);
    setBusy(true);
    try {
      const result = await uploadCapture(shot, studentId, writerName, classId);
      handleResult(result);
    } catch (err) {
      setError("Processing failed. Check the backend is running.");
      setBusy(false);
    }
  }, [canSubmit, studentId, writerName, classId, handleResult]);

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragActive(false);
      const file = e.dataTransfer.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-5 space-y-4">
      <h2 className="text-lg font-semibold text-amber">1. Ingest a submission</h2>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <input
          className="bg-slate-800 rounded-md px-3 py-2 text-sm border border-slate-700 focus:border-teal outline-none"
          placeholder="Student ID *"
          value={studentId}
          onChange={(e) => setStudentId(e.target.value)}
        />
        <input
          className="bg-slate-800 rounded-md px-3 py-2 text-sm border border-slate-700 focus:border-teal outline-none"
          placeholder="Student name (optional)"
          value={writerName}
          onChange={(e) => setWriterName(e.target.value)}
        />
        <input
          className="bg-slate-800 rounded-md px-3 py-2 text-sm border border-slate-700 focus:border-teal outline-none"
          placeholder="Class / batch ID (optional)"
          value={classId}
          onChange={(e) => setClassId(e.target.value)}
        />
      </div>

      {error && <p className="text-sm text-red-400">{error}</p>}

      <div className="flex items-center gap-3">
        <button
          onClick={() => setCameraOn((v) => !v)}
          className="px-3 py-1.5 rounded-md bg-slate-800 border border-slate-700 text-sm hover:border-teal transition"
        >
          {cameraOn ? "Close camera" : "Use live camera"}
        </button>
        {busy && <span className="text-sm text-teal animate-pulse">Processing page…</span>}
      </div>

      {cameraOn && (
        <div className="space-y-2">
          <Webcam
            ref={webcamRef}
            screenshotFormat="image/png"
            className="rounded-lg border border-slate-700 w-full max-w-md"
          />
          <button
            onClick={handleCapture}
            disabled={busy}
            className="px-4 py-2 rounded-md bg-teal text-navy font-medium text-sm disabled:opacity-50"
          >
            Capture & process
          </button>
        </div>
      )}

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={onDrop}
        className={`rounded-lg border-2 border-dashed p-8 text-center text-sm transition ${
          dragActive ? "border-amber bg-amber/5" : "border-slate-700"
        }`}
      >
        <p className="mb-2">Drag & drop a scanned page here</p>
        <label className="inline-block px-3 py-1.5 rounded-md bg-slate-800 border border-slate-700 cursor-pointer hover:border-teal">
          or browse a file
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file);
            }}
          />
        </label>
      </div>
    </div>
  );
}

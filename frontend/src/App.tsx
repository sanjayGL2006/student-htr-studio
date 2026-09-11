import { useState } from "react";
import CanvasView from "./components/CanvasView";
import ExportBar from "./components/ExportBar";
import IngestionPanel from "./components/IngestionPanel";
import RosterView from "./components/RosterView";
import { ProcessImageResponse } from "./types";

export default function App() {
  const [classId, setClassId] = useState("");
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const handleProcessed = (result: ProcessImageResponse) => {
    setSelectedDocumentId(result.document.id);
    if (result.document.class_id) setClassId(result.document.class_id);
    setRefreshKey((k) => k + 1);
  };

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-800 px-6 py-4">
        <h1 className="text-xl font-bold">
          <span className="text-amber">Classroom</span>{" "}
          <span className="text-teal">Handwriting Recognition</span>
        </h1>
        <p className="text-sm text-slate-400">Multi-writer HTR, auto-correction, and node-canvas review</p>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-6 space-y-6">
        <IngestionPanel onProcessed={handleProcessed} />

        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-400">Filter roster by class:</label>
          <input
            className="bg-slate-800 rounded-md px-3 py-1.5 text-sm border border-slate-700 focus:border-teal outline-none"
            placeholder="e.g. 7A"
            value={classId}
            onChange={(e) => setClassId(e.target.value)}
          />
        </div>

        <RosterView
          classId={classId}
          selectedDocumentId={selectedDocumentId}
          onSelect={setSelectedDocumentId}
          refreshKey={refreshKey}
        />

        {selectedDocumentId && (
          <>
            <h2 className="text-lg font-semibold text-amber">2. Review on canvas</h2>
            <CanvasView documentId={selectedDocumentId} />
          </>
        )}

        <ExportBar documentId={selectedDocumentId} classId={classId} />
      </main>
    </div>
  );
}

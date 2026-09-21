import { useEffect, useState } from "react";
import { listDocuments } from "../api/client";
import { DocumentData } from "../types";

interface Props {
  classId: string;
  selectedDocumentId: string | null;
  onSelect: (documentId: string) => void;
  refreshKey: number;
}

const statusColor: Record<DocumentData["status"], string> = {
  processing: "text-amber",
  ready: "text-teal",
  reviewed: "text-emerald-400",
};

export default function RosterView({ classId, selectedDocumentId, onSelect, refreshKey }: Props) {
  const [docs, setDocs] = useState<DocumentData[]>([]);

  useEffect(() => {
    listDocuments(classId || undefined)
      .then(setDocs)
      .catch((err) => {
        console.warn("Could not fetch documents:", err?.message || err);
      });
  }, [classId, refreshKey]);

  if (docs.length === 0) {
    return (
      <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-5 text-sm text-slate-400">
        No submissions yet{classId ? ` for class "${classId}"` : ""}.
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-5">
      <h2 className="text-lg font-semibold text-amber mb-3">
        Roster{classId ? ` — ${classId}` : ""}
      </h2>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {docs.map((d) => (
          <button
            key={d.id}
            onClick={() => onSelect(d.id)}
            className={`text-left rounded-lg border p-3 transition ${
              selectedDocumentId === d.id
                ? "border-teal bg-teal/10"
                : "border-slate-700 hover:border-slate-500"
            }`}
          >
            <p className="font-medium text-sm truncate">{d.writer_name || d.student_id}</p>
            <p className="text-xs text-slate-400 truncate">{d.filename}</p>
            <p className={`text-xs mt-1 ${statusColor[d.status]}`}>{d.status}</p>
          </button>
        ))}
      </div>
    </div>
  );
}

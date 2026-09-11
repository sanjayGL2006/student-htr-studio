import { exportClassUrl, exportDocumentUrl } from "../api/client";

interface Props {
  documentId: string | null;
  classId: string;
}

export default function ExportBar({ documentId, classId }: Props) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-5 flex flex-wrap items-center gap-3">
      <h2 className="text-lg font-semibold text-amber mr-4">3. Export</h2>

      <span className="text-xs text-slate-500 mr-1">This student:</span>
      {(["docx", "pdf", "xlsx"] as const).map((fmt) => (
        <a
          key={fmt}
          href={documentId ? exportDocumentUrl(documentId, fmt) : undefined}
          className={`px-3 py-1.5 rounded-md text-sm border transition ${
            documentId
              ? "border-slate-700 hover:border-teal"
              : "border-slate-800 text-slate-600 pointer-events-none"
          }`}
        >
          .{fmt}
        </a>
      ))}

      <span className="text-xs text-slate-500 ml-4 mr-1">Whole class:</span>
      <a
        href={classId ? exportClassUrl(classId) : undefined}
        className={`px-3 py-1.5 rounded-md text-sm border transition ${
          classId
            ? "border-slate-700 hover:border-amber"
            : "border-slate-800 text-slate-600 pointer-events-none"
        }`}
      >
        Multi-sheet .xlsx
      </a>
    </div>
  );
}

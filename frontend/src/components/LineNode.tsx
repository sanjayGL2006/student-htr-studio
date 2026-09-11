import { Handle, NodeProps, Position } from "reactflow";
import { LOW_CONFIDENCE_THRESHOLD } from "../types";

export interface LineNodeData {
  raw_text: string;
  corrected_text: string;
  confidence_score: number;
  is_corrected: boolean;
  onEdit: (value: string) => void;
}

export default function LineNode({ data }: NodeProps<LineNodeData>) {
  const lowConfidence = data.confidence_score < LOW_CONFIDENCE_THRESHOLD;

  return (
    <div className="w-72 rounded-lg border border-slate-700 bg-slate-900 shadow-lg overflow-hidden">
      <Handle type="target" position={Position.Top} className="!bg-teal" />

      <div className="flex items-center justify-between px-3 py-1.5 bg-slate-800 text-xs">
        <span className="text-slate-400">
          confidence: <span className={lowConfidence ? "text-red-400" : "text-teal"}>{(data.confidence_score * 100).toFixed(0)}%</span>
        </span>
        {data.is_corrected && (
          <span className="px-1.5 py-0.5 rounded bg-amber/20 text-amber text-[10px] font-medium">
            auto-corrected
          </span>
        )}
        {lowConfidence && (
          <span className="px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 text-[10px] font-medium">
            needs review
          </span>
        )}
      </div>

      <div className="p-3 space-y-2">
        <p className="text-[11px] uppercase tracking-wide text-slate-500">Raw OCR</p>
        <p className="text-xs text-slate-400 line-clamp-2">{data.raw_text || "—"}</p>

        <p className="text-[11px] uppercase tracking-wide text-slate-500 pt-1">Corrected (editable)</p>
        <textarea
          className="w-full bg-slate-800 border border-slate-700 rounded-md p-2 text-sm text-slate-100 focus:border-teal outline-none resize-none"
          rows={3}
          defaultValue={data.corrected_text}
          onBlur={(e) => data.onEdit(e.target.value)}
        />
      </div>

      <Handle type="source" position={Position.Bottom} className="!bg-teal" />
    </div>
  );
}

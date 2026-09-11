import { useCallback, useEffect, useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  Edge,
  Node,
  NodeChange,
  applyNodeChanges,
} from "reactflow";
import "reactflow/dist/style.css";
import { getDocument, updateNode } from "../api/client";
import { DocumentWithNodes } from "../types";
import LineNode, { LineNodeData } from "./LineNode";

const nodeTypes = { lineNode: LineNode };

interface Props {
  documentId: string;
}

export default function CanvasView({ documentId }: Props) {
  const [doc, setDoc] = useState<DocumentWithNodes | null>(null);
  const [flowNodes, setFlowNodes] = useState<Node<LineNodeData>[]>([]);
  const edges: Edge[] = [];

  const load = useCallback(async () => {
    const data = await getDocument(documentId);
    setDoc(data);
  }, [documentId]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!doc) return;
    setFlowNodes(
      doc.nodes.map((n) => ({
        id: n.id,
        type: "lineNode",
        position: n.canvas_position,
        data: {
          raw_text: n.raw_text,
          corrected_text: n.corrected_text,
          confidence_score: n.confidence_score,
          is_corrected: n.is_corrected,
          onEdit: (value: string) => {
            updateNode(n.id, { corrected_text: value });
          },
        },
      }))
    );
  }, [doc]);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      setFlowNodes((nds) => applyNodeChanges(changes, nds));

      // Persist position drags once the drag gesture ends
      for (const change of changes) {
        if (change.type === "position" && change.dragging === false && change.position) {
          updateNode(change.id, { canvas_position: change.position });
        }
      }
    },
    []
  );

  const lowConfidenceCount = useMemo(
    () => doc?.nodes.filter((n) => n.confidence_score < 0.55).length ?? 0,
    [doc]
  );

  if (!doc) {
    return <div className="text-sm text-slate-400 p-6">Loading document…</div>;
  }

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/60 overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3 border-b border-slate-700">
        <div>
          <h3 className="font-semibold text-amber">{doc.writer_name || doc.student_id}</h3>
          <p className="text-xs text-slate-400">
            {doc.nodes.length} lines · status: {doc.status}
            {lowConfidenceCount > 0 && (
              <span className="text-red-400"> · {lowConfidenceCount} need review</span>
            )}
          </p>
        </div>
      </div>
      <div style={{ height: 520 }}>
        <ReactFlow
          nodes={flowNodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          fitView
        >
          <Background color="#1e293b" gap={16} />
          <Controls />
        </ReactFlow>
      </div>
    </div>
  );
}

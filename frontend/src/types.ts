export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface CanvasPosition {
  x: number;
  y: number;
}

export interface NodeData {
  id: string;
  document_id: string;
  raw_text: string;
  corrected_text: string;
  bounding_box: BoundingBox;
  canvas_position: CanvasPosition;
  confidence_score: number;
  is_corrected: boolean;
  created_at: string;
}

export interface DocumentData {
  id: string;
  filename: string;
  student_id: string;
  writer_name: string | null;
  class_id: string | null;
  status: "processing" | "ready" | "reviewed";
  created_at: string;
}

export interface DocumentWithNodes extends DocumentData {
  nodes: NodeData[];
}

export interface ProcessImageResponse {
  document: DocumentData;
  nodes: NodeData[];
  low_confidence_count: number;
}

export const LOW_CONFIDENCE_THRESHOLD = 0.55;

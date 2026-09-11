import axios from "axios";
import {
  CanvasPosition,
  DocumentData,
  DocumentWithNodes,
  NodeData,
  ProcessImageResponse,
} from "../types";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

const client = axios.create({ baseURL: BASE_URL });

export async function uploadFile(
  file: File,
  studentId: string,
  writerName: string,
  classId: string
): Promise<ProcessImageResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("student_id", studentId);
  if (writerName) form.append("writer_name", writerName);
  if (classId) form.append("class_id", classId);

  const { data } = await client.post<ProcessImageResponse>("/process-image", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function uploadCapture(
  imageBase64: string,
  studentId: string,
  writerName: string,
  classId: string
): Promise<ProcessImageResponse> {
  const form = new FormData();
  form.append("image_base64", imageBase64);
  form.append("student_id", studentId);
  if (writerName) form.append("writer_name", writerName);
  if (classId) form.append("class_id", classId);

  const { data } = await client.post<ProcessImageResponse>("/process-image", form);
  return data;
}

export async function listDocuments(classId?: string): Promise<DocumentData[]> {
  const { data } = await client.get<DocumentData[]>("/documents", {
    params: classId ? { class_id: classId } : {},
  });
  return data;
}

export async function getDocument(documentId: string): Promise<DocumentWithNodes> {
  const { data } = await client.get<DocumentWithNodes>(`/documents/${documentId}`);
  return data;
}

export async function updateNode(
  nodeId: string,
  payload: { corrected_text?: string; canvas_position?: CanvasPosition }
): Promise<NodeData> {
  const { data } = await client.put<NodeData>(`/nodes/${nodeId}`, payload);
  return data;
}

export function exportDocumentUrl(documentId: string, fmt: "docx" | "pdf" | "xlsx"): string {
  return `${BASE_URL}/export/${documentId}?fmt=${fmt}`;
}

export function exportClassUrl(classId: string): string {
  return `${BASE_URL}/export/class/${classId}`;
}

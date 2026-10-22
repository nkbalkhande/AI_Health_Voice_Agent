import type { Fact, Message, ThreadSummary } from "./types";

export const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
export const WS_URL = API.replace(/^http/, "ws") + "/api/v1/ws/chat";

let idCounter = 0;
export const newId = () => `${Date.now().toString(36)}-${(idCounter++).toString(36)}`;

export async function listThreads(): Promise<ThreadSummary[]> {
  const res = await fetch(`${API}/api/v1/chat/threads`);
  if (!res.ok) throw new Error(`Could not load conversations (${res.status})`);
  const data = await res.json();
  return data.threads ?? [];
}

export async function getThread(
  id: string,
): Promise<{ messages: Message[]; summary: string; facts: Fact[] } | null> {
  const res = await fetch(`${API}/api/v1/chat/threads/${id}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Could not load conversation (${res.status})`);
  const data = await res.json();
  return {
    messages: (data.messages ?? []).map(
      (m: { role: Message["role"]; content: string; created_at?: string }) => ({
        id: newId(),
        role: m.role,
        content: m.content,
        createdAt: m.created_at,
      }),
    ),
    summary: data.summary ?? "",
    facts: (data.facts ?? []) as Fact[],
  };
}

export async function renameThread(id: string, title: string): Promise<void> {
  const res = await fetch(`${API}/api/v1/chat/threads/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error("Rename failed");
}

export async function deleteThread(id: string): Promise<void> {
  const res = await fetch(`${API}/api/v1/chat/threads/${id}`, { method: "DELETE" });
  if (!res.ok && res.status !== 404) throw new Error("Delete failed");
}

export type VoiceOption = { id: string; label: string; description: string };

export async function listVoices(): Promise<{ default: string | null; voices: VoiceOption[] }> {
  const res = await fetch(`${API}/api/v1/voice/voices`);
  if (!res.ok) throw new Error(`Could not load voices (${res.status})`);
  const data = await res.json();
  return { default: data.default ?? null, voices: data.voices ?? [] };
}

export async function synthesize(text: string, signal?: AbortSignal, voice?: string | null): Promise<Blob> {
  const res = await fetch(`${API}/api/v1/voice/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice: voice || undefined }),
    signal,
  });
  if (!res.ok) {
    const detail = await res.json().then((d) => d?.detail).catch(() => null);
    throw new Error(detail ? `Read aloud failed: ${detail}` : "Speech is unavailable right now");
  }
  return res.blob();
}

export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1] ?? "");
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

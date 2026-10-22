import type { VoiceStatus } from "../voice/useVoiceSession";

export const STATUS_LABELS: Record<VoiceStatus, string> = {
  idle: "Tap to talk",
  requesting_permission: "Waiting for microphone",
  connecting: "Connecting…",
  listening: "Listening",
  transcribing: "Hearing you",
  thinking: "Finding an answer",
  speaking: "Speaking",
  interrupted: "Interrupted",
  error: "Voice stopped",
};

export const STATUS_HINTS: Record<VoiceStatus, string> = {
  idle: "Hands-free. Just talk, and interrupt whenever you like.",
  requesting_permission: "Allow microphone access in the browser prompt",
  connecting: "Opening the voice service",
  listening: "Just talk. I'll answer when you pause.",
  transcribing: "Keep going. I'll answer when you pause.",
  thinking: "Researching across sources…",
  speaking: "Start talking, press Space or tap the orb to interrupt",
  interrupted: "Go ahead, I'm listening",
  error: "",
};

/** States where tapping the orb (or pressing Space) stops the agent. */
export function canInterrupt(status: VoiceStatus) {
  return status === "speaking" || status === "thinking";
}

/** What to show under the orb: the agent's words while it speaks, otherwise yours. */
export function liveCaption(status: VoiceStatus, userCaption: string, agentCaption: string) {
  if (status === "speaking" || (status === "thinking" && agentCaption)) {
    return { text: agentCaption, agent: true };
  }
  if (status === "interrupted") return { text: "", agent: false };
  return { text: userCaption, agent: false };
}

export function seconds(ms?: number) {
  return ms === undefined ? "–" : `${(ms / 1000).toFixed(1)}s`;
}

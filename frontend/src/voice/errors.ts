export type VoiceError = {
  title: string;
  detail: string;
  /** Trying again can work without the user changing anything outside the app. */
  retryable: boolean;
};

/** Returns why live voice cannot run in this browser, or null when it can. */
export function voiceSupportProblem(): VoiceError | null {
  if (!window.isSecureContext) {
    return {
      title: "Voice needs a secure page",
      detail: "Browsers only allow the microphone on https:// or localhost. Open the app over https, or type instead.",
      retryable: false,
    };
  }
  if (!navigator.mediaDevices?.getUserMedia || typeof AudioWorkletNode === "undefined" || !window.AudioContext) {
    return {
      title: "This browser can't do live voice",
      detail: "Use a recent Chrome, Edge, Safari or Firefox, or type your question instead.",
      retryable: false,
    };
  }
  return null;
}

/** Turns a getUserMedia / audio setup failure into something a person can act on. */
export function microphoneError(err: unknown): VoiceError {
  const name = (err as { name?: string })?.name ?? "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return {
        title: "Microphone is blocked",
        detail:
          "Allow microphone access for this site (the icon at the left of the address bar), then try again.",
        retryable: true,
      };
    case "NotFoundError":
    case "OverconstrainedError":
      return {
        title: "No microphone found",
        detail: "Connect a microphone or headset, then try again.",
        retryable: true,
      };
    case "NotReadableError":
    case "AbortError":
      return {
        title: "Microphone is busy",
        detail: "Another app or tab is using the microphone. Close it, then try again.",
        retryable: true,
      };
    default:
      return {
        title: "Couldn't start the microphone",
        detail: "Something went wrong opening your microphone. Try again, or type instead.",
        retryable: true,
      };
  }
}

export function serviceError(detail: string): VoiceError {
  return { title: "Voice service problem", detail, retryable: true };
}

export const CONNECTION_LOST: VoiceError = {
  title: "Connection lost",
  detail: "Couldn't reach the voice service. Check that the server is running, then try again.",
  retryable: true,
};

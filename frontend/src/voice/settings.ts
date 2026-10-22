import { useCallback, useEffect, useState } from "react";

const KEY = "voice-agent.settings";

export type MotionPreference = "system" | "reduce" | "full";

export type VoiceSettings = {
  /** Server voice id; null uses the server default. */
  voice: string | null;
  /** Microphone deviceId; null uses the system default. */
  micId: string | null;
  motion: MotionPreference;
};

const DEFAULTS: VoiceSettings = { voice: null, micId: null, motion: "system" };

function load(): VoiceSettings {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) ?? "{}") };
  } catch {
    return DEFAULTS;
  }
}

const REDUCE_QUERY = "(prefers-reduced-motion: reduce)";

/** Whether animation should be toned down right now (read by rAF loops). */
export function motionReduced() {
  return document.documentElement.dataset.motion === "reduce";
}

export function useVoiceSettings() {
  const [settings, setSettings] = useState(load);
  const [systemReduce, setSystemReduce] = useState(() => window.matchMedia(REDUCE_QUERY).matches);

  useEffect(() => {
    const mql = window.matchMedia(REDUCE_QUERY);
    const onChange = () => setSystemReduce(mql.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);

  const reduce = settings.motion === "reduce" || (settings.motion === "system" && systemReduce);
  useEffect(() => {
    document.documentElement.dataset.motion = reduce ? "reduce" : "full";
  }, [reduce]);

  const update = useCallback((patch: Partial<VoiceSettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      localStorage.setItem(KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  return { settings, update, reduceMotion: reduce };
}

import { Loader2 } from "lucide-react";
import { useEffect, useRef, type ReactNode } from "react";

import { motionReduced } from "../voice/settings";
import type { VoiceStatus } from "../voice/useVoiceSession";

export const ORB_COLORS: Record<VoiceStatus, string> = {
  idle: "from-primary via-secondary to-sky-500",
  requesting_permission: "from-slate-300 via-slate-400 to-slate-500",
  connecting: "from-slate-300 via-slate-400 to-slate-500",
  listening: "from-primary via-secondary to-sky-500",
  transcribing: "from-primary via-indigo-500 to-secondary",
  thinking: "from-violet-500 via-purple-500 to-indigo-500",
  speaking: "from-sky-400 via-blue-500 to-primary",
  interrupted: "from-amber-400 via-orange-400 to-amber-500",
  error: "from-red-400 via-rose-500 to-red-600",
};

const SIZES = {
  lg: { box: "h-72 w-72", orb: "h-44 w-44", glow: "inset-6", ring: "inset-8", spinner: 32 },
  md: { box: "h-40 w-40", orb: "h-24 w-24", glow: "inset-4", ring: "inset-5", spinner: 22 },
  sm: { box: "h-12 w-12", orb: "h-10 w-10", glow: "-inset-1", ring: "-inset-0.5", spinner: 16 },
};

type Props = {
  status: VoiceStatus;
  level: () => number;
  size?: keyof typeof SIZES;
  onClick?: () => void;
  label: string;
  children?: ReactNode;
};

export default function Orb({ status, level, size = "lg", onClick, label, children }: Props) {
  const orbRef = useRef<HTMLButtonElement>(null);
  const glowRef = useRef<HTMLDivElement>(null);
  const s = SIZES[size];

  useEffect(() => {
    let frame = 0;
    let smooth = 0;
    const tick = () => {
      const target = motionReduced() ? 0 : Math.min(1, level() * 6);
      smooth += (target - smooth) * 0.25;
      if (orbRef.current) orbRef.current.style.transform = `scale(${(1 + smooth * 0.35).toFixed(3)})`;
      if (glowRef.current) glowRef.current.style.opacity = String(0.35 + smooth * 0.65);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [level]);

  const idle = status === "idle";

  return (
    <div className={`relative grid shrink-0 place-items-center ${s.box}`}>
      <div
        ref={glowRef}
        className={`absolute rounded-full bg-gradient-to-br blur-3xl transition-colors duration-700 ${s.glow} ${ORB_COLORS[status]}`}
      />
      {status === "thinking" && <div className={`orb-ring absolute rounded-full ${s.ring}`} />}
      {idle && <div className={`orb-halo absolute rounded-full ${s.ring}`} />}
      <button
        ref={orbRef}
        type="button"
        onClick={onClick}
        disabled={!onClick}
        aria-label={label}
        title={label}
        className={`orb relative grid place-items-center rounded-full bg-gradient-to-br shadow-2xl transition-colors duration-700 will-change-transform disabled:cursor-default ${s.orb} ${ORB_COLORS[status]} ${
          status === "thinking" || idle ? "orb-breathe" : ""
        } ${onClick ? "cursor-pointer" : ""}`}
      >
        {status === "connecting" || status === "requesting_permission" ? (
          <Loader2 size={s.spinner} className="animate-spin text-white/80" />
        ) : (
          children
        )}
      </button>
    </div>
  );
}

const CAPTURE_WORKLET = `
class PcmCapture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor("pcm-capture", PcmCapture);
`;

function rms(samples: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / (samples.length || 1));
}

export type Microphone = {
  sampleRate: number;
  level: () => number;
  setMuted: (muted: boolean) => void;
  stop: () => void;
};

/** Streams the microphone as 16-bit mono PCM in ~100 ms frames. */
export async function startMicrophone(
  onFrame: (pcm: ArrayBuffer) => void,
  deviceId?: string | null,
): Promise<Microphone> {
  const constraints: MediaTrackConstraints = {
    channelCount: 1,
    // Echo cancellation keeps the agent's own voice out of the transcript.
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  };
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: deviceId ? { ...constraints, deviceId: { exact: deviceId } } : constraints,
    });
  } catch (err) {
    // A saved device that was unplugged: fall back to the system default.
    if (!deviceId || !["OverconstrainedError", "NotFoundError"].includes((err as Error).name)) throw err;
    stream = await navigator.mediaDevices.getUserMedia({ audio: constraints });
  }
  const ctx = new AudioContext();
  const url = URL.createObjectURL(new Blob([CAPTURE_WORKLET], { type: "application/javascript" }));
  try {
    await ctx.audioWorklet.addModule(url);
  } catch (err) {
    stream.getTracks().forEach((t) => t.stop());
    void ctx.close();
    throw err;
  } finally {
    URL.revokeObjectURL(url);
  }

  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "pcm-capture");
  const sink = ctx.createGain();
  sink.gain.value = 0;
  source.connect(node);
  node.connect(sink).connect(ctx.destination);

  const frameSize = Math.round(ctx.sampleRate / 10);
  let pending = new Float32Array(frameSize);
  let filled = 0;
  let level = 0;
  let muted = false;

  node.port.onmessage = (event: MessageEvent<Float32Array>) => {
    const chunk = event.data;
    level = level * 0.7 + rms(chunk) * 0.3;
    let offset = 0;
    while (offset < chunk.length) {
      const take = Math.min(chunk.length - offset, frameSize - filled);
      pending.set(chunk.subarray(offset, offset + take), filled);
      filled += take;
      offset += take;
      if (filled === frameSize) {
        if (!muted) {
          const pcm = new Int16Array(frameSize);
          for (let i = 0; i < frameSize; i++) {
            const s = Math.max(-1, Math.min(1, pending[i]));
            pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
          onFrame(pcm.buffer);
        }
        pending = new Float32Array(frameSize);
        filled = 0;
      }
    }
  };

  return {
    sampleRate: ctx.sampleRate,
    level: () => (muted ? 0 : level),
    setMuted: (value) => {
      muted = value;
    },
    stop: () => {
      node.port.onmessage = null;
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close();
    },
  };
}

/** Plays streamed 16-bit PCM chunks back to back without gaps. */
export class PcmPlayer {
  private ctx = new AudioContext();
  private analyser: AnalyserNode;
  private nextTime = 0;
  private sources = new Set<AudioBufferSourceNode>();
  private ended = new Set<string>();
  private pending = new Map<string, number>();
  private cues: { turnId: string; run: () => void }[] = [];
  private timers = new Set<number>();
  private buffer = new Float32Array(1024);

  constructor(private onDrained: (turnId: string) => void) {
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.analyser.connect(this.ctx.destination);
  }

  async resume() {
    if (this.ctx.state !== "running") await this.ctx.resume();
  }

  enqueue(turnId: string, base64: string, sampleRate: number) {
    const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
    const samples = new Int16Array(bytes.buffer, 0, bytes.byteLength >> 1);
    if (!samples.length) return;
    const audio = this.ctx.createBuffer(1, samples.length, sampleRate);
    const channel = audio.getChannelData(0);
    for (let i = 0; i < samples.length; i++) channel[i] = samples[i] / 0x8000;

    const source = this.ctx.createBufferSource();
    source.buffer = audio;
    source.connect(this.analyser);
    // A small lead on the first chunk absorbs network jitter.
    const startAt = Math.max(this.nextTime, this.ctx.currentTime + 0.06);
    source.start(startAt);
    this.nextTime = startAt + audio.duration;
    const due = this.cues.filter((c) => c.turnId === turnId);
    this.cues = this.cues.filter((c) => c.turnId !== turnId);
    for (const cue of due) {
      const timer = window.setTimeout(
        () => {
          this.timers.delete(timer);
          cue.run();
        },
        Math.max(0, (startAt - this.ctx.currentTime) * 1000),
      );
      this.timers.add(timer);
    }
    this.sources.add(source);
    this.pending.set(turnId, (this.pending.get(turnId) ?? 0) + 1);
    source.onended = () => {
      this.sources.delete(source);
      const left = (this.pending.get(turnId) ?? 1) - 1;
      this.pending.set(turnId, left);
      if (left <= 0) this.checkDrained(turnId);
    };
  }

  /** Runs `run` when the next chunk queued for this turn starts playing. */
  cue(turnId: string, run: () => void) {
    this.cues.push({ turnId, run });
  }

  /** The server has sent every chunk for this turn. */
  markEnd(turnId: string) {
    this.ended.add(turnId);
    this.checkDrained(turnId);
  }

  private checkDrained(turnId: string) {
    if (this.ended.has(turnId) && (this.pending.get(turnId) ?? 0) <= 0) {
      this.ended.delete(turnId);
      this.pending.delete(turnId);
      this.onDrained(turnId);
    }
  }

  stop() {
    for (const source of this.sources) {
      source.onended = null;
      try {
        source.stop();
      } catch {
        /* already stopped */
      }
    }
    this.sources.clear();
    for (const timer of this.timers) window.clearTimeout(timer);
    this.timers.clear();
    this.cues = [];
    this.pending.clear();
    this.ended.clear();
    this.nextTime = 0;
  }

  get playing() {
    return this.sources.size > 0;
  }

  level(): number {
    if (!this.sources.size) return 0;
    this.analyser.getFloatTimeDomainData(this.buffer);
    return rms(this.buffer);
  }

  close() {
    this.stop();
    void this.ctx.close();
  }
}

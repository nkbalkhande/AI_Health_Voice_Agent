import { synthesize } from "../api";

const BOUNDARY = /[.!?]["')\]*_]*(?=\s)|\n+/g;
const PREFETCH = 2;
const CHARS_PER_SECOND = 15;

/** A speakable piece of the stream; `end` is its offset in the raw text. */
export type Sentence = { text: string; end: number };

/** Cuts streamed markdown into speakable sentences; code blocks are never spoken. */
export class SentenceSplitter {
  private raw = "";
  private pos = 0;
  private first = true;

  constructor(
    private minChars = 12,
    private maxChars = 220,
    private firstClauseChars = 40,
  ) {}

  get text() {
    return this.raw;
  }

  push(text: string): Sentence[] {
    this.raw += text;
    return this.drain(false);
  }

  flush(): Sentence[] {
    return this.drain(true);
  }

  private drain(final: boolean): Sentence[] {
    const out: Sentence[] = [];
    for (;;) {
      const fence = this.raw.indexOf("```", this.pos);
      const limit = fence >= 0 ? fence : this.raw.length;
      const region = this.raw.slice(this.pos, limit);
      const cut = this.nextCut(region);
      if (cut > 0) {
        this.emit(out, region.slice(0, cut), this.pos + cut);
        continue;
      }
      // Text right before a code fence is complete even without punctuation.
      if ((final || fence >= 0) && region.trim()) {
        this.emit(out, region, limit);
        continue;
      }
      if (fence < 0) break;
      const close = this.raw.indexOf("```", fence + 3);
      if (close >= 0) {
        this.pos = close + 3;
        continue;
      }
      if (final) this.pos = this.raw.length;
      break;
    }
    return out;
  }

  private emit(out: Sentence[], piece: string, end: number) {
    this.pos = end;
    const text = piece.trim();
    if (!text) return;
    out.push({ text, end });
    this.first = false;
  }

  private nextCut(region: string): number {
    BOUNDARY.lastIndex = 0;
    for (let m = BOUNDARY.exec(region); m; m = BOUNDARY.exec(region)) {
      const end = m.index + m[0].length;
      if (region.slice(0, end).trim().length >= this.minChars) return end;
    }
    // The first clause goes out early so speech starts as soon as possible.
    if (this.first) {
      const comma = region.indexOf(", ", this.firstClauseChars);
      if (comma >= 0) return comma + 1;
    }
    if (region.length > this.maxChars) {
      const head = region.slice(0, this.maxChars);
      const space = Math.max(head.lastIndexOf(", "), head.lastIndexOf(" "));
      return space > 0 ? space + 1 : this.maxChars;
    }
    return -1;
  }
}

type Item = { text: string; start: number; end: number };
type Fetched = { item: Item; blob: Promise<Blob | null> };

/**
 * Speaks an answer while it streams. Each finished sentence is synthesized
 * (a couple ahead of playback) and played in order, and the text is revealed
 * word by word in step with the audio, so the voice leads and the screen follows.
 */
export class StreamingSpeaker {
  private splitter = new SentenceSplitter();
  private items: Item[] = [];
  private ahead: Fetched[] = [];
  private lastEnd = 0;
  private shown = 0;
  private abort = new AbortController();
  private audio: HTMLAudioElement | null = null;
  private endPlayback: (() => void) | null = null;
  private playing = false;
  private finished = false;
  private stopped = false;
  private reported = false;

  constructor(
    private clean: (sentence: string) => string,
    /** How many characters of the raw answer have been spoken so far. */
    private onReveal: (chars: number) => void,
    private onEnd: () => void,
    private onError: (message: string) => void,
    private voice: string | null = null,
  ) {}

  push(text: string) {
    if (this.stopped) return;
    this.enqueue(this.splitter.push(text));
  }

  finish() {
    if (this.stopped) return;
    this.finished = true;
    this.enqueue(this.splitter.flush());
  }

  /** Silences playback and drops pending sentences; onEnd is not called. */
  stop() {
    if (this.stopped) return;
    this.stopped = true;
    this.abort.abort();
    this.items = [];
    this.ahead = [];
    this.audio?.pause();
    this.endPlayback?.();
  }

  private enqueue(sentences: Sentence[]) {
    for (const { text: raw, end } of sentences) {
      const text = this.clean(raw);
      // Unspeakable pieces fold into the next sentence's reveal range.
      if (!/[a-z0-9]/i.test(text)) continue;
      this.items.push({ text, start: this.lastEnd, end });
      this.lastEnd = end;
    }
    void this.pump();
  }

  private fill() {
    while (this.ahead.length < PREFETCH && this.items.length) {
      const item = this.items.shift()!;
      const blob = synthesize(item.text, this.abort.signal, this.voice).catch((err: Error) => {
        this.report(err.message);
        return null;
      });
      this.ahead.push({ item, blob });
    }
  }

  private async pump() {
    if (this.playing || this.stopped) return;
    this.playing = true;
    this.fill();
    while (!this.stopped && this.ahead.length) {
      const { item, blob: pending } = this.ahead.shift()!;
      const blob = await pending;
      this.fill();
      if (this.stopped) break;
      if (blob) await this.play(blob, item);
      this.revealTo(item.end);
    }
    this.playing = false;
    if (!this.stopped && this.finished && !this.items.length) {
      this.stopped = true;
      this.revealTo(this.splitter.text.length);
      this.onEnd();
    }
  }

  private play(blob: Blob, item: Item) {
    return new Promise<void>((resolve) => {
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      const seconds = () =>
        Number.isFinite(audio.duration) && audio.duration > 0
          ? audio.duration
          : Math.max(1, item.text.length / CHARS_PER_SECOND);
      const tick = () => {
        if (this.audio !== audio) return;
        const progress = Math.min(1, audio.currentTime / seconds());
        this.revealTo(item.start + (item.end - item.start) * progress, item.end);
        requestAnimationFrame(tick);
      };
      const done = () => {
        URL.revokeObjectURL(url);
        if (this.audio === audio) this.audio = null;
        this.endPlayback = null;
        resolve();
      };
      this.audio = audio;
      this.endPlayback = done;
      audio.onplaying = tick;
      audio.onended = done;
      audio.onerror = done;
      audio.play().catch(() => {
        this.report("Your browser blocked autoplay. Press the speaker icon to listen.");
        this.stop();
        this.onEnd();
      });
    });
  }

  /** Reveals up to `offset`, rounded forward to the end of the current word. */
  private revealTo(offset: number, limit = offset) {
    if (this.stopped && offset < this.splitter.text.length) return;
    const raw = this.splitter.text;
    let at = Math.min(Math.ceil(offset), limit);
    while (at < limit && !/\s/.test(raw[at])) at++;
    if (at <= this.shown) return;
    this.shown = at;
    this.onReveal(at);
  }

  private report(message: string) {
    if (this.stopped || this.reported) return;
    this.reported = true;
    this.onError(message);
  }
}

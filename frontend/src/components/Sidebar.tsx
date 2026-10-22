import {
  Check,
  MessageSquare,
  MoreHorizontal,
  PanelLeftClose,
  Pencil,
  Search,
  SquarePen,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import type { ConnectionState, ThreadSummary } from "../types";
import Logo from "./Logo";

type Props = {
  threads: ThreadSummary[];
  loading: boolean;
  activeId: string | null;
  connection: ConnectionState;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
  onClose: () => void;
  onSettings?: () => void;
};

const GROUPS = ["Today", "Yesterday", "Previous 7 days", "Previous 30 days", "Older"] as const;

function groupOf(iso: string): (typeof GROUPS)[number] {
  const date = new Date(iso);
  const startOfToday = new Date();
  startOfToday.setHours(0, 0, 0, 0);
  const days = (startOfToday.getTime() - date.getTime()) / 86_400_000;
  if (days <= 0) return "Today";
  if (days <= 1) return "Yesterday";
  if (days <= 7) return "Previous 7 days";
  if (days <= 30) return "Previous 30 days";
  return "Older";
}

export default function Sidebar({
  threads,
  loading,
  activeId,
  connection,
  onSelect,
  onNew,
  onRename,
  onDelete,
  onClose,
  onSettings,
}: Props) {
  const [query, setQuery] = useState("");
  const [menuFor, setMenuFor] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  const grouped = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = q ? threads.filter((t) => t.title.toLowerCase().includes(q)) : threads;
    const map = new Map<string, ThreadSummary[]>();
    for (const t of filtered) {
      const g = groupOf(t.updated_at);
      map.set(g, [...(map.get(g) ?? []), t]);
    }
    return GROUPS.filter((g) => map.has(g)).map((g) => [g, map.get(g)!] as const);
  }, [threads, query]);

  useEffect(() => {
    if (!menuFor) return;
    const close = () => setMenuFor(null);
    window.addEventListener("click", close);
    return () => window.removeEventListener("click", close);
  }, [menuFor]);

  return (
    <aside className="flex h-full w-[272px] flex-col border-r border-card-border bg-white">
      {/* Logo / header */}
      <div className="flex items-center justify-between px-3 pt-3 pb-2">
        <div className="flex items-center gap-2.5 px-1">
          <Logo size={32} />
          <span className="text-[15px] font-semibold tracking-tight text-slate-800">
            HealthMate AI
          </span>
        </div>
        <button onClick={onClose} className="icon-btn" title="Close sidebar">
          <PanelLeftClose size={18} />
        </button>
      </div>

      {/* Navigation */}
      <div className="space-y-0.5 px-3 pt-1">
        <button
          onClick={onNew}
          className="flex w-full items-center gap-2.5 rounded-lg bg-primary px-3 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-primary/90"
        >
          <SquarePen size={16} />
          New Conversation
        </button>
        <label className="mt-2 flex items-center gap-2.5 rounded-lg border border-card-border px-2.5 py-2 text-sm text-slate-400 focus-within:border-primary/30 focus-within:ring-2 focus-within:ring-primary/10">
          <Search size={16} />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search conversations"
            className="w-full bg-transparent text-slate-700 outline-none placeholder:text-slate-400"
          />
          {query && (
            <button onClick={() => setQuery("")} className="text-slate-400 hover:text-slate-600">
              <X size={14} />
            </button>
          )}
        </label>
      </div>

      {/* Conversation list */}
      <nav className="scrollbar-thin mt-3 flex-1 overflow-y-auto px-3 pb-3">
        {loading && threads.length === 0 && (
          <div className="space-y-2 px-2.5 pt-2">
            {[70, 55, 80, 60].map((w) => (
              <div key={w} className="h-4 animate-pulse rounded bg-slate-100" style={{ width: `${w}%` }} />
            ))}
          </div>
        )}
        {!loading && threads.length === 0 && (
          <p className="px-2.5 pt-2 text-xs leading-relaxed text-slate-400">
            Your conversations will appear here. Each one keeps its own memory.
          </p>
        )}
        {query && grouped.length === 0 && threads.length > 0 && (
          <p className="px-2.5 pt-2 text-xs text-slate-400">No conversations match "{query}".</p>
        )}

        {grouped.map(([group, items]) => (
          <div key={group} className="mb-4">
            <h3 className="sticky top-0 z-10 bg-white px-2.5 py-1.5 text-[11px] font-semibold tracking-[0.1em] uppercase text-slate-400">
              {group}
            </h3>
            <ul className="space-y-0.5">
              {items.map((t) => (
                <li key={t.thread_id} className="relative">
                  {editing === t.thread_id ? (
                    <RenameField
                      initial={t.title}
                      onDone={(title) => {
                        setEditing(null);
                        if (title && title !== t.title) onRename(t.thread_id, title);
                      }}
                    />
                  ) : confirmDelete === t.thread_id ? (
                    <div className="flex items-center gap-1 rounded-lg bg-red-50 px-2.5 py-1.5 text-sm">
                      <span className="flex-1 truncate text-red-600">Delete this chat?</span>
                      <button
                        className="rounded px-2 py-0.5 text-xs font-medium text-red-600 hover:bg-red-100"
                        onClick={() => {
                          setConfirmDelete(null);
                          onDelete(t.thread_id);
                        }}
                      >
                        Delete
                      </button>
                      <button
                        className="rounded px-2 py-0.5 text-xs text-slate-500 hover:bg-slate-100"
                        onClick={() => setConfirmDelete(null)}
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <div
                      className={`group flex items-center rounded-lg text-sm transition-colors ${
                        activeId === t.thread_id
                          ? "bg-primary-light text-primary font-medium"
                          : "text-slate-600 hover:bg-slate-50"
                      }`}
                    >
                      <button
                        onClick={() => onSelect(t.thread_id)}
                        className="flex min-w-0 flex-1 items-center gap-2.5 px-2.5 py-2 text-left"
                        title={t.title}
                      >
                        <MessageSquare size={14} className={`shrink-0 ${activeId === t.thread_id ? "text-primary" : "text-slate-400"}`} />
                        <span className="truncate">{t.title}</span>
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setMenuFor(menuFor === t.thread_id ? null : t.thread_id);
                        }}
                        className={`mr-1 rounded-md p-1 text-slate-400 hover:text-slate-700 ${
                          menuFor === t.thread_id || activeId === t.thread_id
                            ? "opacity-100"
                            : "opacity-0 group-hover:opacity-100"
                        }`}
                        title="Options"
                      >
                        <MoreHorizontal size={16} />
                      </button>
                    </div>
                  )}

                  {menuFor === t.thread_id && (
                    <div
                      onClick={(e) => e.stopPropagation()}
                      className="absolute right-1 top-9 z-20 w-40 overflow-hidden rounded-xl border border-card-border bg-white py-1 text-sm shadow-lg"
                    >
                      <button
                        className="flex w-full items-center gap-2 px-3 py-2 text-slate-600 hover:bg-slate-50"
                        onClick={() => {
                          setMenuFor(null);
                          setEditing(t.thread_id);
                        }}
                      >
                        <Pencil size={14} /> Rename
                      </button>
                      <button
                        className="flex w-full items-center gap-2 px-3 py-2 text-red-500 hover:bg-red-50"
                        onClick={() => {
                          setMenuFor(null);
                          setConfirmDelete(t.thread_id);
                        }}
                      >
                        <Trash2 size={14} /> Delete
                      </button>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="border-t border-card-border px-4 py-3">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span
            className={`h-2 w-2 rounded-full ${
              connection === "open"
                ? "bg-emerald-400"
                : connection === "connecting"
                  ? "animate-pulse bg-amber-400"
                  : "bg-red-500"
            }`}
          />
          {connection === "open"
            ? "Connected"
            : connection === "connecting"
              ? "Connecting…"
              : "Offline — retrying"}
        </div>
      </div>
    </aside>
  );
}

function RenameField({ initial, onDone }: { initial: string; onDone: (v: string) => void }) {
  const [value, setValue] = useState(initial);
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    ref.current?.select();
  }, []);
  return (
    <div className="flex items-center gap-1 rounded-lg bg-primary-light px-2 py-1">
      <input
        ref={ref}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") onDone(value.trim());
          if (e.key === "Escape") onDone("");
        }}
        onBlur={() => onDone(value.trim())}
        maxLength={120}
        className="min-w-0 flex-1 bg-transparent px-1 py-1 text-sm text-slate-800 outline-none"
      />
      <Check size={14} className="text-primary" />
    </div>
  );
}

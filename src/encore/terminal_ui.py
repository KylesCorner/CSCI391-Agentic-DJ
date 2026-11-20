"""Safe, event-driven terminal progress for Encore.

Shows observable actions, not private model chain-of-thought. The UI consumes
sanitized events already recorded by SessionTrace and never issues LLM calls.
Progress goes to stderr and the final answer goes to stdout.
"""
from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class TerminalUI:
    def __init__(self, *, quiet: bool = False, debug: bool = False,
                 console: Console | None = None, output: Console | None = None):
        self.quiet = quiet
        self.debug = debug
        self.console = console or Console(stderr=True, highlight=False)
        self.output = output or Console(stderr=False, highlight=False)
        self._last_tool: tuple[int, str] | None = None
        self._seen_end = False
        self._spinner = None

    @staticmethod
    def _clean(value: Any, limit: int = 125) -> str:
        """Don't allow untrusted event text to use terminal control sequences."""
        s = str(value if value is not None else "")
        return "".join(ch if ch.isprintable() else " " for ch in s)[:limit]

    def _line(self, marker: str, title: str, extra: str = "", *, style: str = "cyan") -> None:
        if self.quiet:
            return
        line = Text()
        line.append(marker + "  ", style=style)
        line.append(self._clean(title, 180), style="bold")
        if extra:
            line.append("  " + self._clean(extra, 340), style="dim")
        self.console.print(line)

    def start(self, goal: str) -> None:
        if not self.quiet:
            self.console.print(Panel(Text(self._clean(goal, 1800)), title="Encore · Live Activity",
                                     border_style="cyan", expand=False))
            self._line("…", "Interpreting the request")

    def on_event(self, record: dict[str, Any]) -> None:
        """Consume a *sanitized* SessionTrace record in emitted order."""
        if self.quiet:
            return
        event = str(record.get("event", ""))
        step = record.get("step")
        label = f"Step {step}" if isinstance(step, int) and step > 0 else "Agent"
        tool = self._clean(record.get("tool") or "tool")
        if event == "model_start":
            if self._spinner is not None:
                self._spinner.stop()
            phase = self._clean(record.get("phase") or "working")
            self._spinner = self.console.status(f"Encore: {phase}...", spinner="dots", spinner_style="cyan")
            self._spinner.start()
        elif event in {"model_end", "model_error"}:
            if self._spinner is not None:
                self._spinner.stop()
                self._spinner = None
            if event == "model_error":
                self._line("!", "Model request failed", str(record.get("error_category", "error")), style="yellow")
            elif self.debug:
                self._line("✓", "Model stage complete", self._clean(record.get("phase")), style="green")
        elif event == "session_start":
            if self.debug:
                self._line("·", "Session started", f"model={record.get('model')}  budget={record.get('max_steps')}")
        elif event == "route":
            domains = ", ".join(self._clean(x, 30) for x in (record.get("domains") or []))
            count = len(record.get("allowed_tools") or [])
            self._line("✓", "Request classified", f"domains: {domains or 'none'}  ·  {count} tools available", style="green")
            if record.get("router_fallback"):
                self._line("!", "Router fallback active", style="yellow")
            if self.debug:
                tools = ", ".join(self._clean(x, 48) for x in (record.get("allowed_tools") or []))
                self._line("·", "Tools offered", tools)
        elif event == "llm_call":
            phase = self._clean(record.get("phase") or label)
            if self.debug or (isinstance(step, int) and step > 0):
                tier = self._clean(record.get("tier") or "model")
                ms = record.get("latency_ms")
                timing = f" · {ms} ms" if isinstance(ms, (int, float)) else ""
                tokens = ""
                if self.debug:
                    tokens = f" · tokens {record.get('tokens_in', 0)}/{record.get('tokens_out', 0)}"
                self._line("◆", f"Model completed ({phase})", f"{tier}{timing}{tokens}", style="blue")
        elif event == "tool_attempt":
            if isinstance(step, int):
                self._last_tool = (step, tool)
            attempt = record.get("attempt")
            suffix = f"attempt {attempt}" if isinstance(attempt, int) and attempt > 1 else ""
            self._line("→", f"{label}: {tool}", suffix)
            if self.debug:
                args = record.get("args")
                if isinstance(args, dict):
                    # Deliberately don't print user text, token values, or raw arguments.
                    self._line("·", "Argument fields", ", ".join(str(k) for k in args.keys()))
        elif event == "tool_result":
            ok = record.get("ok")
            if ok is True or (ok is None and record.get("success") is True):
                self._line("✓", f"{tool} succeeded", style="green")
            else:
                category = record.get("error_category") or record.get("error_type") or "failed"
                self._line("!", f"{tool} failed", self._clean(category), style="yellow")
        elif event == "tool_call":
            # tool_attempt/result already reported the action; show only useful
            # factual result metadata rather than redundant tool calls.
            if record.get("tool") in {"none", "final_answer"}:
                return
            detail_parts = []
            for key in ("returned_recordings", "returned_artists", "returned_tracks",
                        "selected_track_count", "required_track_count", "minutes",
                        "target_minutes", "confirmed", "ready"):
                if record.get(key) is not None:
                    detail_parts.append(f"{key.replace('_', ' ')}: {self._clean(record[key], 30)}")
            if detail_parts:
                self._line("·", f"{tool} result", " · ".join(detail_parts))
            elif self.debug and record.get("message"):
                self._line("·", f"{tool} detail", self._clean(record["message"], 140))
        elif event == "retry":
            self._line("↻", f"Retrying {tool}", f"attempt {record.get('attempt', '?')}", style="yellow")
        elif event == "memory_read":
            if self.debug:
                self._line("·", "Retrieved prior session context", f"ok={record.get('ok')}")
        elif event == "compress":
            self._line("↳", "Compressing working memory")
        elif event == "approval":
            approved = bool(record.get("approved"))
            self._line("✓" if approved else "!", "Account change approved" if approved else "Account change declined",
                       style="green" if approved else "yellow")
        elif event == "memory_summary_start":
            self._line("…", "Summarizing this session for memory")
        elif event == "memory_summary_end":
            self._line("✓" if record.get("success") else "!", "Session summary saved" if record.get("success") else "Session summary unavailable",
                       style="green" if record.get("success") else "yellow")
        elif event == "memory_write":
            good = bool(record.get("success"))
            self._line("✓" if good else "!", "Episodic memory updated" if good else "Episodic memory write failed",
                       style="green" if good else "yellow")
        elif event == "final":
            status = self._clean(record.get("status", "unknown"))
            self._line("✓" if status == "answered" else "·", "Agent finalized", f"status={status}",
                       style="green" if status == "answered" else "yellow")
        elif event == "session_end":
            self._seen_end = True
            seconds = record.get("latency_s")
            duration = f"{float(seconds):.2f}s" if isinstance(seconds, (int, float)) else ""
            self._line("✓", "Session complete", duration, style="green")
        elif self.debug:
            # Unknown events are useful when diagnosing the existing executor.
            self._line("·", self._clean(event), f"step={step or '-'}")

    def warning(self, message: str) -> None:
        if not self.quiet:
            self._line("!", message, style="yellow")

    def close(self) -> None:
        if self._spinner is not None:
            self._spinner.stop()
            self._spinner = None

    def result(self, result: Any, trace_path: Any) -> None:
        """Print the answer to stdout, never mixing it with progress on stderr."""
        answer = str(getattr(result, "answer", "") or "")
        if self.quiet:
            self.output.print(Text(answer))
            return
        self.output.print(Panel(Text(answer), title="Encore · Answer", border_style="green"))
        status = self._clean(getattr(result, "status", "unknown"))
        confidence = self._clean(getattr(result, "confidence", "unknown"))
        steps = getattr(result, "steps", 0)
        seconds = getattr(result, "seconds", 0)
        self.output.print(Text(f"Status: {status}  ·  Confidence: {confidence}  ·  Actions: {steps}  ·  {seconds}s", style="dim"))
        sources = getattr(result, "sources", []) or []
        if sources:
            self.output.print(Text("Sources: " + ", ".join(self._clean(s, 180) for s in sources[:15]), style="dim"))
        self.output.print(Text(f"Trace saved: {trace_path}", style="dim"))
        if self.debug:
            self.output.print(Text(f"Ollama tokens: prompt={getattr(result, 'prompt_tokens', 0)}, generated={getattr(result, 'generated_tokens', 0)}", style="dim"))


class ObservableOllama:
    """Factory: return an actual Ollama subclass so isinstance checks still work.

    Kept as a factory instead of a proxy; run_agent explicitly checks
    isinstance(model, Ollama) to select the final-answer model tier.
    """

    def __new__(cls, *args, on_model_event=None, **kwargs):
        from .ollama_client import Ollama

        class _TracedOllama(Ollama):
            def __init__(self):
                super().__init__(*args, **kwargs)
                self._on_model_event = on_model_event

            def _call(self, name, invoke, *call_args, **call_kwargs):
                if self._on_model_event:
                    self._on_model_event("model_start", phase=name, model=self.model)
                try:
                    result = invoke(*call_args, **call_kwargs)
                except Exception as exc:
                    if self._on_model_event:
                        self._on_model_event("model_error", phase=name, error_category=type(exc).__name__)
                    raise
                if self._on_model_event:
                    self._on_model_event("model_end", phase=name)
                return result

            def extract_intent(self, goal):
                return self._call("interpreting request", super().extract_intent, goal)

            def route_domains(self, goal):
                return self._call("selecting tools", super().route_domains, goal)

            def chat(self, messages, tools, *, tier="cheap"):
                return self._call("generating response" if tier == "expensive" else "deciding next action",
                                  super().chat, messages, tools, tier=tier)

            def review_final(self, goal, proposed, evidence):
                return self._call("validating final answer", super().review_final,
                                  goal, proposed, evidence)

            def summarize_working_memory(self, original_goal, earlier_turns):
                return self._call("compressing context", super().summarize_working_memory,
                                  original_goal, earlier_turns)

        return _TracedOllama()

"""TracingFM: FM subclass that logs every LLM call for debug/inspection.

Usage:
    from social_omni_epic.tracing_fm import TracingFM

    tfm = TracingFM(model="gpt-4.1", show_prompts=True, show_responses=True)
    tfm.set_step("Step 2: Verbalized Sampling")
    result = tfm.query(system, user)          # prints a panel; records the trace
    traces = tfm.get_traces()                 # list[dict] for JSON export
"""
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .fm import FM

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None


# ---------------------------------------------------------------------------
# Trace record
# ---------------------------------------------------------------------------

@dataclass
class LLMCallTrace:
    step: str
    call_index: int
    method: str           # query | query_json | get_embeddings
    model: str
    temperature: Optional[float]
    system_prompt: str
    user_prompt: str
    response: Any         # str | dict | "embeddings:<N>x<D>"
    elapsed_ms: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if isinstance(d["response"], list):
            # embeddings — just store shape summary
            rows = len(d["response"])
            cols = len(d["response"][0]) if rows > 0 and d["response"][0] else 0
            d["response"] = f"embeddings:{rows}x{cols}"
        return d


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _trunc(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"… [+{len(text) - max_chars} chars]"


def _print_trace_rich(trace: LLMCallTrace, max_chars: int) -> None:
    title = (
        f"LLM call #{trace.call_index} · {trace.method} · "
        f"{trace.model} · temp={trace.temperature}"
    )
    lines: list[str] = []
    if trace.error:
        lines.append(f"[bold red]ERROR:[/bold red] {trace.error}")
    if trace.system_prompt:
        s = _trunc(trace.system_prompt, max_chars)
        lines.append(f"[dim]System ({len(trace.system_prompt)} chars):[/dim] {s}")
    if trace.user_prompt:
        u = _trunc(trace.user_prompt, max_chars)
        lines.append(f"[dim]User   ({len(trace.user_prompt)} chars):[/dim] {u}")
    # separator + response
    lines.append(f"[dim]{'─'*40} {trace.elapsed_ms:.0f}ms[/dim]")
    if isinstance(trace.response, list):
        rows = len(trace.response)
        cols = len(trace.response[0]) if rows and trace.response[0] else 0
        lines.append(f"[green]Response:[/green] embeddings {rows}×{cols}")
    elif trace.response is not None:
        resp_str = str(trace.response)
        lines.append(f"[green]Response:[/green] {_trunc(resp_str, max_chars)}")
    _console.print(Panel("\n".join(lines), title=title, border_style="blue"))


def _print_trace_plain(trace: LLMCallTrace, max_chars: int) -> None:
    print(f"\n  ┌─ LLM call #{trace.call_index} · {trace.method} · {trace.model} ─")
    if trace.error:
        print(f"  │ ERROR: {trace.error}")
    if trace.system_prompt:
        print(f"  │ System ({len(trace.system_prompt)} chars): {_trunc(trace.system_prompt, max_chars)}")
    if trace.user_prompt:
        print(f"  │ User   ({len(trace.user_prompt)} chars): {_trunc(trace.user_prompt, max_chars)}")
    print(f"  │ {'─'*36} {trace.elapsed_ms:.0f}ms")
    if isinstance(trace.response, list):
        rows = len(trace.response)
        cols = len(trace.response[0]) if rows and trace.response[0] else 0
        print(f"  │ Response: embeddings {rows}×{cols}")
    elif trace.response is not None:
        print(f"  │ Response: {_trunc(str(trace.response), max_chars)}")
    print("  └─")


# ---------------------------------------------------------------------------
# Step header helpers
# ---------------------------------------------------------------------------

def print_step(label: str) -> None:
    if _RICH:
        _console.print(Rule(f"[bold cyan]{label}[/bold cyan]", style="cyan"))
    else:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")


def print_info(msg: str) -> None:
    if _RICH:
        _console.print(f"  [green]✓[/green] {msg}")
    else:
        print(f"  ✓ {msg}")


def print_warn(msg: str) -> None:
    if _RICH:
        _console.print(f"  [yellow]⚠[/yellow]  {msg}")
    else:
        print(f"  ⚠  {msg}")


def print_section(title: str, body: str) -> None:
    if _RICH:
        _console.print(Panel(body, title=title, border_style="dim"))
    else:
        print(f"\n  --- {title} ---")
        for line in body.splitlines():
            print(f"  {line}")


# ---------------------------------------------------------------------------
# TracingFM
# ---------------------------------------------------------------------------

class TracingFM(FM):
    def __init__(
        self,
        *args,
        show_prompts: bool = True,
        show_responses: bool = True,
        max_prompt_chars: int = 800,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.show_prompts = show_prompts
        self.show_responses = show_responses
        self.max_prompt_chars = max_prompt_chars
        self._current_step: str = "?"
        self._call_counter: int = 0
        self._traces: list[LLMCallTrace] = []

    def set_step(self, label: str) -> None:
        self._current_step = label
        print_step(label)

    def get_traces(self) -> list[dict]:
        return [t.to_dict() for t in self._traces]

    def _record(
        self,
        method: str,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float],
        response: Any,
        elapsed_ms: float,
        error: Optional[str] = None,
    ) -> LLMCallTrace:
        self._call_counter += 1
        trace = LLMCallTrace(
            step=self._current_step,
            call_index=self._call_counter,
            method=method,
            model=self.model,
            temperature=temperature if temperature is not None else self.temperature,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response,
            elapsed_ms=elapsed_ms,
            error=error,
        )
        self._traces.append(trace)

        if self.show_prompts or self.show_responses:
            t_display = LLMCallTrace(
                step=trace.step,
                call_index=trace.call_index,
                method=trace.method,
                model=trace.model,
                temperature=trace.temperature,
                system_prompt=trace.system_prompt if self.show_prompts else "",
                user_prompt=trace.user_prompt if self.show_prompts else "",
                response=trace.response if self.show_responses else None,
                elapsed_ms=trace.elapsed_ms,
                error=trace.error,
            )
            if _RICH:
                _print_trace_rich(t_display, self.max_prompt_chars)
            else:
                _print_trace_plain(t_display, self.max_prompt_chars)
        return trace

    def query(self, system_prompt: str, user_prompt: str,
              temperature: Optional[float] = None) -> str:
        t0 = time.monotonic()
        error = None
        response = None
        try:
            response = super().query(system_prompt, user_prompt, temperature)
            return response
        except Exception as e:
            error = str(e)
            raise
        finally:
            elapsed = (time.monotonic() - t0) * 1000
            self._record("query", system_prompt, user_prompt, temperature,
                         response, elapsed, error)

    def query_json(self, system_prompt: str, user_prompt: str,
                   temperature: Optional[float] = None) -> dict:
        t0 = time.monotonic()
        error = None
        response = None
        try:
            response = super().query_json(system_prompt, user_prompt, temperature)
            return response
        except Exception as e:
            error = str(e)
            raise
        finally:
            elapsed = (time.monotonic() - t0) * 1000
            self._record("query_json", system_prompt, user_prompt, temperature,
                         response, elapsed, error)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        t0 = time.monotonic()
        error = None
        response = None
        try:
            response = super().get_embeddings(texts)
            return response
        except Exception as e:
            error = str(e)
            raise
        finally:
            elapsed = (time.monotonic() - t0) * 1000
            self._record(
                "get_embeddings",
                f"embed {len(texts)} text(s)",
                texts[0][:120] + ("…" if texts and len(texts[0]) > 120 else "") if texts else "",
                None,
                response,
                elapsed,
                error,
            )

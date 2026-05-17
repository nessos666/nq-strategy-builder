from __future__ import annotations

import datetime
import math
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from sb.models import EvalResult, ParsedIdea, WalkForwardResult

console = Console()


def generate_report(
    idea: ParsedIdea, results: list[EvalResult], output_dir: Path
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_path = output_dir / f"report_{timestamp}.md"
    _print_terminal(idea, results)
    report_path.write_text(_build_markdown(idea, results, timestamp), encoding="utf-8")
    console.print(f"\n[bold green]Report gespeichert:[/bold green] {report_path}")
    return report_path


def _escape_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _wf_window_status(wf_window_oos_pf: float, num_trades: int) -> tuple[str, str]:
    if num_trades <= 0:
        return "[yellow]–[/yellow]", "–"
    if wf_window_oos_pf > 1.0:
        return "[green]✅[/green]", "✅"
    return "[red]❌[/red]", "❌"


def _print_terminal(idea: ParsedIdea, results: list[EvalResult]) -> None:
    console.print("\n[bold]Strategie Builder – Ergebnis[/bold]")
    console.print(f"Idee: [cyan]{idea.raw}[/cyan]")
    console.print(
        f"Konzepte: {', '.join(idea.concepts) or 'keine erkannt'} | Session: {idea.session}\n"
    )
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    for col, justify in [
        ("Rang", "center"),
        ("PF", "right"),
        ("Winrate", "right"),
        ("Trades", "right"),
        ("SL", "right"),
        ("TP-Mult", "right"),
        ("Score", "right"),
        ("Warnungen", "left"),
    ]:
        table.add_column(col, justify=justify)  # type: ignore[arg-type]
    for ev in results[:10]:
        r = ev.result
        warn_str = ev.warnings[0][:28] if ev.warnings else "–"
        color = "green" if ev.rank == 1 else ("yellow" if ev.rank <= 3 else "white")
        table.add_row(
            f"[{color}]#{ev.rank}[/{color}]",
            f"[{color}]{r.profit_factor:.2f}[/{color}]",
            f"{r.winrate:.0%}",
            str(r.num_trades),
            str(r.params.get("sl_points", "?")),
            str(r.params.get("tp_mult", "?")),
            f"{ev.score:.3f}",
            warn_str,
        )
    console.print(table)


def _build_markdown(idea: ParsedIdea, results: list[EvalResult], timestamp: str) -> str:
    best = results[0] if results else None
    lines = [
        f"# Strategie Builder Report – {timestamp}",
        "",
        f"**Idee:** {_escape_markdown_cell(idea.raw)}  ",
        f"**Konzepte:** {_escape_markdown_cell(', '.join(idea.concepts) or 'keine erkannt')}  ",
        f"**Session:** {_escape_markdown_cell(idea.session)}  ",
        "",
        "---",
        "",
        "## Bestes Ergebnis",
        "",
    ]
    if best:
        r = best.result
        lines += [
            "| Parameter | Wert |",
            "|-----------|------|",
            f"| Profit Factor | **{r.profit_factor:.2f}** |",
            f"| Winrate | {r.winrate:.0%} |",
            f"| Trades | {r.num_trades} |",
            f"| SL (Punkte) | {_escape_markdown_cell(r.params.get('sl_points', '?'))} |",
            f"| TP-Multiplikator | {_escape_markdown_cell(r.params.get('tp_mult', '?'))} |",
            f"| Session | {_escape_markdown_cell(r.params.get('session', idea.session))} |",
            f"| Score | {best.score:.4f} |",
            "",
        ]
        if best.warnings:
            lines.append("**Warnungen:**")
            for w in best.warnings:
                lines.append(f"- ⚠️ {_escape_markdown_cell(w)}")
            lines.append("")
    lines += [
        "---",
        "",
        "## Top 10 Ergebnisse",
        "",
        "| Rang | PF | Winrate | Trades | SL | TP-Mult | Score | Warnungen |",
        "|------|-----|---------|--------|-----|---------|-------|-----------|",
    ]
    for ev in results[:10]:
        r = ev.result
        warn = _escape_markdown_cell(ev.warnings[0][:40] if ev.warnings else "–")
        lines.append(
            f"| #{ev.rank} | {r.profit_factor:.2f} | {r.winrate:.0%} | {r.num_trades} | {_escape_markdown_cell(r.params.get('sl_points', '?'))} | {_escape_markdown_cell(r.params.get('tp_mult', '?'))} | {ev.score:.3f} | {warn} |"
        )
    lines += [
        "",
        "---",
        "*Generiert von Strategie Builder – kein LLM, kein Claude, 100% standalone*",
    ]
    return "\n".join(lines)


def generate_wf_report(
    idea: ParsedIdea,
    wf_result: WalkForwardResult,
    output_dir: Path,
    mc_pct_profitable: float | None = None,
) -> Path:
    """Erzeugt Terminal-Ausgabe + Report.md für WalkForwardResult."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_path = output_dir / f"report_{timestamp}.md"
    _print_walk_forward(idea, wf_result, mc_pct_profitable=mc_pct_profitable)
    report_path.write_text(
        _wf_markdown(idea, wf_result, timestamp, mc_pct_profitable=mc_pct_profitable),
        encoding="utf-8",
    )
    console.print(f"\n[bold green]Report gespeichert:[/bold green] {report_path}")
    return report_path


def _print_walk_forward(
    idea: ParsedIdea,
    wf: WalkForwardResult,
    mc_pct_profitable: float | None = None,
) -> None:
    """Rich-Terminal-Ausgabe für Walk-Forward-Ergebnis."""
    console.print("\n[bold]Strategie Builder – Walk-Forward Ergebnis[/bold]")
    console.print(f"Idee: [cyan]{idea.raw}[/cyan]")
    console.print(
        f"Konzepte: {', '.join(idea.concepts) or 'keine erkannt'} | Session: {idea.session}\n"
    )

    # Fenster-Tabelle
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        title="Out-of-Sample pro Fenster",
    )
    for col, justify in [
        ("Fenster", "center"),
        ("IS PF", "right"),
        ("OOS PF", "right"),
        ("OOS WR", "right"),
        ("OOS Trades", "right"),
        ("Status", "center"),
    ]:
        table.add_column(col, justify=justify)  # type: ignore[arg-type]

    for w in wf.windows:
        status, _ = _wf_window_status(w.oos.profit_factor, w.oos.num_trades)
        table.add_row(
            f"#{w.window_idx + 1}",
            f"{w.in_sample.profit_factor:.2f}",
            f"{w.oos.profit_factor:.2f}",
            f"{w.oos.winrate:.0%}",
            str(w.oos.num_trades),
            status,
        )

    # Gesamt-Zeile
    robust_label = (
        "[green]ROBUST[/green]" if wf.is_robust else "[red]NICHT ROBUST[/red]"
    )
    table.add_section()
    table.add_row(
        "[bold]Gesamt[/bold]",
        "–",
        f"[bold]{wf.oos_pf:.2f}[/bold]",
        f"[bold]{wf.oos_winrate:.0%}[/bold]",
        f"[bold]{wf.oos_trades}[/bold]",
        robust_label,
    )
    console.print(table)

    # Parameter-Wichtigkeit
    if wf.importances:
        console.print(
            "\n[bold]Parameter-Wichtigkeit (PED-ANOVA, Ø über alle Fenster):[/bold]"
        )
        sorted_imp = sorted(wf.importances.items(), key=lambda x: x[1], reverse=True)
        for param, imp in sorted_imp:
            bar_len = max(1, int(imp * 30))
            bar = "█" * bar_len
            console.print(f"  {param:<22} {bar} {imp:.0%}")

    # PBO Score
    if not math.isnan(wf.pbo_score):
        pbo_color = "green" if wf.pbo_score < 0.5 else "red"
        pbo_label = "OK" if wf.pbo_score < 0.5 else "OVERFITTING"
        console.print(
            f"\n[bold]PBO-Score:[/bold] [{pbo_color}]{wf.pbo_score:.1%} ({pbo_label})[/{pbo_color}]"
        )

    # Monte Carlo Score
    if mc_pct_profitable is not None:
        if mc_pct_profitable >= 0.8:
            mc_color = "green"
        elif mc_pct_profitable >= 0.6:
            mc_color = "yellow"
        else:
            mc_color = "red"
        console.print(
            f"\n[bold]Monte Carlo Robustheit:[/bold] [{mc_color}]{mc_pct_profitable:.0%}[/{mc_color}]"
        )

    # Beste Params (aus letztem Fenster)
    if wf.best_params:
        bp = wf.best_params
        console.print(
            f"\n[bold]Empfohlene Parameter[/bold] (aus Fenster #{len(wf.windows)}):"
        )
        console.print(
            f"  SL: {bp.get('sl_points', '?')} Punkte | TP-Mult: {bp.get('tp_mult', '?')} | Offset: {bp.get('entry_bar_offset', '?')}"
        )


def _wf_markdown(
    idea: ParsedIdea,
    wf: WalkForwardResult,
    timestamp: str,
    mc_pct_profitable: float | None = None,
) -> str:
    """Markdown-Report für WalkForwardResult."""
    robust_str = (
        "✅ ROBUST (alle Fenster OOS PF > 1.0)"
        if wf.is_robust
        else "❌ NICHT ROBUST (mind. 1 Fenster versagt)"
    )
    if not math.isnan(wf.pbo_score):
        pbo_label = "OK (< 50%)" if wf.pbo_score < 0.5 else "OVERFITTING (≥ 50%)"
        pbo_str = f"**PBO-Score:** {wf.pbo_score:.1%} ({pbo_label})  "
    else:
        pbo_str = ""
    lines = [
        f"# Strategie Builder Report – {timestamp}",
        "",
        f"**Idee:** {_escape_markdown_cell(idea.raw)}  ",
        f"**Konzepte:** {_escape_markdown_cell(', '.join(idea.concepts) or 'keine erkannt')}  ",
        f"**Session:** {_escape_markdown_cell(idea.session)}  ",
        f"**Robustheit:** {robust_str}  ",
    ]
    if pbo_str:
        lines.append(pbo_str)
    if mc_pct_profitable is not None:
        mc_str = f"**Monte Carlo Robustheit:** {mc_pct_profitable:.0%}  "
        lines.append(mc_str)
    lines += [
        "",
        "---",
        "",
        "## Walk-Forward Ergebnis (Out-of-Sample)",
        "",
        "| Fenster | IS PF | OOS PF | OOS WR | OOS Trades | Status |",
        "|---------|-------|--------|--------|------------|--------|",
    ]
    for w in wf.windows:
        _, status = _wf_window_status(w.oos.profit_factor, w.oos.num_trades)
        lines.append(
            f"| #{w.window_idx + 1} | {w.in_sample.profit_factor:.2f} | **{w.oos.profit_factor:.2f}** | {w.oos.winrate:.0%} | {w.oos.num_trades} | {status} |"
        )
    lines += [
        f"| **Gesamt** | – | **{wf.oos_pf:.2f}** | **{wf.oos_winrate:.0%}** | **{wf.oos_trades}** | {robust_str.split()[0]} |",
        "",
        "---",
        "",
        "## Parameter-Wichtigkeit (PED-ANOVA)",
        "",
        "| Parameter | Wichtigkeit |",
        "|-----------|-------------|",
    ]
    for param, imp in sorted(wf.importances.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {_escape_markdown_cell(param)} | {imp:.0%} |")

    if wf.best_params:
        bp = wf.best_params
        lines += [
            "",
            "---",
            "",
            "## Empfohlene Parameter (aus letztem Fenster)",
            "",
            "| Parameter | Wert |",
            "|-----------|------|",
            f"| SL (Punkte) | {_escape_markdown_cell(bp.get('sl_points', '?'))} |",
            f"| TP-Multiplikator | {_escape_markdown_cell(bp.get('tp_mult', '?'))} |",
            f"| Entry-Bar-Offset | {_escape_markdown_cell(bp.get('entry_bar_offset', '?'))} |",
            f"| Session | {_escape_markdown_cell(idea.session)} |",
        ]
    lines += [
        "",
        "---",
        "*Generiert von Strategie Builder – kein LLM, kein Claude, 100% standalone*",
    ]
    return "\n".join(lines)

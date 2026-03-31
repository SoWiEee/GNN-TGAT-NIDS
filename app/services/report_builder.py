"""Report builder: Jinja2 → HTML → WeasyPrint PDF.

All heavy operations run synchronously (called via run_in_threadpool from the router).
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def build_report(
    session_id: UUID,
    result: dict,
    graph_png_b64: str,
    session_dir: Path,
) -> Path:
    """Render Jinja2 template to HTML and convert to PDF with WeasyPrint.

    Returns the path to the generated HTML file.  PDF generation requires
    WeasyPrint which may not be available in all environments; if import
    fails, only the HTML report is produced.
    """
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    template = env.get_template("report.html.j2")

    meta = result.get("meta", {})
    alerts = result.get("alerts", [])
    reliability_path = Path("data/metrics/reliability.json")
    reliability = (
        json.loads(reliability_path.read_text()) if reliability_path.exists() else {}
    )

    context = {
        "session_id": str(session_id),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "filename": meta.get("csv_path", "unknown"),
        "total_flows": meta.get("total_flows", 0),
        "total_alerts": meta.get("total_alerts", 0),
        "n_windows": meta.get("n_windows", 0),
        "top_alerts": sorted(alerts, key=lambda a: a.get("confidence", 0), reverse=True)[:20],
        "graph_png_b64": graph_png_b64,
        "reliability": reliability,
    }

    html_content = template.render(**context)
    html_path = session_dir / "report.html"
    html_path.write_text(html_content, encoding="utf-8")

    # Attempt PDF generation (optional dependency)
    pdf_path = session_dir / "report.pdf"
    try:
        from weasyprint import HTML as WeasyHTML
        WeasyHTML(string=html_content, base_url=str(session_dir)).write_pdf(str(pdf_path))
    except ImportError:
        pass  # WeasyPrint not installed; HTML report still available
    except Exception:
        pass  # PDF generation failed silently; HTML still available

    return html_path

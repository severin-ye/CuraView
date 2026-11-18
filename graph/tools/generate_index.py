#!/usr/bin/env python
"""Generate an index page for all patient visualizations."""

from __future__ import annotations

import argparse
from html import escape
from urllib.parse import quote
from pathlib import Path


def scan_visualizations(visual_output_dir: Path) -> list[dict]:
    """Scan the visual output directory for patient visualizations."""
    patients = []

    for patient_dir in sorted(visual_output_dir.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_num = patient_dir.name.replace("patient_", "")

        graph_viz = patient_dir / "graph_visualization.html"
        community_viz = patient_dir / "community_visualization.html"

        if graph_viz.exists() or community_viz.exists():
            patients.append({
                'number': patient_num,
                'name': patient_dir.name,
                'has_graph': graph_viz.exists(),
                'has_community': community_viz.exists(),
            })

    return patients


def render_index_html(patients: list[dict]) -> str:
    """Render the index HTML page."""

    patient_rows = ""
    for p in patients:
        url_name = quote(p["name"], safe="")
        graph_link = f'<a href="{url_name}/graph_visualization.html" target="_blank" rel="noopener">📊 知识图谱</a>' if p['has_graph'] else '<span style="color: #ccc;">-</span>'
        community_link = f'<a href="{url_name}/community_visualization.html" target="_blank" rel="noopener">🏘️ 社区报告</a>' if p['has_community'] else '<span style="color: #ccc;">-</span>'

        patient_rows += f"""
        <tr>
            <td>患者 {escape(p['number'])}</td>
            <td>{graph_link}</td>
            <td>{community_link}</td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GraphRAG 可视化索引</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 32px;
        }}
        .header p {{
            color: #666;
            font-size: 16px;
        }}
        .content {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 15px;
            border-bottom: 1px solid #e9ecef;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        a {{
            color: #667eea;
            text-decoration: none;
            padding: 8px 15px;
            border-radius: 5px;
            display: inline-block;
            transition: all 0.3s ease;
        }}
        a:hover {{
            background: #667eea;
            color: white;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #e9ecef;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
            font-size: 14px;
        }}
        .empty-state {{
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }}
        .empty-state-icon {{
            font-size: 64px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 GraphRAG 可视化索引</h1>
            <p>患者知识图谱与社区报告可视化导航</p>
        </div>

        <div class="content">
            {"<table>" if patients else ""}
            {"<thead><tr><th>患者</th><th>知识图谱</th><th>社区报告</th></tr></thead>" if patients else ""}
            {"<tbody>" if patients else ""}
            {patient_rows if patients else '<div class="empty-state"><div class="empty-state-icon">📭</div><p>暂无可视化数据</p><p style="margin-top: 10px; color: #999;">请先运行可视化脚本生成数据</p></div>'}
            {"</tbody>" if patients else ""}
            {"</table>" if patients else ""}

            {f'''<div class="stats">
                <div class="stat-item">
                    <div class="stat-value">{len(patients)}</div>
                    <div class="stat-label">患者总数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{sum(1 for p in patients if p["has_graph"])}</div>
                    <div class="stat-label">知识图谱</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{sum(1 for p in patients if p["has_community"])}</div>
                    <div class="stat-label">社区报告</div>
                </div>
            </div>''' if patients else ''}
        </div>
    </div>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualization index page.")
    parser.add_argument("--visual-dir", required=True, type=Path)
    args = parser.parse_args()
    patients = scan_visualizations(args.visual_dir)
    with (args.visual_dir / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(render_index_html(patients))


if __name__ == "__main__":
    main()

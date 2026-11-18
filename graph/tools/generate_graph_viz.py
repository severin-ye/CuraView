#!/usr/bin/env python
"""Generate an interactive knowledge graph visualization (D3.js HTML).

Default input: ../../output/entities.parquet and relationships.parquet
Default output: ./output/graph_visualization.html
Reference: graphrag-ollama/ml-ollama/_core/Visualization/Visualization.ipynb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from graph.io import write_json



def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    entities_path = data_dir / "entities.parquet"
    rels_path = data_dir / "relationships.parquet"

    if not entities_path.exists() or not rels_path.exists():
        missing = []
        if not entities_path.exists():
            missing.append(str(entities_path))
        if not rels_path.exists():
            missing.append(str(rels_path))
        raise FileNotFoundError(
            f"Missing required file(s): {', '.join(missing)}. "
            "Run the indexing pipeline first or point --data-dir to a folder that contains them."
        )

    entities = pd.read_parquet(entities_path)
    rels = pd.read_parquet(rels_path)
    return entities, rels


def normalize_entities(
    entities: pd.DataFrame, rels: pd.DataFrame
) -> pd.DataFrame:
    df = entities.copy()

    # Ensure required numeric columns exist.
    if "frequency" not in df.columns:
        df["frequency"] = 1
    df["frequency"] = df["frequency"].fillna(0).astype(int)

    if "degree" not in df.columns:
        # Fallback: compute degree from relationships by title.
        source_counts = rels["source"].value_counts()
        target_counts = rels["target"].value_counts()
        degree_map = source_counts.add(target_counts, fill_value=0)
        df["degree"] = df["title"].map(degree_map).fillna(0).astype(int)
    else:
        df["degree"] = df["degree"].fillna(0).astype(int)

    return df


def build_graph_data(
    entities: pd.DataFrame, rels: pd.DataFrame
) -> dict[str, list[dict[str, Any]]]:
    entities = normalize_entities(entities, rels)

    title_to_id = {row["title"]: row["id"] for _, row in entities.iterrows()}

    nodes = [
        {
            "id": row["id"],
            "label": str(row["title"])[:50],
            "title": row["title"],
            "frequency": int(row["frequency"]),
            "degree": int(row["degree"]),
        }
        for _, row in entities.iterrows()
    ]

    edges: list[dict[str, Any]] = []
    for _, row in rels.iterrows():
        source_title = row.get("source")
        target_title = row.get("target")
        source_id = title_to_id.get(source_title)
        target_id = title_to_id.get(target_title)
        if not source_id or not target_id:
            continue

        description = row.get("description") or ""
        try:
            weight_val = float(row.get("weight", 1.0))
        except (TypeError, ValueError):
            weight_val = 1.0

        edges.append(
            {
                "source": source_id,
                "target": target_id,
                "label": description[:100],
                "weight": weight_val,
            }
        )

    return {"nodes": nodes, "links": edges}


def script_json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False).replace("<", r"\u003c")


def render_html(graph_data: dict[str, Any]) -> str:
    node_count = len(graph_data["nodes"])
    edge_count = len(graph_data["links"])
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GraphRAG 知识图谱可视化</title>
    <script src="d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background: #f5f5f5;
        }}
        #graph {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .links line {{
            stroke: #999;
            stroke-opacity: 0.6;
        }}
        .nodes circle {{
            stroke: #fff;
            stroke-width: 1.5px;
            cursor: pointer;
        }}
        .nodes text {{
            font-size: 10px;
            pointer-events: none;
            text-anchor: middle;
        }}
        .edge-label {{
            font-size: 8px;
            fill: #666;
            pointer-events: none;
            text-anchor: middle;
        }}
        .tooltip {{
            position: absolute;
            padding: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 5px;
            pointer-events: none;
            font-size: 12px;
            display: none;
            z-index: 1000;
        }}
        .controls {{
            margin-bottom: 10px;
            padding: 10px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .controls button {{
            margin-right: 10px;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background: #007bff;
            color: white;
            cursor: pointer;
        }}
        .controls button:hover {{
            background: #0056b3;
        }}
        .stats-info {{
            font-size: 14px;
            color: #666;
        }}
        .stats-info strong {{
            color: #333;
            margin: 0 5px;
        }}
    </style>
</head>
<body>
    <div class="controls">
        <div>
            <button onclick="resetZoom()">重置视图</button>
        </div>
        <div class="stats-info">
            📊 实体 (节点): <strong>{node_count}</strong> |
            🔗 关系 (边): <strong>{edge_count}</strong>
        </div>
    </div>
    <div id="graph"></div>
    <div class="tooltip"></div>
    <script>
        const data = {script_json(graph_data)};
        const width = window.innerWidth - 40;
        const height = window.innerHeight - 100;

        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", [0, 0, width, height]);

        const g = svg.append("g");

        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});

        svg.call(zoom);

        window.resetZoom = function() {{
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity);
        }}

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links)
                .id(d => d.id)
                .distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(30));

        const link = g.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(data.links)
            .enter().append("line")
            .attr("stroke-width", d => Math.sqrt(d.weight) * 2);

        // 添加边的标签
        const edgeLabel = g.append("g")
            .attr("class", "edge-labels")
            .selectAll("text")
            .data(data.links)
            .enter().append("text")
            .attr("class", "edge-label")
            .text(d => d.label || "");

        const node = g.append("g")
            .attr("class", "nodes")
            .selectAll("g")
            .data(data.nodes)
            .enter().append("g")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.append("circle")
            .attr("r", d => Math.sqrt(d.degree || 0) * 3 + 5)
            .attr("fill", d => d3.interpolateViridis((d.degree || 0) / 20));

        node.append("text")
            .text(d => d.label)
            .attr("dy", ".35em")
            .attr("y", d => Math.sqrt(d.degree || 0) * 3 + 15);

        const tooltip = d3.select(".tooltip");

        node.on("mouseover", function(event, d) {{
            tooltip
                .style("display", "block")
                .text(`${{d.title}} — Degree: ${{d.degree}}, Frequency: ${{d.frequency}}`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");

            d3.select(this).select("circle")
                .attr("stroke", "orange")
                .attr("stroke-width", 3);
        }})
        .on("mouseout", function() {{
            tooltip.style("display", "none");
            d3.select(this).select("circle")
                .attr("stroke", "#fff")
                .attr("stroke-width", 1.5);
        }});

        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            // 更新边标签位置（在边的中点）
            edgeLabel
                .attr("x", d => (d.source.x + d.target.x) / 2)
                .attr("y", d => (d.source.y + d.target.y) / 2);

            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});

        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate knowledge graph visualization HTML.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.suffix.lower() != ".html":
        raise ValueError("output filename must end in .html")
    data_json = args.out.with_suffix(".json")
    if args.out.exists() or data_json.exists():
        raise FileExistsError("choose an unused output path")
    if not (args.out.parent / "d3.v7.min.js").is_file():
        raise FileNotFoundError("place the local D3 v7 script beside the output HTML")
    data = build_graph_data(*load_data(args.data_dir))
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(render_html(data))
    write_json(data_json, data)


if __name__ == "__main__":
    main()

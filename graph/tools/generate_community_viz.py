#!/usr/bin/env python
"""Generate an interactive community reports visualization (hierarchical HTML).

Default input: ../../output/communities.parquet and community_reports.parquet
Default output: ./output/community_visualization.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from graph.io import write_json
from .generate_graph_viz import script_json



def load_community_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    communities_path = data_dir / "communities.parquet"
    reports_path = data_dir / "community_reports.parquet"

    if not communities_path.exists() or not reports_path.exists():
        missing = []
        if not communities_path.exists():
            missing.append(str(communities_path))
        if not reports_path.exists():
            missing.append(str(reports_path))
        raise FileNotFoundError(
            f"Missing required file(s): {', '.join(missing)}. "
            "Run the indexing pipeline first or point --data-dir to a folder that contains them."
        )

    communities = pd.read_parquet(communities_path)
    reports = pd.read_parquet(reports_path)
    return communities, reports


def build_community_hierarchy(
    communities: pd.DataFrame, reports: pd.DataFrame
) -> list[dict[str, Any]]:
    """Build hierarchical community data structure."""
    # Merge communities with reports, using report's title
    merged = pd.merge(
        communities[['community', 'human_readable_id', 'level', 'parent', 'size']],
        reports[['community', 'title', 'summary', 'full_content', 'rank', 'rating_explanation', 'findings']],
        on='community',
        how='left'
    )

    # Build community nodes
    community_nodes = []
    for _, row in merged.iterrows():
        # Parse findings if available
        findings_list = []
        findings_value = row.get('findings')
        if findings_value is not None and not (isinstance(findings_value, float) and pd.isna(findings_value)):
            try:
                # Handle numpy array, list, or string
                if isinstance(findings_value, str):
                    findings_data = json.loads(findings_value)
                else:
                    findings_data = findings_value.tolist() if hasattr(findings_value, 'tolist') else findings_value

                if isinstance(findings_data, list):
                    findings_list = [
                        f.get('summary', '') if isinstance(f, dict) else str(f)
                        for f in findings_data
                    ]
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                findings_list = []

        node = {
            'id': str(row['community']),
            'human_readable_id': int(row['human_readable_id']) if pd.notna(row['human_readable_id']) else 0,
            'title': str(row.get('title', f"Community {row['community']}")),
            'level': int(row['level']) if pd.notna(row['level']) else 0,
            'size': int(row['size']) if pd.notna(row['size']) else 0,
            'rank': float(row['rank']) if pd.notna(row['rank']) else 0.0,
            'summary': str(row.get('summary', ''))[:500],  # Limit summary length
            'full_content': str(row.get('full_content', '')),
            'rating_explanation': str(row.get('rating_explanation', '')),
            'findings': findings_list,
            'parent': str(row['parent']) if pd.notna(row['parent']) else None,
        }
        community_nodes.append(node)

    return community_nodes


def render_community_html(communities_data: list[dict[str, Any]]) -> str:
    community_count = len(communities_data)

    # Calculate statistics
    total_size_sum = sum(c['size'] for c in communities_data)
    max_level = max((c['level'] for c in communities_data), default=0)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GraphRAG 社区报告可视化</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .stats {{
            display: flex;
            gap: 20px;
            margin-top: 10px;
        }}
        .stat-item {{
            padding: 10px 20px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .controls {{
            margin-bottom: 15px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .controls label {{
            margin-right: 10px;
            font-weight: 500;
        }}
        .controls select {{
            padding: 5px 10px;
            border-radius: 4px;
            border: 1px solid #ddd;
            margin-right: 20px;
        }}
        #visualization {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            padding: 20px;
            min-height: 600px;
        }}
        .community-card {{
            background: white;
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .community-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            border-color: #667eea;
        }}
        .community-card.expanded {{
            border-color: #667eea;
            background: #f0f4ff;
        }}
        .community-title {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .expand-icon {{
            font-size: 14px;
            color: #667eea;
            transition: transform 0.3s ease;
        }}
        .expand-icon.expanded {{
            transform: rotate(180deg);
        }}
        .community-meta {{
            display: flex;
            gap: 15px;
            margin-bottom: 10px;
            font-size: 12px;
            color: #666;
            flex-wrap: wrap;
        }}
        .meta-badge {{
            background: #e9ecef;
            padding: 5px 10px;
            border-radius: 3px;
            font-weight: 500;
            cursor: help;
        }}
        .meta-badge[title]:hover {{
            background: #667eea;
            color: white;
        }}
        .community-summary {{
            color: #555;
            font-size: 14px;
            line-height: 1.8;
            margin-bottom: 15px;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .rating-explanation {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px 12px;
            margin: 10px 0;
            font-size: 13px;
            border-radius: 3px;
        }}
        .rating-label {{
            font-weight: bold;
            color: #856404;
            margin-bottom: 5px;
        }}
        .full-content {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 2px solid #e9ecef;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }}
        .full-content.visible {{
            max-height: 2000px;
        }}
        .full-content-text {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            line-height: 1.8;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }}
        .findings {{
            margin-top: 15px;
            padding: 12px;
            background: #e7f3ff;
            border-radius: 5px;
            border-left: 4px solid #0066cc;
        }}
        .findings-label {{
            font-weight: bold;
            color: #0066cc;
            margin-bottom: 10px;
            display: block;
            font-size: 14px;
        }}
        .finding-item {{
            background: white;
            padding: 10px 12px;
            margin: 8px 0;
            border-left: 3px solid #667eea;
            border-radius: 3px;
            font-size: 13px;
            line-height: 1.6;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .level-0 {{ border-color: #667eea; }}
        .level-1 {{ border-color: #48bb78; }}
        .level-2 {{ border-color: #ed8936; }}
        .level-3 {{ border-color: #9f7aea; }}
        .grid-view {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 15px;
        }}
        .list-view {{
            display: flex;
            flex-direction: column;
        }}
        .list-view .community-card {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏘️ 社区报告分析</h1>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-label">社区总数</div>
                    <div class="stat-value">{community_count}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">社区规模总和</div>
                    <div class="stat-value">{total_size_sum}</div>
                    <div style="font-size: 10px; color: #999; margin-top: 3px;">*实体可能在多个社区中</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">最大层级</div>
                    <div class="stat-value">{max_level}</div>
                </div>
            </div>
        </div>

        <div class="controls">
            <label>排序方式:</label>
            <select id="sortBy" onchange="updateView()">
                <option value="rank">按重要性</option>
                <option value="size">按规模</option>
                <option value="level">按层级</option>
                <option value="id">按ID</option>
            </select>

            <label>筛选层级:</label>
            <select id="filterLevel" onchange="updateView()">
                <option value="all">全部层级</option>
                <option value="0">Level 0</option>
                <option value="1">Level 1</option>
                <option value="2">Level 2</option>
                <option value="3">Level 3</option>
            </select>

            <label>视图模式:</label>
            <select id="viewMode" onchange="updateView()">
                <option value="grid">网格视图</option>
                <option value="list">列表视图</option>
            </select>
        </div>

        <div id="visualization"></div>
    </div>

    <script>
        const communitiesData = {script_json(communities_data)};
        let expandedCommunities = new Set();
        function asText(value) {{
            const span = document.createElement('span');
            span.textContent = String(value ?? '');
            return span.innerHTML;
        }}

        function updateView() {{
            const sortBy = document.getElementById('sortBy').value;
            const filterLevel = document.getElementById('filterLevel').value;
            const viewMode = document.getElementById('viewMode').value;

            // Filter data
            let filtered = communitiesData;
            if (filterLevel !== 'all') {{
                filtered = communitiesData.filter(c => c.level === parseInt(filterLevel));
            }}

            // Sort data
            filtered.sort((a, b) => {{
                if (sortBy === 'rank') return b.rank - a.rank;
                if (sortBy === 'size') return b.size - a.size;
                if (sortBy === 'level') return a.level - b.level;
                if (sortBy === 'id') return a.human_readable_id - b.human_readable_id;
                return 0;
            }});

            renderCommunities(filtered, viewMode);
        }}

        function toggleCommunity(communityId, event) {{
            event.stopPropagation();
            if (expandedCommunities.has(communityId)) {{
                expandedCommunities.delete(communityId);
            }} else {{
                expandedCommunities.add(communityId);
            }}
            updateView();
        }}

        function renderCommunities(data, viewMode) {{
            const viz = document.getElementById('visualization');
            viz.className = viewMode === 'grid' ? 'grid-view' : 'list-view';
            viz.innerHTML = '';

            data.forEach(community => {{
                const card = document.createElement('div');
                const isExpanded = expandedCommunities.has(community.id);
                card.className = `community-card level-${{community.level}}${{isExpanded ? ' expanded' : ''}}`;

                card.innerHTML = `
                    <div class="community-title">
                        <span>🏷️ ${{asText(community.title)}}</span>
                        <span class="expand-icon${{isExpanded ? ' expanded' : ''}}">▼</span>
                    </div>
                    <div class="community-meta">
                        <span class="meta-badge">🆔 Community: ${{asText(community.id)}}</span>
                        <span class="meta-badge">📊 可读ID: ${{community.human_readable_id}}</span>
                        <span class="meta-badge">📈 层级 (Level): ${{community.level}}</span>
                        <span class="meta-badge">📦 规模 (Size): ${{community.size}} 个实体</span>
                        <span class="meta-badge" title="重要性评分：值越大表示该社区在整体知识图谱中越重要">⭐ 重要性评分 (Rank): ${{community.rank.toFixed(1)}} 分</span>
                    </div>
                    <div class="community-summary">
                        <strong style="color: #667eea;">📝 摘要 (Summary):</strong><br>
                        ${{asText(community.summary || '无摘要')}}
                    </div>
                    ${{community.rating_explanation ? `
                        <div class="rating-explanation">
                            <div class="rating-label">⭐ 重要性评分说明 (Rank Explanation):</div>
                            ${{asText(community.rating_explanation)}}
                        </div>
                    ` : ''}}
                    ${{community.findings && community.findings.length > 0 ? `
                        <div class="findings">
                            <span class="findings-label">🔍 关键发现 (Findings) - 共 ${{community.findings.length}} 条:</span>
                            ${{community.findings.map((f, idx) => `<div class="finding-item"><strong>[${{idx + 1}}]</strong> ${{asText(f)}}</div>`).join('')}}
                        </div>
                    ` : ''}}
                    <div class="full-content${{isExpanded ? ' visible' : ''}}">
                        <strong style="display: block; margin-bottom: 10px; color: #333; font-size: 15px;">📄 完整报告内容 (Full Content):</strong>
                        <div class="full-content-text">${{asText(community.full_content || '无完整内容')}}</div>
                    </div>
                `;

                card.addEventListener('click', (e) => toggleCommunity(community.id, e));

                viz.appendChild(card);
            }});
        }}

        // Initial render
        updateView();
    </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate community reports visualization HTML.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.suffix.lower() != ".html":
        raise ValueError("output filename must end in .html")
    data_json = args.out.with_suffix(".json")
    if args.out.exists() or data_json.exists():
        raise FileExistsError("choose an unused output path")
    data = build_community_hierarchy(*load_community_data(args.data_dir))
    html = render_community_html(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(html)
    write_json(data_json, data)


if __name__ == "__main__":
    main()

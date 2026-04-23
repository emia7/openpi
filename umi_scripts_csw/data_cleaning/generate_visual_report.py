#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成可视化HTML/Markdown报告
从JSON报告转换为人类可读的可视化报告

Usage:
    python generate_visual_report.py \
        --json_report ./anomaly_reports/trajectory_anomaly_report.json \
        --output_dir ./anomaly_reports \
        --format html
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def generate_html_report(report: Dict, output_path: Path):
    """生成HTML可视化报告"""
    
    summary = report["summary"]
    by_type = summary["by_type"]
    classified = report["classified_reports"]
    
    # 计算百分比
    total = summary["total_episodes"]
    critical_count = summary["statistics"]["critical"]
    warning_count = summary["statistics"]["warning"]
    minor_count = summary["statistics"]["minor"]
    ok_count = summary["statistics"]["ok"]
    
    critical_pct = (critical_count / total) * 100
    warning_pct = (warning_count / total) * 100
    minor_pct = (minor_count / total) * 100
    ok_pct = (ok_count / total) * 100
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>数据质量检测报告 - {report['data_source'].split('/')[-1]}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .header .subtitle {{
            opacity: 0.9;
            font-size: 14px;
        }}
        .content {{
            padding: 30px;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #667eea;
        }}
        .card.critical {{
            border-left-color: #dc3545;
            background: #fff5f5;
        }}
        .card.warning {{
            border-left-color: #ffc107;
            background: #fffbf0;
        }}
        .card.minor {{
            border-left-color: #17a2b8;
            background: #f0f9ff;
        }}
        .card.ok {{
            border-left-color: #28a745;
            background: #f0fff4;
        }}
        .card h3 {{
            font-size: 14px;
            color: #666;
            margin-bottom: 8px;
            text-transform: uppercase;
        }}
        .card .number {{
            font-size: 36px;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }}
        .card .percentage {{
            font-size: 14px;
            color: #666;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            font-size: 20px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }}
        .anomaly-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .anomaly-table th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #dee2e6;
        }}
        .anomaly-table td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        .anomaly-table tr:hover {{
            background: #f8f9fa;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-critical {{
            background: #dc3545;
            color: white;
        }}
        .badge-warning {{
            background: #ffc107;
            color: #333;
        }}
        .badge-minor {{
            background: #17a2b8;
            color: white;
        }}
        .badge-ok {{
            background: #28a745;
            color: white;
        }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
            display: flex;
        }}
        .progress-segment {{
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }}
        .progress-critical {{
            background: #dc3545;
            width: {critical_pct:.1f}%;
        }}
        .progress-warning {{
            background: #ffc107;
            color: #333;
            width: {warning_pct:.1f}%;
        }}
        .progress-minor {{
            background: #17a2b8;
            width: {minor_pct:.1f}%;
        }}
        .progress-ok {{
            background: #28a745;
            width: {ok_pct:.1f}%;
        }}
        .type-distribution {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .type-item {{
            display: flex;
            justify-content: space-between;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            align-items: center;
        }}
        .type-name {{
            font-weight: 500;
        }}
        .type-count {{
            font-size: 18px;
            font-weight: bold;
            color: #667eea;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 12px;
        }}
        .recommendation {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .recommendation h3 {{
            color: #856404;
            margin-bottom: 10px;
        }}
        .recommendation ul {{
            margin-left: 20px;
            color: #856404;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>数据质量检测报告</h1>
            <div class="subtitle">{report['data_source']} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <div class="content">
            <!-- 统计卡片 -->
            <div class="summary-cards">
                <div class="card">
                    <h3>总Episodes</h3>
                    <div class="number">{total}</div>
                    <div class="percentage">平均帧数: {summary['avg_frames']:.1f}</div>
                </div>
                <div class="card ok">
                    <h3>正常</h3>
                    <div class="number">{ok_count}</div>
                    <div class="percentage">{ok_pct:.1f}%</div>
                </div>
                <div class="card critical">
                    <h3>严重异常 (建议剔除)</h3>
                    <div class="number">{critical_count}</div>
                    <div class="percentage">{critical_pct:.1f}%</div>
                </div>
                <div class="card warning">
                    <h3>警告 (可选剔除)</h3>
                    <div class="number">{warning_count}</div>
                    <div class="percentage">{warning_pct:.1f}%</div>
                </div>
                <div class="card minor">
                    <h3>轻微异常 (可保留)</h3>
                    <div class="number">{minor_count}</div>
                    <div class="percentage">{minor_pct:.1f}%</div>
                </div>
            </div>
            
            <!-- 分布进度条 -->
            <div class="section">
                <h2>质量分布概览</h2>
                <div class="progress-bar">
                    <div class="progress-segment progress-critical" style="width: {critical_pct:.1f}%" title="严重: {critical_count}">{f'{critical_count}' if critical_pct > 5 else ''}</div>
                    <div class="progress-segment progress-warning" style="width: {warning_pct:.1f}%" title="警告: {warning_count}">{f'{warning_count}' if warning_pct > 5 else ''}</div>
                    <div class="progress-segment progress-minor" style="width: {minor_pct:.1f}%" title="轻微: {minor_count}">{f'{minor_count}' if minor_pct > 5 else ''}</div>
                    <div class="progress-segment progress-ok" style="width: {ok_pct:.1f}%" title="正常: {ok_count}">{f'{ok_count}' if ok_pct > 5 else ''}</div>
                </div>
                <div style="display: flex; justify-content: center; gap: 20px; margin-top: 10px; font-size: 12px;">
                    <span><span style="color: #dc3545;">■</span> 严重 {critical_pct:.1f}%</span>
                    <span><span style="color: #ffc107;">■</span> 警告 {warning_pct:.1f}%</span>
                    <span><span style="color: #17a2b8;">■</span> 轻微 {minor_pct:.1f}%</span>
                    <span><span style="color: #28a745;">■</span> 正常 {ok_pct:.1f}%</span>
                </div>
            </div>
            
            <!-- 异常类型分布 -->
            <div class="section">
                <h2>异常类型分布</h2>
                <div class="type-distribution">
"""
    
    # 异常类型映射
    type_names = {
        "sync_critical": "严重同步问题",
        "sync_warning": "中等同步问题",
        "sync_minor": "轻微同步问题",
        "static_trajectory": "静态轨迹",
        "too_few_frames": "帧数过少",
        "end_jump": "末端跳变",
        "clamp_abnormal": "夹爪异常",
        "ok": "正常"
    }
    
    for type_key, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
        type_name = type_names.get(type_key, type_key)
        html += f"""
                    <div class="type-item">
                        <span class="type-name">{type_name}</span>
                        <span class="type-count">{count}</span>
                    </div>
"""
    
    html += """
                </div>
            </div>
            
            <!-- 处理建议 -->
            <div class="recommendation">
                <h3>处理建议</h3>
                <ul>
"""
    
    if critical_count > 0:
        html += f"<li><strong>严重异常 ({critical_count}个)</strong>: 建议立即剔除，这些episodes存在严重同步问题或静态轨迹，无法用于训练</li>"
    if warning_count > 0:
        html += f"<li><strong>警告异常 ({warning_count}个)</strong>: 建议剔除，主要为末端跳变问题，可能影响动作学习质量</li>"
    if minor_count > 0:
        html += f"<li><strong>轻微异常 ({minor_count}个)</strong>: 可保留观察，轻微同步问题对训练影响较小</li>"
    
    html += f"""
                    <li><strong>正常数据 ({ok_count}个)</strong>: 质量良好，可直接用于训练</li>
                </ul>
            </div>
            
            <!-- 严重异常列表 -->
            <div class="section">
                <h2>严重异常详情 (建议剔除)</h2>
                <table class="anomaly-table">
                    <thead>
                        <tr>
                            <th>Episode</th>
                            <th>异常类型</th>
                            <th>严重程度</th>
                            <th>描述</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    for r in classified["critical"]:
        html += f"""
                        <tr>
                            <td>{r['episode']}</td>
                            <td>{type_names.get(r['anomaly_type'], r['anomaly_type'])}</td>
                            <td><span class="badge badge-critical">严重</span></td>
                            <td>{r['description']}</td>
                        </tr>
"""
    
    html += """
                    </tbody>
                </table>
            </div>
            
            <!-- 警告异常列表 -->
            <div class="section">
                <h2>警告异常详情 (可选剔除)</h2>
                <table class="anomaly-table">
                    <thead>
                        <tr>
                            <th>Episode</th>
                            <th>异常类型</th>
                            <th>严重程度</th>
                            <th>描述</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    # 显示所有警告异常
    for r in classified["warning"]:
        badge_class = "badge-warning"
        
        html += f"""
                        <tr>
                            <td>{r['episode']}</td>
                            <td>{type_names.get(r['anomaly_type'], r['anomaly_type'])}</td>
                            <td><span class="badge {badge_class}">警告</span></td>
                            <td>{r['description']}</td>
                        </tr>
"""
    
    html += """
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="footer">
            报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            检测工具: check_trajectory_anomalies.py | 
            规范版本: v1.1
        </div>
    </div>
</body>
</html>
"""
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"[OK] HTML报告已生成: {output_path}")


def generate_markdown_report(report: Dict, output_path: Path):
    """生成Markdown格式报告"""
    
    summary = report["summary"]
    by_type = summary["by_type"]
    classified = report["classified_reports"]
    
    total = summary["total_episodes"]
    critical_count = summary["statistics"]["critical"]
    warning_count = summary["statistics"]["warning"]
    minor_count = summary["statistics"]["minor"]
    ok_count = summary["statistics"]["ok"]
    
    md = f"""# 数据质量检测报告

> 数据源: `{report['data_source']}`  
> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
> 检测工具: check_trajectory_anomalies.py  
> 规范版本: v1.1

---

## 统计摘要

| 指标 | 数值 |
|------|------|
| 总Episodes | **{total}** |
| 平均帧数 | {summary['avg_frames']:.1f} |
| 正常数据 | {ok_count} ({(ok_count/total)*100:.1f}%) |
| 严重异常 (建议剔除) | **{critical_count} ({(critical_count/total)*100:.1f}%)** |
| 警告异常 (可选剔除) | {warning_count} ({(warning_count/total)*100:.1f}%) |
| 轻微异常 (可保留) | {minor_count} ({(minor_count/total)*100:.1f}%) |

---

## 质量分布

```
正常:    {'█' * int((ok_count/total)*50)}{'░' * (50-int((ok_count/total)*50))} {(ok_count/total)*100:.1f}%
警告:    {'█' * int((warning_count/total)*50)}{'░' * (50-int((warning_count/total)*50))} {(warning_count/total)*100:.1f}%
严重:    {'█' * int((critical_count/total)*50)}{'░' * (50-int((critical_count/total)*50))} {(critical_count/total)*100:.1f}%
```

---

## 异常类型分布

| 类型 | 数量 | 说明 |
|------|------|------|
"""
    
    type_names = {
        "sync_critical": "严重同步问题",
        "sync_warning": "中等同步问题",
        "sync_minor": "轻微同步问题",
        "static_trajectory": "静态轨迹",
        "too_few_frames": "帧数过少",
        "end_jump": "末端跳变",
        "clamp_abnormal": "夹爪异常",
        "ok": "正常"
    }
    
    for type_key, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
        type_name = type_names.get(type_key, type_key)
        md += f"| {type_name} | {count} | - |\n"
    
    md += f"""

---

## 处理建议

"""
    
    if critical_count > 0:
        md += f"- **严重异常 ({critical_count}个)**: 建议立即剔除，这些episodes存在严重同步问题或静态轨迹，无法用于训练\n"
    if warning_count > 0:
        md += f"- **警告异常 ({warning_count}个)**: 建议剔除，主要为末端跳变问题，可能影响动作学习质量\n"
    if minor_count > 0:
        md += f"- **轻微异常 ({minor_count}个)**: 可保留观察，轻微同步问题对训练影响较小\n"
    
    md += f"- **正常数据 ({ok_count}个)**: 质量良好，可直接用于训练\n\n"
    
    md += """---

## 严重异常列表 (建议剔除)

| Episode | 异常类型 | 严重程度 | 描述 |
|---------|---------|---------|------|
"""
    
    for r in classified["critical"]:
        type_name = type_names.get(r['anomaly_type'], r['anomaly_type'])
        md += f"| {r['episode']} | {type_name} | **严重** | {r['description']} |\n"
    
    md += """

---

## 警告异常列表 (全部)

| Episode | 异常类型 | 严重程度 | 描述 |
|---------|---------|---------|------|
"""
    
    # 显示所有警告异常
    for r in classified["warning"]:
        type_name = type_names.get(r['anomaly_type'], r['anomaly_type'])
        md += f"| {r['episode']} | {type_name} | 警告 | {r['description']} |\n"
    
    md += """

---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)
    
    print(f"[OK] Markdown报告已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="从JSON报告生成可视化HTML/Markdown报告")
    parser.add_argument("--json_report", required=True, help="输入的JSON报告路径")
    parser.add_argument("--output_dir", default="./anomaly_reports", help="输出目录")
    parser.add_argument("--format", choices=["html", "markdown", "both"], default="both",
                       help="输出格式: html, markdown, or both")
    args = parser.parse_args()
    
    # 读取JSON报告
    json_path = Path(args.json_report)
    if not json_path.exists():
        print(f"[ERROR] JSON报告不存在: {json_path}")
        return
    
    with open(json_path) as f:
        report = json.load(f)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成报告
    base_name = json_path.stem.replace("_report", "")
    
    if args.format in ["html", "both"]:
        html_path = output_dir / f"{base_name}_visual.html"
        generate_html_report(report, html_path)
    
    if args.format in ["markdown", "both"]:
        md_path = output_dir / f"{base_name}_visual.md"
        generate_markdown_report(report, md_path)
    
    print(f"\n[COMPLETE] 可视化报告生成完成!")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
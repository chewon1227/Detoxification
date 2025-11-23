"""
최종 리포트 시각화
final_report.json을 읽어서 HTML 도표 생성
"""

import json
import sys


def load_json(filepath):
    """JSON 파일 로드"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {filepath} 파일이 없습니다")
        return None


def generate_html(report_data):
    """HTML 리포트 생성"""
    
    base = report_data.get("base", {})
    detox = report_data.get("detox", {})
    ter = report_data.get("ter", {})
    stor = report_data.get("stor", {})
    verdicts = report_data.get("verdicts", {})
    
    base_explicit = base.get("explicit_metrics", {})
    detox_explicit = detox.get("explicit_metrics", {})
    base_implicit = base.get("implicit_bias", {})
    detox_implicit = detox.get("implicit_bias", {})
    
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Base vs Detox 최종 분석 리포트</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 50px;
            border-bottom: 2px solid #eee;
            padding-bottom: 30px;
        }}
        
        .section:last-child {{
            border-bottom: none;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
            padding-left: 15px;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
        }}
        
        .card-title {{
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            font-size: 1.1em;
        }}
        
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .metric:last-child {{
            border-bottom: none;
        }}
        
        .metric-label {{
            color: #666;
        }}
        
        .metric-value {{
            font-weight: bold;
            color: #333;
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 30px 0;
            background: white;
            border-radius: 8px;
            padding: 20px;
        }}
        
        .verdict {{
            background: #f0f7ff;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        
        .verdict.excellent {{
            background: #d4edda;
            border-left-color: #28a745;
        }}
        
        .verdict.good {{
            background: #d1ecf1;
            border-left-color: #17a2b8;
        }}
        
        .verdict.poor {{
            background: #f8d7da;
            border-left-color: #dc3545;
        }}
        
        .verdict-title {{
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 12px;
            text-align: right;
            border-bottom: 1px solid #dee2e6;
        }}
        
        th {{
            background: #f8f9fa;
            font-weight: bold;
            color: #333;
        }}
        
        th:first-child, td:first-child {{
            text-align: left;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Base vs Detox</h1>
            <p>최종 분석 리포트</p>
        </div>
        
        <div class="content">
            <!-- 명시적 지표 섹션 -->
            <div class="section">
                <div class="section-title">명시적 지표 (Explicit Metrics)</div>
                
                <table>
                    <tr>
                        <th>메트릭</th>
                        <th>Base</th>
                        <th>Detox</th>
                    </tr>
                    <tr>
                        <td>평균 유해도</td>
                        <td>{base_explicit.get('avg_score', 0):.4f}</td>
                        <td>{detox_explicit.get('avg_score', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td>최고 점수</td>
                        <td>{base_explicit.get('max_score', 0):.4f}</td>
                        <td>{detox_explicit.get('max_score', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td>최저 점수</td>
                        <td>{base_explicit.get('min_score', 0):.4f}</td>
                        <td>{detox_explicit.get('min_score', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td>심각한 혐오 개수</td>
                        <td>{base_explicit.get('severe_count', 0)}</td>
                        <td>{detox_explicit.get('severe_count', 0)}</td>
                    </tr>
                </table>
                
                <div class="chart-container">
                    <canvas id="explicit-chart"></canvas>
                </div>
            </div>
            
            <!-- 암시적 편향 섹션 -->
            <div class="section">
                <div class="section-title">암시적 편향 평가 (Implicit Bias)</div>
                
                <table>
                    <tr>
                        <th>메트릭</th>
                        <th>Base</th>
                        <th>Detox</th>
                    </tr>
                    <tr>
                        <td>전체 편향 점수</td>
                        <td>{base_implicit.get('overall_score', 0):.2f}</td>
                        <td>{detox_implicit.get('overall_score', 0):.2f}</td>
                    </tr>
                </table>
                
                <div class="chart-container">
                    <canvas id="implicit-chart"></canvas>
                </div>
            </div>
            
            <!-- TER 섹션 -->
            <div class="section">
                <div class="section-title">TER (턴별 혐오 증가율)</div>
                
                <table>
                    <tr>
                        <th>항목</th>
                        <th>점수</th>
                    </tr>
                    <tr>
                        <td>Base TER</td>
                        <td>{ter.get('base', 0):+.2f}%</td>
                    </tr>
                    <tr>
                        <td>Detox TER</td>
                        <td>{ter.get('detox', 0):+.2f}%</td>
                    </tr>
                    <tr>
                        <td>감소율</td>
                        <td>{ter.get('reduction', 0):+.2f}%</td>
                    </tr>
                </table>
            </div>
            
            <!-- STOR 섹션 -->
            <div class="section">
                <div class="section-title">STOR (심각한 혐오 발생 비율)</div>
                
                <table>
                    <tr>
                        <th>항목</th>
                        <th>점수</th>
                    </tr>
                    <tr>
                        <td>Base STOR</td>
                        <td>{stor.get('base', 0):.2f}%</td>
                    </tr>
                    <tr>
                        <td>Detox STOR</td>
                        <td>{stor.get('detox', 0):.2f}%</td>
                    </tr>
                    <tr>
                        <td>감소율</td>
                        <td>{stor.get('reduction_rate', 0):.2f}%</td>
                    </tr>
                </table>
                
                <div class="chart-container">
                    <canvas id="stor-chart"></canvas>
                </div>
            </div>
            
            <!-- 최종 평가 섹션 -->
            <div class="section">
                <div class="section-title">최종 평가</div>
                
                <div class="verdict excellent">
                    <div class="verdict-title">1. 점수 개선도</div>
                    <div>{verdicts.get('score_improvement', {}).get('verdict', 'N/A')}</div>
                    <div>개선율: {verdicts.get('score_improvement', {}).get('rate', 0):.2f}%</div>
                </div>
                
                <div class="verdict {'good' if verdicts.get('risk_reduction', {}).get('rate', 0) >= 50 else 'poor'}">
                    <div class="verdict-title">2. 위험 관리</div>
                    <div>{verdicts.get('risk_reduction', {}).get('verdict', 'N/A')}</div>
                    <div>감소율: {verdicts.get('risk_reduction', {}).get('rate', 0):.2f}%</div>
                </div>
                
                <div class="verdict excellent">
                    <div class="verdict-title">3. 배포 권장</div>
                    <div>{verdicts.get('deployment_readiness', {}).get('status', 'N/A')}</div>
                    <div>Detox STOR: {verdicts.get('deployment_readiness', {}).get('detox_stor', 0):.2f}%</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Base vs Detox 최종 분석 리포트 | 생성일: 2024-11-23</p>
        </div>
    </div>
    
    <script>
        // 명시적 지표 차트
        const explicitCtx = document.getElementById('explicit-chart').getContext('2d');
        new Chart(explicitCtx, {{
            type: 'bar',
            data: {{
                labels: ['평균 점수', '최고 점수', '최저 점수', '심각한 혐오'],
                datasets: [
                    {{
                        label: 'Base',
                        data: [{base_explicit.get('avg_score', 0)}, {base_explicit.get('max_score', 0)}, {base_explicit.get('min_score', 0)}, {base_explicit.get('severe_count', 0) / 10}],
                        backgroundColor: '#ff6b6b'
                    }},
                    {{
                        label: 'Detox',
                        data: [{detox_explicit.get('avg_score', 0)}, {detox_explicit.get('max_score', 0)}, {detox_explicit.get('min_score', 0)}, {detox_explicit.get('severe_count', 0) / 10}],
                        backgroundColor: '#51cf66'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1
                    }}
                }}
            }}
        }});
        
        // 암시적 편향 차트
        const implicitCtx = document.getElementById('implicit-chart').getContext('2d');
        new Chart(implicitCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['Base', 'Detox'],
                datasets: [{{
                    data: [{base_implicit.get('overall_score', 0)}, {detox_implicit.get('overall_score', 0)}],
                    backgroundColor: ['#ff6b6b', '#51cf66']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false
            }}
        }});
        
        // STOR 차트
        const storCtx = document.getElementById('stor-chart').getContext('2d');
        new Chart(storCtx, {{
            type: 'bar',
            data: {{
                labels: ['STOR'],
                datasets: [
                    {{
                        label: 'Base (%)',
                        data: [{stor.get('base', 0)}],
                        backgroundColor: '#ff6b6b'
                    }},
                    {{
                        label: 'Detox (%)',
                        data: [{stor.get('detox', 0)}],
                        backgroundColor: '#51cf66'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    return html


def main():
    """메인 함수"""
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "final_report.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "visualization_report.html"
    
    print(f"\nStep 1: {input_file} 로드\n")
    
    report_data = load_json(input_file)
    if not report_data:
        return
    
    print(f"SUCCESS: {input_file} 로드 완료\n")
    
    print(f"Step 2: HTML 리포트 생성\n")
    
    html_content = generate_html(report_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"SUCCESS: {output_file} 생성 완료\n")
    print(f"브라우저에서 {output_file}을 열어주세요\n")


if __name__ == "__main__":
    main()

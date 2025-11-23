"""
최종 리포트 시각화
final_report.json을 읽어서 PNG/JPEG 이미지 생성
"""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False


def load_json(filepath):
    """JSON 파일 로드"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {filepath} 파일이 없습니다")
        return None


def create_comparison_chart(report_data):
    """비교 차트 생성"""
    
    base = report_data.get("base", {})
    detox = report_data.get("detox", {})
    
    base_explicit = base.get("explicit_metrics", {})
    detox_explicit = detox.get("explicit_metrics", {})
    
    # 그림 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Base vs Detox - Explicit Metrics Comparison', fontsize=16, fontweight='bold')
    
    # 1. 평균 점수
    ax = axes[0, 0]
    models = ['Base', 'Detox']
    avg_scores = [base_explicit.get('avg_score', 0), detox_explicit.get('avg_score', 0)]
    colors = ['#ff6b6b', '#51cf66']
    ax.bar(models, avg_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Average Score', fontweight='bold')
    ax.set_title('Average Toxicity Score')
    ax.set_ylim([0, 1])
    for i, v in enumerate(avg_scores):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # 2. 심각한 혐오 개수
    ax = axes[0, 1]
    severe_counts = [base_explicit.get('severe_count', 0), detox_explicit.get('severe_count', 0)]
    ax.bar(models, severe_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Severe Hate Speech Count (Score >= 0.7)')
    for i, v in enumerate(severe_counts):
        ax.text(i, v + 0.2, f'{int(v)}', ha='center', fontweight='bold')
    
    # 3. 최고 점수
    ax = axes[1, 0]
    max_scores = [base_explicit.get('max_score', 0), detox_explicit.get('max_score', 0)]
    ax.bar(models, max_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Maximum Score')
    ax.set_ylim([0, 1])
    for i, v in enumerate(max_scores):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # 4. 최저 점수
    ax = axes[1, 1]
    min_scores = [base_explicit.get('min_score', 0), detox_explicit.get('min_score', 0)]
    ax.bar(models, min_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Minimum Score')
    ax.set_ylim([0, 1])
    for i, v in enumerate(min_scores):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_implicit_bias_chart(report_data):
    """암시적 편향 차트 생성"""
    
    base = report_data.get("base", {})
    detox = report_data.get("detox", {})
    
    base_implicit = base.get("implicit_bias", {})
    detox_implicit = detox.get("implicit_bias", {})
    
    # 그림 생성
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Implicit Bias Comparison', fontsize=16, fontweight='bold')
    
    # 1. 전체 편향 점수
    ax = axes[0]
    models = ['Base', 'Detox']
    overall_scores = [base_implicit.get('overall_score', 0), detox_implicit.get('overall_score', 0)]
    colors = ['#ff6b6b', '#51cf66']
    ax.bar(models, overall_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Overall Bias Score (0~10)')
    ax.set_ylim([0, 10])
    for i, v in enumerate(overall_scores):
        ax.text(i, v + 0.2, f'{v:.2f}', ha='center', fontweight='bold')
    
    # 2. 5가지 차원 비교 (레이더 차트)
    ax = axes[1]
    
    dimensions = ['Sarcasm', 'Bias Reinforcement', 'Condescension', 'Stereotyping', 'Hostility']
    base_dims = base_implicit.get('dimensions', {})
    detox_dims = detox_implicit.get('dimensions', {})
    
    base_values = [
        base_dims.get('sarcasm_mockery', 0),
        base_dims.get('bias_reinforcement', 0),
        base_dims.get('condescension', 0),
        base_dims.get('stereotyping', 0),
        base_dims.get('emotional_hostility', 0)
    ]
    
    detox_values = [
        detox_dims.get('sarcasm_mockery', 0),
        detox_dims.get('bias_reinforcement', 0),
        detox_dims.get('condescension', 0),
        detox_dims.get('stereotyping', 0),
        detox_dims.get('emotional_hostility', 0)
    ]
    
    x = np.arange(len(dimensions))
    width = 0.35
    
    ax.bar(x - width/2, base_values, width, label='Base', color='#ff6b6b', alpha=0.7, edgecolor='black')
    ax.bar(x + width/2, detox_values, width, label='Detox', color='#51cf66', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Bias Dimensions Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 10])
    
    plt.tight_layout()
    return fig


def create_stor_chart(report_data):
    """STOR 차트 생성"""
    
    stor = report_data.get("stor", {})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('STOR (Severe Toxicity Occurrence Rate) Comparison', fontsize=16, fontweight='bold')
    
    # 1. STOR 비교
    ax = axes[0]
    models = ['Base', 'Detox']
    stor_values = [stor.get('base', 0), stor.get('detox', 0)]
    colors = ['#ff6b6b', '#51cf66']
    bars = ax.bar(models, stor_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title('STOR Percentage')
    ax.set_ylim([0, 100])
    for i, (bar, v) in enumerate(zip(bars, stor_values)):
        ax.text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')
    
    # 2. STOR 개선율
    ax = axes[1]
    improvement = stor.get('reduction_rate', 0)
    labels = ['Reduced', 'Remaining']
    sizes = [improvement, 100 - improvement]
    colors_pie = ['#51cf66', '#ff6b6b']
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                       colors=colors_pie, startangle=90)
    ax.set_title(f'STOR Reduction Rate: {improvement:.2f}%')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    return fig


def create_ter_chart(report_data):
    """TER 차트 생성"""
    
    ter = report_data.get("ter", {})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('TER (Turn Escalation Rate) Comparison', fontsize=16, fontweight='bold')
    
    categories = ['Base', 'Detox', 'Reduction']
    values = [ter.get('base', 0), ter.get('detox', 0), ter.get('reduction', 0)]
    colors = ['#ff6b6b', '#51cf66', '#4dabf7']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title('Turn Escalation Rate')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    for bar, v in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                f'{v:+.2f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_summary_chart(report_data):
    """최종 평가 요약 차트"""
    
    verdicts = report_data.get("verdicts", {})
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    fig.suptitle('Final Evaluation Summary', fontsize=16, fontweight='bold')
    
    # 1. 점수 개선도
    ax1 = fig.add_subplot(gs[0, :])
    improvement_rate = verdicts.get('score_improvement', {}).get('rate', 0)
    verdict_text = verdicts.get('score_improvement', {}).get('verdict', 'N/A')
    
    ax1.barh(['Score Improvement'], [improvement_rate], color='#667eea', alpha=0.7, edgecolor='black')
    ax1.set_xlim([0, 100])
    ax1.set_xlabel('Percentage (%)', fontweight='bold')
    ax1.text(improvement_rate + 2, 0, f'{improvement_rate:.2f}%\n{verdict_text}', 
             va='center', fontweight='bold')
    
    # 2. 위험 감소
    ax2 = fig.add_subplot(gs[1, :])
    risk_rate = verdicts.get('risk_reduction', {}).get('rate', 0)
    risk_text = verdicts.get('risk_reduction', {}).get('verdict', 'N/A')
    
    ax2.barh(['Risk Reduction'], [risk_rate], color='#764ba2', alpha=0.7, edgecolor='black')
    ax2.set_xlim([0, 100])
    ax2.set_xlabel('Percentage (%)', fontweight='bold')
    ax2.text(risk_rate + 2, 0, f'{risk_rate:.2f}%\n{risk_text}', 
             va='center', fontweight='bold')
    
    # 3. 배포 권장
    ax3 = fig.add_subplot(gs[2, :])
    deployment_status = verdicts.get('deployment_readiness', {}).get('status', 'N/A')
    detox_stor = verdicts.get('deployment_readiness', {}).get('detox_stor', 0)
    
    status_color = '#51cf66' if '[READY]' in deployment_status else ('#17a2b8' if '[REVIEW]' in deployment_status else '#ff6b6b')
    
    ax3.text(0.5, 0.7, deployment_status, ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3))
    ax3.text(0.5, 0.3, f'Detox STOR: {detox_stor:.2f}%', ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.axis('off')
    
    plt.tight_layout()
    return fig


def main():
    """메인 함수"""
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "final_report.json"
    output_format = sys.argv[2] if len(sys.argv) > 2 else "png"  # png 또는 jpeg
    
    print(f"\nStep 1: {input_file} 로드\n")
    
    report_data = load_json(input_file)
    if not report_data:
        return
    
    print(f"SUCCESS: {input_file} 로드 완료\n")
    print(f"Step 2: 이미지 생성 중 ({output_format.upper()})\n")
    
    # 차트 생성
    charts = {
        'comparison': create_comparison_chart(report_data),
        'implicit_bias': create_implicit_bias_chart(report_data),
        'stor': create_stor_chart(report_data),
        'ter': create_ter_chart(report_data),
        'summary': create_summary_chart(report_data)
    }
    
    # 이미지 저장
    for name, fig in charts.items():
        output_file = f"report_{name}.{output_format}"
        fig.savefig(output_file, dpi=300, bbox_inches='tight', format=output_format)
        print(f"   ✓ {output_file} 저장 완료")
        plt.close(fig)
    
    print(f"\n✅ 모든 이미지가 생성되었습니다!\n")
    print("생성된 파일:")
    print("   • report_comparison.png (명시적 지표 비교)")
    print("   • report_implicit_bias.png (암시적 편향 분석)")
    print("   • report_stor.png (STOR 비교)")
    print("   • report_ter.png (TER 비교)")
    print("   • report_summary.png (최종 평가 요약)")
    print()


if __name__ == "__main__":
    main()
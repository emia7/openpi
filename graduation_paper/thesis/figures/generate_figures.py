#!/usr/bin/env python3
"""
生成论文可视化图表
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置全局样式
plt.style.use('seaborn-v0_8-whitegrid')

# 1. 数据量-性能曲线图
def plot_data_performance_curve():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_amounts = [100, 200, 400]
    success_rates = [55, 70, 90]
    
    ax.plot(data_amounts, success_rates, 'o-', linewidth=2.5, markersize=10, 
            color='#2E86AB', label='Task Success Rate')
    
    # 添加数据标签
    for x, y in zip(data_amounts, success_rates):
        ax.annotate(f'{y}%', (x, y), textcoords="offset points", xytext=(0,10), 
                   ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Training Data Amount (Episodes)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Data Scale on Task Performance\n(Pick-Place Task, Pure Image Input)', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xlim(50, 500)
    ax.set_ylim(40, 100)
    ax.grid(True, alpha=0.3)
    
    # 添加趋势线说明
    ax.text(0.5, 0.02, 'Success rate increases from 55% to 90% as data scales from 100 to 400 episodes',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('data_performance_curve.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('data_performance_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: data_performance_curve.pdf/png")

# 2. 跨平台成功率对比柱状图
def plot_cross_platform_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    platforms = ['FastUMI\n(Ours)', 'XV Single Arm\n(Cross-Hardware)', 'Franka Teleop\n(Baseline)']
    success_rates = [90, 87.5, 100]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = ax.bar(platforms, success_rates, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # 添加数据标签
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.annotate(f'{rate}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=13, fontweight='bold')
    
    ax.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Cross-Platform Performance Comparison\n(Pick-Place Task)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加说明文字
    ax.text(0.5, 0.02, 'Our embodiment-free approach achieves 90% success rate, close to the 100% teleoperation baseline',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('cross_platform_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('cross_platform_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: cross_platform_comparison.pdf/png")

# 3. handover失败模式饼图
def plot_handover_failure_modes():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左侧：失败模式分布
    failure_modes = ['Grasping\nFailure\n(40%)', 'Handover\nAlignment\n(45%)', 'Receiving\nFailure\n(15%)']
    percentages = [40, 45, 15]
    colors = ['#E74C3C', '#F39C12', '#3498DB']
    explode = (0.05, 0.05, 0.05)
    
    wedges, texts, autotexts = ax1.pie(percentages, labels=failure_modes, colors=colors,
                                        autopct='%1.0f%%', startangle=90, explode=explode,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    ax1.set_title('Handover Task Failure Mode Distribution\n(Current Success Rate: 35%)', 
                 fontsize=13, fontweight='bold', pad=20)
    
    # 右侧：改进方向建议
    ax2.axis('off')
    improvements = [
        "Improvement Directions:",
        "",
        "1. Enhance Grasping Data (40% failures)",
        "   → Add more phase-1 demonstrations",
        "",
        "2. Optimize Alignment (45% failures)",
        "   → Focus on handover spatial coordination",
        "",
        "3. Refine Receiving (15% failures)",
        "   → Improve grip timing and force control",
        "",
        "4. Data Quality Optimization",
        "   → Apply trajectory analysis for filtering"
    ]
    
    y_pos = 0.95
    for line in improvements:
        if line.startswith("Improvement"):
            ax2.text(0.1, y_pos, line, fontsize=12, fontweight='bold', 
                    transform=ax2.transAxes, va='top')
        elif line.startswith(("1.", "2.", "3.", "4.")):
            ax2.text(0.1, y_pos, line, fontsize=11, fontweight='bold',
                    color='#2E86AB', transform=ax2.transAxes, va='top')
        elif line.startswith("   →"):
            ax2.text(0.15, y_pos, line, fontsize=10, style='italic',
                    transform=ax2.transAxes, va='top')
        else:
            ax2.text(0.1, y_pos, line, fontsize=10,
                    transform=ax2.transAxes, va='top')
        y_pos -= 0.08
    
    plt.tight_layout()
    plt.savefig('handover_failure_modes.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('handover_failure_modes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: handover_failure_modes.pdf/png")

# 4. 相机选型对比雷达图
def plot_camera_radar():
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    categories = ['FOV\nCoverage', 'Stability', 'Comfort', 'Sync\nConvenience']
    N = len(categories)
    
    # 评分数据
    d435_scores = [4, 5, 0, 5]  # D435固定（舒适度不计）
    nano_scores = [5, 4, 5, 2]  # DJI Nano头戴
    
    # 角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # 数据闭合
    d435_scores += d435_scores[:1]
    nano_scores += nano_scores[:1]
    
    # 绘制
    ax.plot(angles, d435_scores, 'o-', linewidth=2, label='D435 Fixed', color='#2E86AB')
    ax.fill(angles, d435_scores, alpha=0.25, color='#2E86AB')
    
    ax.plot(angles, nano_scores, 's-', linewidth=2, label='DJI Nano Head-mounted', color='#A23B72')
    ax.fill(angles, nano_scores, alpha=0.25, color='#A23B72')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 5)
    ax.set_title('Camera Selection Comparison\n(Evaluation Dimensions)', 
                fontsize=13, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('camera_comparison_radar.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('camera_comparison_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: camera_comparison_radar.pdf/png")

# 5. 系统架构流程图（简化版）
def plot_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 定义颜色
    color_collect = '#3498DB'
    color_process = '#2ECC71'
    color_train = '#E74C3C'
    color_deploy = '#F39C12'
    
    # 绘制四个阶段框
    stages = [
        (0.5, 4, 2, 3, 'Data\nCollection', color_collect, 
         '• FastUMI/XV devices\n• D435 third-view camera\n• ROS bag recording'),
        (3, 4, 2, 3, 'Data\nProcessing', color_process,
         '• rosbag → mp4/json\n• Time synchronization\n• LeRobot format'),
        (5.5, 4, 2, 3, 'Model\nTraining', color_train,
         '• π0.5 fine-tuning\n• Image input\n• Relative action output'),
        (8, 4, 1.5, 3, 'Policy\nDeployment', color_deploy,
         '• Real-time inference\n• Robot control')
    ]
    
    for x, y, w, h, title, color, content in stages:
        # 绘制矩形
        rect = plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.3, 
                             edgecolor=color, linewidth=3)
        ax.add_patch(rect)
        
        # 标题
        ax.text(x + w/2, y + h - 0.3, title, ha='center', va='top',
               fontsize=12, fontweight='bold', color=color)
        
        # 内容
        ax.text(x + w/2, y + h/2, content, ha='center', va='center',
               fontsize=9, linespacing=1.5)
    
    # 绘制箭头
    arrow_style = dict(arrowstyle='->', lw=2.5, color='#34495E')
    ax.annotate('', xy=(3, 5.5), xytext=(2.5, 5.5), arrowprops=arrow_style)
    ax.annotate('', xy=(5.5, 5.5), xytext=(5, 5.5), arrowprops=arrow_style)
    ax.annotate('', xy=(8, 5.5), xytext=(7.5, 5.5), arrowprops=arrow_style)
    
    # 数据格式标签
    formats = [
        (2.75, 3.2, 'rosbag'),
        (5.25, 3.2, 'mp4+json'),
        (7.75, 3.2, 'LeRobot'),
    ]
    for x, y, text in formats:
        ax.text(x, y, text, ha='center', va='center', fontsize=9, 
               style='italic', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 标题
    ax.text(5, 7.5, 'Embodiment-Free Data Collection System Architecture', 
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('system_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: system_architecture.pdf/png")

if __name__ == '__main__':
    print("Generating visualization figures for thesis...\n")
    
    plot_data_performance_curve()
    plot_cross_platform_comparison()
    plot_handover_failure_modes()
    plot_camera_radar()
    plot_system_architecture()
    
    print("\n✅ All figures generated successfully!")
    print("Files saved in: figures/")

#!/usr/bin/env python3
"""
键盘触发测试脚本 - 验证 ⬆️+Enter 和 Ctrl+C 触发成功率

使用:
    python test_keyboard_trigger.py
    
测试流程:
    1. 启动监听
    2. 提示用户进行 10 次 ⬆️+Enter 测试
    3. 提示用户进行 10 次 Ctrl+C 测试
    4. 提示用户进行 5 次正常打字测试（误触发检查）
    5. 生成测试报告
"""

import time
import sys
import os


def print_header():
    """打印标题"""
    print("=" * 60)
    print("键盘触发测试")
    print("=" * 60)
    print()


def test_sequence(test_name, num_tests, instruction):
    """
    执行一系列测试并收集结果
    
    Args:
        test_name: 测试名称
        num_tests: 测试次数
        instruction: 操作说明
    
    Returns:
        list: 每次测试的结果 (True/False)
    """
    results = []
    
    print(f"【{test_name}】")
    print(f"需要进行 {num_tests} 次测试")
    print(f"操作: {instruction}")
    print()
    
    for i in range(1, num_tests + 1):
        print(f"测试 {i}/{num_tests}:")
        print(f"  准备...")
        time.sleep(0.5)
        
        print(f"  请执行: {instruction}")
        print(f"  听到蜂鸣声后输入 y 表示成功，n 表示失败，s 表示跳过: ", end='', flush=True)
        
        # 等待用户输入
        try:
            response = input().strip().lower()
            if response == 'y':
                results.append(True)
                print(f"  ✓ 成功")
            elif response == 'n':
                results.append(False)
                print(f"  ✗ 失败")
            elif response == 's':
                print(f"  - 跳过")
                continue
            else:
                print(f"  ? 无效输入，视为失败")
                results.append(False)
        except EOFError:
            print()
            print("[ERROR] 输入中断")
            sys.exit(1)
        
        print()
    
    return results


def calculate_stats(results):
    """计算统计信息"""
    if not results:
        return 0, 0, 0.0
    
    success = sum(results)
    total = len(results)
    rate = (success / total) * 100 if total > 0 else 0.0
    
    return success, total, rate


def main():
    """主函数"""
    print_header()
    
    # 检查依赖
    script_dir = os.path.dirname(os.path.abspath(__file__))
    listener_script = os.path.join(script_dir, 'keyboard_beep_listener.py')
    
    if not os.path.exists(listener_script):
        print(f"[ERROR] 找不到键盘监听脚本: {listener_script}")
        sys.exit(1)
    
    print("前提条件检查:")
    print(f"  ✓ 找到 keyboard_beep_listener.py")
    print()
    
    print("【重要】测试前请先启动键盘监听:")
    print(f"  cd {script_dir}")
    print("  python keyboard_beep_listener.py")
    print()
    print("监听启动后，按 Enter 开始测试...")
    input()
    print()
    
    # 执行三组测试
    all_results = {}
    
    # 测试1: ⬆️+Enter
    results1 = test_sequence(
        "⬆️+Enter 触发测试",
        10,
        "⬆️ 然后 0.5s 内按 Enter"
    )
    all_results['up_enter'] = results1
    print()
    
    # 测试2: Ctrl+C
    results2 = test_sequence(
        "Ctrl+C 触发测试",
        10,
        "按 Ctrl+C"
    )
    all_results['ctrl_c'] = results2
    print()
    
    # 测试3: 误触发检查（正常打字）
    results3 = test_sequence(
        "误触发检查（正常打字）",
        5,
        "正常输入: hello world 12345"
    )
    all_results['false_positive'] = results3
    print()
    
    # 生成报告
    print("=" * 60)
    print("测试报告")
    print("=" * 60)
    print()
    
    # ⬆️+Enter 统计
    s1, t1, r1 = calculate_stats(all_results['up_enter'])
    print(f"⬆️+Enter 触发:")
    print(f"  成功率: {r1:.1f}% ({s1}/{t1})")
    print(f"  状态: {'✓ 通过' if r1 >= 100 else '✗ 不通过'}")
    print()
    
    # Ctrl+C 统计
    s2, t2, r2 = calculate_stats(all_results['ctrl_c'])
    print(f"Ctrl+C 触发:")
    print(f"  成功率: {r2:.1f}% ({s2}/{t2})")
    print(f"  状态: {'✓ 通过' if r2 >= 100 else '✗ 不通过'}")
    print()
    
    # 误触发统计
    s3, t3, r3 = calculate_stats(all_results['false_positive'])
    # 误触发率应该是 0%，所以成功率是 (1 - 误触发率)
    false_positive_rate = ((t3 - s3) / t3) * 100 if t3 > 0 else 0.0
    print(f"误触发检查:")
    print(f"  误触发次数: {t3 - s3}/{t3}")
    print(f"  误触发率: {false_positive_rate:.1f}%")
    print(f"  状态: {'✓ 通过' if false_positive_rate == 0 else '⚠ 警告'}")
    print()
    
    # 总体评估
    print("=" * 60)
    print("总体评估:")
    
    passed = (r1 >= 100 and r2 >= 100 and false_positive_rate == 0)
    
    if passed:
        print("  ✓ 通过 - 键盘触发系统可以投入使用")
    else:
        print("  ✗ 不通过 - 请检查问题并重新测试")
        if r1 < 100:
            print("    - ⬆️+Enter 触发成功率不足")
        if r2 < 100:
            print("    - Ctrl+C 触发成功率不足")
        if false_positive_rate > 0:
            print("    - 存在误触发，需要优化监听逻辑")
    
    print("=" * 60)
    
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())

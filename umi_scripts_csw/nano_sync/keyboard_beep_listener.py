#!/usr/bin/env python3
"""
键盘监听脚本 - 在终端激活时监听特定按键组合并播放蜂鸣声

监听范围: 全局监听（仅在 macOS/Linux 终端使用时有效）
触发条件:
  - ⬆️ 键后 0.5s 内按下 Enter → 播放开始音（单蜂鸣）
  - Ctrl+C → 播放结束音（双蜂鸣）

使用:
    python keyboard_beep_listener.py
    
输出:
    - 终端打印 "[START BEEP] Played at HH:MM:SS"
    - 终端打印 "[END BEEP] Played at HH:MM:SS"
    
退出:
    - 按 Ctrl+Shift+Q 退出监听
"""

import time
import threading
import subprocess
import sys
import os

# 配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
START_BEEP_FILE = os.path.join(SCRIPT_DIR, 'start_beep.wav')
END_BEEP_FILE = os.path.join(SCRIPT_DIR, 'end_beep.wav')
UP_ARROW_TIMEOUT = 0.5  # ⬆️ 后多少秒内按 Enter 有效

# 状态
up_arrow_pressed = False
up_arrow_time = 0
ctrl_pressed = False


def check_audio_files():
    """检查音频文件是否存在"""
    if not os.path.exists(START_BEEP_FILE):
        print(f"[ERROR] 开始音文件不存在: {START_BEEP_FILE}")
        print("请先运行: python generate_beep.py")
        sys.exit(1)
    
    if not os.path.exists(END_BEEP_FILE):
        print(f"[ERROR] 结束音文件不存在: {END_BEEP_FILE}")
        print("请先运行: python generate_beep.py")
        sys.exit(1)
    
    print("[OK] 音频文件检查通过")


def play_beep(beep_file, label):
    """后台播放蜂鸣声并打印确认"""
    timestamp = time.strftime('%H:%M:%S')
    print(f"\n[{label} BEEP] Played at {timestamp}", flush=True)
    
    def _play():
        # 尝试多种播放方式
        players = [
            # macOS 系统自带
            ['afplay', beep_file],
            # Linux PulseAudio
            ['paplay', beep_file],
            # Linux ALSA
            ['aplay', beep_file],
            # ffmpeg ffplay (如果安装了)
            ['ffplay', '-nodisp', '-autoexit', beep_file],
        ]
        
        for cmd in players:
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=3)
                if result.returncode == 0:
                    return  # 播放成功
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue  # 尝试下一个播放器
        
        # 都失败了
        print(f"[ERROR] 无法播放音频文件: {beep_file}")
        print(f"[ERROR] 请安装音频播放器:")
        print(f"        macOS: 系统自带 afplay 应该可用")
        print(f"        Linux: sudo apt-get install pulseaudio-utils 或 alsa-utils")
    
    # 后台线程播放，不阻塞主循环
    threading.Thread(target=_play, daemon=True).start()


def reset_up_arrow_state():
    """重置 ⬆️ 键状态"""
    global up_arrow_pressed
    time.sleep(UP_ARROW_TIMEOUT)
    up_arrow_pressed = False


# 尝试导入 pynput
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("[ERROR] pynput 未安装")
    print("安装命令: pip install pynput")
    sys.exit(1)


def on_press(key):
    """按键按下回调"""
    global up_arrow_pressed, up_arrow_time, ctrl_pressed
    
    try:
        # 检测 ⬆️ 键
        if key == keyboard.Key.up:
            up_arrow_pressed = True
            up_arrow_time = time.time()
            # 0.5s 后重置状态
            threading.Thread(target=reset_up_arrow_state, daemon=True).start()
        
        # 检测 Enter 键（在 ⬆️ 后 0.5s 内）
        elif key == keyboard.Key.enter and up_arrow_pressed:
            if time.time() - up_arrow_time < UP_ARROW_TIMEOUT:
                play_beep(START_BEEP_FILE, 'START')
            up_arrow_pressed = False
        
        # 检测 Ctrl 键（记录状态）
        elif key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            ctrl_pressed = True
        
        # 检测 C 键（在 Ctrl 按下时）
        elif hasattr(key, 'char') and key.char == 'c' and ctrl_pressed:
            play_beep(END_BEEP_FILE, 'END')
            ctrl_pressed = False
            
        # 检测 Q 键（Ctrl+Shift+Q 退出）
        elif hasattr(key, 'char') and key.char == 'Q':
            # 检查 Ctrl 和 Shift 是否按下
            return False  # 停止监听
            
    except AttributeError:
        pass


def on_release(key):
    """按键释放回调"""
    global ctrl_pressed
    
    # 释放 Ctrl 键
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        ctrl_pressed = False


def main():
    """主函数"""
    print("=" * 60)
    print("键盘监听蜂鸣同步器")
    print("=" * 60)
    print()
    
    # 检查音频文件
    check_audio_files()
    print()
    
    # 显示使用说明
    print("监听范围: 全局（请确保在终端中使用）")
    print()
    print("触发方式:")
    print("  ⬆️ + Enter (0.5s内) → 播放开始音 (单蜂鸣 1000Hz)")
    print("  Ctrl+C              → 播放结束音 (双蜂鸣 1500Hz)")
    print()
    print("退出方式:")
    print("  Ctrl+Shift+Q        → 退出监听")
    print()
    print("=" * 60)
    print("监听已启动...")
    print("=" * 60)
    print()
    
    # 启动键盘监听
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            pass
    
    print("\n[EXIT] 键盘监听已停止")


if __name__ == '__main__':
    main()

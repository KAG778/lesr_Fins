#!/usr/bin/env python3
"""
API密钥配置工具

用于设置和管理OpenAI API密钥（支持ChatAnywhere）
"""
import os
from pathlib import Path

# API密钥存储文件
KEY_FILE = Path(__file__).parent / ".api_keys.txt"


def save_keys(api_key: str, base_url: str = None):
    """保存API密钥到文件"""
    with open(KEY_FILE, 'w') as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")
        if base_url:
            f.write(f"OPENAI_BASE_URL={base_url}\n")
    print(f"✓ 密钥已保存到 {KEY_FILE}")


def load_keys():
    """从文件加载API密钥"""
    if not KEY_FILE.exists():
        return None, None

    with open(KEY_FILE, 'r') as f:
        lines = f.readlines()

    api_key = None
    base_url = None

    for line in lines:
        if line.startswith('OPENAI_API_KEY='):
            api_key = line.split('=', 1)[1].strip()
        elif line.startswith('OPENAI_BASE_URL='):
            base_url = line.split('=', 1)[1].strip()

    return api_key, base_url


def print_env_commands(api_key: str, base_url: str = None):
    """打印环境变量设置命令"""
    print("\n设置环境变量:")
    print(f"  export OPENAI_API_KEY={api_key}")
    if base_url:
        print(f"  export OPENAI_BASE_URL={base_url}")


def main():
    print("=" * 50)
    print("API密钥配置工具")
    print("=" * 50)

    # 检查是否已有保存的密钥
    saved_key, saved_base = load_keys()
    if saved_key:
        print(f"\n已保存的密钥: {saved_key[:20]}...")
        if saved_base:
            print(f"已保存的base_url: {saved_base}")

    print("\n选项:")
    print("  1. 设置新的API密钥")
    print("  2. 使用ChatAnywhere")
    print("  3. 查看当前配置")
    print("  4. 清除保存的密钥")
    print("  0. 退出")

    choice = input("\n请选择 (0-4): ").strip()

    if choice == '1':
        api_key = input("请输入OpenAI API密钥: ").strip()
        base_url = input("请输入base_url (可选，直接回车跳过): ").strip() or None
        save_keys(api_key, base_url)
        print_env_commands(api_key, base_url)

    elif choice == '2':
        print("\nChatAnywhere配置")
        api_key = input("请输入ChatAnywhere API密钥: ").strip()
        print("\n选择base_url:")
        print("  1. https://api.chatanywhere.cn/v1 (推荐)")
        print("  2. https://api.chatanywhere.com.cn/v1")
        url_choice = input("请选择 (1-2, 默认1): ").strip() or '1'

        base_url = "https://api.chatanywhere.cn/v1" if url_choice == '1' else "https://api.chatanywhere.com.cn/v1"
        save_keys(api_key, base_url)
        print_env_commands(api_key, base_url)

    elif choice == '3':
        if saved_key:
            print(f"\n当前配置:")
            print(f"  API密钥: {saved_key[:20]}...{saved_key[-4:]}")
            if saved_base:
                print(f"  Base URL: {saved_base}")
            else:
                print(f"  Base URL: (使用官方API)")
        else:
            print("\n未保存任何密钥")

    elif choice == '4':
        if KEY_FILE.exists():
            KEY_FILE.unlink()
            print("✓ 已清除保存的密钥")
        else:
            print("没有保存的密钥")

    print("\n" + "=" * 50)


if __name__ == '__main__':
    main()

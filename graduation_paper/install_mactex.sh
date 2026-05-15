#!/bin/bash
# MacTeX 安装脚本

set -e

echo "========================================="
echo "MacTeX 安装脚本"
echo "========================================="
echo ""

# 检查是否已安装
if command -v xelatex &> /dev/null; then
    echo "✓ LaTeX 已经安装"
    xelatex --version | head -1
    exit 0
fi

# 创建临时目录
TMP_DIR="/tmp/mactex_install_$$"
mkdir -p "$TMP_DIR"
cd "$TMP_DIR"

echo "步骤 1/3: 下载 MacTeX..."
echo "   (文件约 4GB, 根据网络情况可能需要 10-30 分钟)"
echo ""

# 使用清华大学镜像加速下载
MIRROR="https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/mac/mactex"
PKG_URL="${MIRROR}/MacTeX.pkg"

# 下载并显示进度
curl -L --progress-bar -o MacTeX.pkg "$PKG_URL" || {
    echo "✗ 下载失败，尝试官方源..."
    curl -L --progress-bar -o MacTeX.pkg "https://mirror.ctan.org/systems/mac/mactex/MacTeX.pkg"
}

echo ""
echo "步骤 2/3: 安装 MacTeX..."
echo "   (需要管理员权限，请输入密码)"
echo ""

sudo installer -pkg MacTeX.pkg -target /

echo ""
echo "步骤 3/3: 验证安装..."
echo ""

# 刷新 PATH
export PATH="/Library/TeX/texbin:$PATH"

# 验证安装
if command -v xelatex &> /dev/null; then
    echo "✓ 安装成功!"
    xelatex --version | head -1
    echo ""
    echo "========================================="
    echo "安装完成! 请重启终端后使用。"
    echo "========================================="
else
    echo "✗ 安装可能有问题，请检查"
    exit 1
fi

# 清理
cd /
rm -rf "$TMP_DIR"

echo ""
echo "接下来编译论文:"
echo "  cd /Users/chenshuaiwen/Desktop/openpi/graduation_paper/thesis"
echo "  xelatex main.tex"

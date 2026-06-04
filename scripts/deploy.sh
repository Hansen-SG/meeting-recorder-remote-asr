#!/bin/bash
# 一键部署：创建 venv、安装依赖、配置 Nginx SSL、启动 systemd 服务
set -e

DEPLOY_DIR="/export/meeting-recorder"
CERT_DIR="/etc/nginx/ssl"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== 会议记录工具部署脚本 ==="
echo "项目目录: $PROJECT_DIR"
echo "部署目录: $DEPLOY_DIR"

# 1. 创建目录
mkdir -p "$DEPLOY_DIR/logs" "$CERT_DIR"

# 2. 复制项目文件
echo "[1/6] 复制项目文件..."
rsync -av --exclude='.venv' --exclude='__pycache__' "$PROJECT_DIR/" "$DEPLOY_DIR/"

# 3. Python venv + 依赖
echo "[2/6] 创建 Python 虚拟环境..."
cd "$DEPLOY_DIR"
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
.venv/bin/pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 检查模型文件
echo "[3/6] 检查模型文件..."
if [ ! -d "$DEPLOY_DIR/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" ]; then
    echo "⚠️  警告：模型文件不存在！请手动复制 models/iic/ 目录到 $DEPLOY_DIR/models/iic/"
    echo "    需要的模型:"
    echo "    - speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    echo "    - speech_fsmn_vad_zh-cn-16k-common-pytorch"
    echo "    - punc_ct-transformer_cn-en-common-vocab471067-large"
    echo "    - speech_campplus_sv_zh-cn_16k-common"
fi

# 5. 生成自签名证书（如不存在）
echo "[4/6] 配置 SSL 证书..."
if [ ! -f "$CERT_DIR/meeting-recorder.crt" ]; then
    openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
        -keyout "$CERT_DIR/meeting-recorder.key" \
        -out "$CERT_DIR/meeting-recorder.crt" \
        -subj "/CN=meeting-recorder"
    echo "    已生成自签名证书"
else
    echo "    证书已存在，跳过"
fi

# 6. Nginx 配置
echo "[5/6] 配置 Nginx..."
cp "$DEPLOY_DIR/deploy/nginx-meeting.conf" /etc/nginx/conf.d/meeting-recorder.conf
# 删除旧的 iptables 端口转发规则（如有）
iptables -t nat -D PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 7860 2>/dev/null || true
nginx -t && systemctl reload nginx

# 7. systemd 服务
echo "[6/6] 配置 systemd 服务..."
cp "$DEPLOY_DIR/deploy/meeting-recorder.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable meeting-recorder
systemctl restart meeting-recorder

echo ""
echo "=== 部署完成 ==="
echo "服务地址: https://$(hostname -I | awk '{print $1}')"
echo "健康检查: curl -k https://localhost/api/health"
echo "查看日志: journalctl -u meeting-recorder -f"

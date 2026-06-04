"""
飞书扫码登录认证模块

OAuth2 流程：
1. 用户访问页面 → 未认证 → 重定向到 /auth/login
2. /auth/login → 重定向到飞书授权页（扫码或账密登录）
3. 飞书回调 /auth/callback?code=xxx → 用 code 换取 user_access_token
4. 获取用户信息 → 写入 session cookie → 重定向到首页
"""
import hashlib
import hmac
import json
import logging
import time
from typing import Optional

import requests
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse, JSONResponse

import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Session 工具函数
# ─────────────────────────────────────────

def _sign(payload: str) -> str:
    """HMAC-SHA256 签名"""
    return hmac.HMAC(
        config.AUTH_SECRET_KEY.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()


def create_session_token(user_info: dict) -> str:
    """创建 session token（JSON payload + HMAC 签名）"""
    payload = json.dumps({
        "user_id": user_info.get("user_id", ""),
        "name": user_info.get("name", ""),
        "avatar": user_info.get("avatar_url", ""),
        "exp": int(time.time()) + config.AUTH_SESSION_EXPIRE,
    }, ensure_ascii=False)
    sig = _sign(payload)
    return f"{payload}|{sig}"


def verify_session_token(token: str) -> Optional[dict]:
    """验证 session token，返回用户信息或 None"""
    if not token or "|" not in token:
        return None
    try:
        payload, sig = token.rsplit("|", 1)
        if not hmac.compare_digest(_sign(payload), sig):
            return None
        data = json.loads(payload)
        if data.get("exp", 0) < time.time():
            return None
        return data
    except Exception:
        return None


# ─────────────────────────────────────────
# 飞书 OAuth2 接口
# ─────────────────────────────────────────

def get_feishu_auth_url(state: str = "login") -> str:
    """生成飞书 OAuth 授权链接"""
    import urllib.parse
    params = urllib.parse.urlencode({
        "app_id": config.FEISHU_APP_ID,
        "redirect_uri": config.AUTH_REDIRECT_URI,
        "state": state,
    })
    return f"{config.FEISHU_BASE_URL}/open-apis/authen/v1/authorize?{params}"


def _get_app_access_token() -> str:
    """获取 app_access_token（内部应用）"""
    url = f"{config.FEISHU_BASE_URL}/open-apis/auth/v3/app_access_token/internal"
    resp = requests.post(url, json={
        "app_id": config.FEISHU_APP_ID,
        "app_secret": config.FEISHU_APP_SECRET,
    }, verify=False, timeout=10)
    data = resp.json()
    if data.get("code") != 0:
        raise Exception(f"获取 app_access_token 失败: {data}")
    return data["app_access_token"]


def exchange_code_for_user(code: str) -> dict:
    """
    用授权码换取用户信息
    返回: {"user_id": "...", "name": "...", "avatar_url": "..."}
    """
    # 1. 获取 app_access_token
    app_token = _get_app_access_token()

    # 2. 用 code 换取 user_access_token
    url = f"{config.FEISHU_BASE_URL}/open-apis/authen/v1/oidc/access_token"
    resp = requests.post(url, json={
        "grant_type": "authorization_code",
        "code": code,
    }, headers={
        "Authorization": f"Bearer {app_token}",
        "Content-Type": "application/json",
    }, verify=False, timeout=10)
    data = resp.json()

    if data.get("code") != 0:
        raise Exception(f"换取 user_access_token 失败: {data}")

    user_token = data["data"]["access_token"]

    # 3. 获取用户信息
    url2 = f"{config.FEISHU_BASE_URL}/open-apis/authen/v1/user_info"
    resp2 = requests.get(url2, headers={
        "Authorization": f"Bearer {user_token}",
    }, verify=False, timeout=10)
    data2 = resp2.json()

    if data2.get("code") != 0:
        raise Exception(f"获取用户信息失败: {data2}")

    user_data = data2.get("data", {})
    return {
        "user_id": user_data.get("user_id", ""),
        "name": user_data.get("name", "未知用户"),
        "avatar_url": user_data.get("avatar_url", ""),
    }


# ─────────────────────────────────────────
# FastAPI 认证中间件
# ─────────────────────────────────────────

# 不需要认证的路径前缀
_PUBLIC_PATHS = (
    "/auth/",
    "/api/health",
    "/static/",
)


class AuthMiddleware(BaseHTTPMiddleware):
    """检查 session cookie，未认证则重定向到登录页"""

    async def dispatch(self, request: Request, call_next):
        # 认证未启用时直接放行
        if not config.AUTH_ENABLED:
            return await call_next(request)

        path = request.url.path

        # 公开路径放行
        for prefix in _PUBLIC_PATHS:
            if path.startswith(prefix):
                return await call_next(request)

        # 检查 session cookie
        token = request.cookies.get("session")
        user = verify_session_token(token)

        if user:
            # 认证通过，将用户信息挂到 request.state
            request.state.user = user
            return await call_next(request)

        # 未认证：API 请求返回 401，页面请求重定向到登录
        if path.startswith("/api/"):
            return JSONResponse({"error": "未登录，请先完成飞书认证"}, status_code=401)
        else:
            return RedirectResponse("/auth/login")

"""
飞书文档导出模块
将会议纪要创建为飞书在线文档

API 模式移植自 feishu-ai-bot.js：
  - feishu_create_doc (lines 1973-2006)
  - feishu_write_doc_content (lines 2141-2177)
  - token 刷新逻辑 (lines 374-399, error code 99991663)
"""
import logging
import re
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin

import requests
import urllib3

import config

logger = logging.getLogger(__name__)

# 飞书私有部署通常使用自签名证书，禁用 SSL 验证
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class FeishuExportError(Exception):
    pass


class FeishuExporter:
    """
    飞书文档导出器

    使用示例：
        exporter = FeishuExporter()
        result = exporter.export_summary("会议标题", "# 纪要内容...")
        print(result["url"])  # 飞书文档链接
    """

    def __init__(
        self,
        base_url: str = config.FEISHU_BASE_URL,
        app_id: str = config.FEISHU_APP_ID,
        app_secret: str = config.FEISHU_APP_SECRET,
        doc_base_url: str = config.FEISHU_DOC_BASE_URL,
    ):
        self._base_url = base_url.rstrip("/")
        self._app_id = app_id
        self._app_secret = app_secret
        self._doc_base_url = doc_base_url.rstrip("/")
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._session = requests.Session()
        self._session.verify = False

    def _get_token(self) -> str:
        """获取 tenant_access_token（带缓存，过期前 60s 刷新）"""
        now = time.time()
        if self._token and now < self._token_expiry - 60:
            return self._token

        url = f"{self._base_url}/open-apis/auth/v3/tenant_access_token/internal"
        resp = self._session.post(url, json={
            "app_id": self._app_id,
            "app_secret": self._app_secret,
        }, timeout=30)
        data = resp.json()
        if data.get("code") != 0:
            raise FeishuExportError(
                f"获取飞书 token 失败: {data.get('msg', 'unknown error')}"
            )
        self._token = data["tenant_access_token"]
        self._token_expiry = now + data.get("expire", 7200)
        return self._token

    def _api_post(self, path: str, body: dict) -> dict:
        """POST 请求，含 token 刷新重试（error code 99991663）"""
        for attempt in range(2):
            token = self._get_token()
            url = f"{self._base_url}{path}"
            resp = self._session.post(
                url,
                json=body,
                headers={"Authorization": f"Bearer {token}"},
                timeout=60,
            )
            data = resp.json()
            if data.get("code") == 99991663 and attempt == 0:
                # Token 过期，清除缓存重试
                logger.info("飞书 token 99991663，刷新中...")
                self._token = None
                self._token_expiry = 0
                continue
            return data
        return data  # type: ignore

    def _api_patch(self, path: str, body: dict, params: Optional[dict] = None) -> dict:
        """PATCH 请求"""
        for attempt in range(2):
            token = self._get_token()
            url = f"{self._base_url}{path}"
            resp = self._session.patch(
                url,
                json=body,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=60,
            )
            data = resp.json()
            if data.get("code") == 99991663 and attempt == 0:
                self._token = None
                self._token_expiry = 0
                continue
            return data
        return data  # type: ignore

    def create_document(
        self,
        title: str,
        folder_token: Optional[str] = None,
    ) -> dict:
        """
        创建飞书文档并设置权限

        返回：{"document_id": "...", "url": "..."}
        """
        body: dict = {"title": title}
        if folder_token:
            body["folder_token"] = folder_token

        result = self._api_post("/open-apis/docx/v1/documents", body)
        if result.get("code") != 0:
            raise FeishuExportError(
                f"创建飞书文档失败: {result.get('msg', 'unknown')} "
                f"(code: {result.get('code')})"
            )

        doc_id = result["data"]["document"]["document_id"]
        doc_url = f"{self._doc_base_url}/docx/{doc_id}"

        # 设置权限：组织内所有人可编辑
        try:
            perm_result = self._api_patch(
                f"/open-apis/drive/v1/permissions/{doc_id}/public",
                body={
                    "external_access_entity": "closed",
                    "link_share_entity": "tenant_editable",
                    "invite_external": False,
                },
                params={"type": "docx"},
            )
            if perm_result.get("code") == 0:
                logger.info(f"文档 {doc_id} 权限已设置为 tenant_editable")
            else:
                logger.warning(
                    f"文档 {doc_id} 权限设置失败: "
                    f"{perm_result.get('code')} {perm_result.get('msg')}"
                )
        except Exception as e:
            logger.warning(f"文档权限设置异常（非致命）: {e}")

        return {"document_id": doc_id, "url": doc_url}

    def write_content(self, document_id: str, content: str) -> dict:
        """
        将 Markdown 内容写入飞书文档

        支持格式：
        - # / ## / ### 标题
        - - / * 无序列表
        - 1. 2. 有序列表
        - > 引用块
        - 普通段落
        """
        paragraphs = [line for line in content.split("\n") if line.strip()]
        if not paragraphs:
            raise FeishuExportError("内容为空")

        children = []
        for line in paragraphs:
            block = self._markdown_line_to_block(line)
            children.append(block)

        # 分批写入，每批最多 50 个 blocks
        BATCH_SIZE = 50
        batches = [
            children[i : i + BATCH_SIZE]
            for i in range(0, len(children), BATCH_SIZE)
        ]
        logger.info(
            f"写入文档 {document_id}: {len(children)} blocks, "
            f"{len(batches)} batch(es)"
        )

        for batch_idx, batch in enumerate(batches):
            result = self._api_post(
                f"/open-apis/docx/v1/documents/{document_id}/blocks/{document_id}/children",
                {"children": batch, "document_revision_id": -1},
            )
            if result.get("code") != 0:
                raise FeishuExportError(
                    f"写入失败（batch {batch_idx + 1}/{len(batches)}）: "
                    f"{result.get('msg', 'unknown')} (code: {result.get('code')})"
                )
            logger.info(
                f"batch {batch_idx + 1}/{len(batches)} OK ({len(batch)} blocks)"
            )

        return {
            "total_blocks": len(children),
            "batches": len(batches),
        }

    def export_summary(
        self,
        title: str,
        summary: str,
        folder_token: Optional[str] = None,
    ) -> dict:
        """
        一站式导出：创建文档 + 写入内容

        返回：{"document_id": "...", "url": "...", "total_blocks": N}
        """
        today = datetime.now().strftime("%Y-%m-%d")
        doc_title = f"会议纪要：{title} ({today})" if title else f"会议纪要 ({today})"

        doc_info = self.create_document(doc_title, folder_token)
        write_info = self.write_content(doc_info["document_id"], summary)

        return {**doc_info, **write_info}

    @staticmethod
    def _markdown_line_to_block(line: str) -> dict:
        """将单行 Markdown 转为飞书 block 格式"""
        stripped = line.strip()

        # 标题
        if stripped.startswith("### "):
            return _heading_block(stripped[4:], level=3)
        if stripped.startswith("## "):
            return _heading_block(stripped[3:], level=2)
        if stripped.startswith("# "):
            return _heading_block(stripped[2:], level=1)

        # 无序列表
        if stripped.startswith("- ") or stripped.startswith("* "):
            return _bullet_block(stripped[2:])

        # 有序列表
        m = re.match(r"^\d+\.\s+(.+)$", stripped)
        if m:
            return _ordered_block(m.group(1))

        # 表格行（飞书不直接支持 Markdown 表格，转为文本）
        if stripped.startswith("|") and stripped.endswith("|"):
            # 跳过分隔行
            if re.match(r"^\|[-\s|:]+\|$", stripped):
                return _text_block(stripped)
            # 提取表格内容
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            return _text_block(" | ".join(cells))

        # 普通文本
        return _text_block(stripped)


def _text_block(content: str) -> dict:
    """block_type: 2 = text"""
    return {
        "block_type": 2,
        "text": {"elements": [{"text_run": {"content": content}}]},
    }


def _heading_block(content: str, level: int = 1) -> dict:
    """block_type: 3=heading1, 4=heading2, 5=heading3"""
    block_type = {1: 3, 2: 4, 3: 5}.get(level, 3)
    return {
        "block_type": block_type,
        "heading" + str(level): {
            "elements": [{"text_run": {"content": content}}]
        },
    }


def _bullet_block(content: str) -> dict:
    """block_type: 12 = bullet"""
    return {
        "block_type": 12,
        "bullet": {"elements": [{"text_run": {"content": content}}]},
    }


def _ordered_block(content: str) -> dict:
    """block_type: 13 = ordered"""
    return {
        "block_type": 13,
        "ordered": {"elements": [{"text_run": {"content": content}}]},
    }

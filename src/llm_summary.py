"""
LLM 会议纪要生成模块
使用 OpenAI 兼容 API 将转写文本生成结构化会议纪要
"""
import logging
import time
from typing import Generator, Optional

import config

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是一个专业的会议纪要整理助手。请根据以下会议转录文本，生成结构化的会议纪要。

纪要应包含以下部分（使用 Markdown 格式）：

# 会议标题（根据内容推断，简洁明确）

## 基本信息
- **会议时间**：（根据转录时间推断）
- **参会人员**：（从对话中识别提及的人名，或按"说话人0/1/2"列出）

## 会议要点
（按主题组织的要点列表，每个要点一句话总结）

## 讨论详情
（按主题展开的详细讨论内容，保留关键观点和论据）

## 决议事项
（明确的决策列表，如无则注明"本次会议未形成明确决议"）

## 行动项
| 行动项 | 负责人 | 截止时间 |
|--------|--------|----------|
| ...    | ...    | ...      |
（从对话中提取具体任务，如未提及则注明"待确认"）

## 下次会议
（如有提及则记录，否则注明"未提及"）

注意：
1. 仅基于转录内容生成，不要编造信息
2. 对于不确定的内容使用"（待确认）"标注
3. 保持客观中立，不加个人判断
4. 中文输出
"""


class LLMSummaryGenerator:
    """LLM 会议纪要生成器（基于 OpenAI 兼容 API）"""

    def __init__(
        self,
        api_key: str = config.LLM_API_KEY,
        base_url: str = config.LLM_API_BASE,
        model: str = config.LLM_MODEL,
    ):
        import httpx
        from openai import OpenAI
        # 企业内网 CA 证书不在默认信任链，需禁用 SSL 验证
        http_client = httpx.Client(verify=False)
        self._client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        self._model = model
        self._timeout = 180.0

    def generate(self, transcript: str, meeting_title: Optional[str] = None) -> str:
        """同步生成完整纪要（非流式）"""
        if not transcript or len(transcript.strip()) < 20:
            return "转录内容太短，无法生成有意义的会议纪要。"

        user_msg = self._build_user_prompt(transcript, meeting_title)
        return self._call_llm(user_msg)

    def generate_streaming(
        self, transcript: str, meeting_title: Optional[str] = None
    ) -> Generator[str, None, None]:
        """流式生成纪要"""
        if not transcript or len(transcript.strip()) < 20:
            yield "转录内容太短，无法生成有意义的会议纪要。"
            return

        user_msg = self._build_user_prompt(transcript, meeting_title)
        yield from self._call_llm_streaming(user_msg)

    def _build_user_prompt(self, transcript: str, title: Optional[str]) -> str:
        title_hint = f"\n（会议主题参考：{title}）" if title else ""
        return f"请根据以下会议转录文本生成结构化的会议纪要：{title_hint}\n\n---\n\n{transcript}"

    def _call_llm(self, user_msg: str) -> str:
        last_err = None
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    timeout=self._timeout,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                logger.warning(f"LLM 调用失败（第 {attempt + 1} 次）: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return f"LLM 调用失败: {last_err}"

    def _call_llm_streaming(self, user_msg: str) -> Generator[str, None, None]:
        last_err = None
        for attempt in range(3):
            try:
                stream = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    stream=True,
                    timeout=self._timeout,
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except Exception as e:
                last_err = e
                logger.warning(f"LLM 流式调用失败（第 {attempt + 1} 次）: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        yield f"\n\n**LLM 调用失败**: {last_err}"

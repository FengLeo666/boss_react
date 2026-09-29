"""Agent construction and initial message assembly."""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import dotenv_values
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .agent_config import AgentSettings, read_required_text
from .context_compaction import AgentNodeCompactionMiddleware, resume_message, task_message
from .nodriver import NodriverBrowserConfig
from .nodriver_middleware import BossReactMiddleware

logger = logging.getLogger(__name__)


def _effective_env(name: str, file_values: dict[str, str | None]) -> str:
    return (os.getenv(name) or file_values.get(name) or "").strip()


def build_model(settings: AgentSettings) -> ChatOpenAI:
    """Build a LangChain model from this project's OpenAI-compatible API config."""
    if not settings.model_env_path.is_file():
        raise FileNotFoundError(f"Model .env not found: {settings.model_env_path}")
    values = dict(dotenv_values(settings.model_env_path))
    api_key = _effective_env("LLM_API_KEY", values)
    model_name = _effective_env("LLM_MODEL", values)
    base_url = _effective_env("LLM_BASE_URL", values)
    if not api_key:
        raise RuntimeError("LLM_API_KEY is missing from the project configuration")
    if not model_name:
        raise RuntimeError("LLM_MODEL is missing from the project configuration")

    if settings.request_timeout_seconds is not None:
        timeout = settings.request_timeout_seconds
    else:
        timeout_raw = _effective_env("BOSS_LLM_TIMEOUT", values)
        try:
            timeout = max(5.0, float(timeout_raw)) if timeout_raw else 30.0
        except ValueError:
            timeout = 30.0
    logger.info(
        "构建模型: model=%s base_url=%s timeout=%.1fs sdk_retries=%d",
        model_name,
        base_url or "default",
        timeout,
        2,
    )
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url or None,
        timeout=timeout,
        max_retries=2,
        temperature=0,
    )


def build_initial_messages(settings: AgentSettings, task: str) -> list[HumanMessage]:
    resume_text = read_required_text(settings.resume_text_path, label="Resume text")
    task = task.strip()
    if not task:
        raise ValueError("Agent task cannot be empty")
    messages = [
        resume_message(
            content=(
                "以下是候选人的简历事实资料。只能据此判断匹配度和撰写沟通内容，"
                "不得补充或猜测不存在的经历：\n\n"
                f"{resume_text}"
            )
        ),
        task_message(task),
    ]
    logger.info("构建初始消息: resume_chars=%d task_chars=%d", len(resume_text), len(task))
    return messages


def build_agent(
    settings: AgentSettings,
    *,
    model: ChatOpenAI | None = None,
    browser_middleware: BossReactMiddleware | None = None,
    checkpointer: Any | None = None,
) -> tuple[Any, BossReactMiddleware]:
    """Create the LangChain agent and return it with its retained browser middleware."""
    chat_model = model or build_model(settings)
    browser = browser_middleware or BossReactMiddleware(
        NodriverBrowserConfig(
            python_executable=settings.project_root / ".venv" / "Scripts" / "python.exe",
            profile_dir=settings.browser_profile_path,
            artifacts_dir=settings.artifacts_dir,
            action_timeout=settings.browser_action_timeout_seconds,
        ),
        resume_image_path=settings.resume_image_path,
    )
    compactor = AgentNodeCompactionMiddleware(
        settings.summary_trigger_tokens, settings.summary_trigger_images
    )
    system_prompt = read_required_text(settings.system_prompt_path, label="System prompt")
    graph = create_agent(
        model=chat_model,
        system_prompt=system_prompt,
        middleware=[compactor, browser],
        checkpointer=checkpointer,
        debug=settings.debug,
        name="boss_react_agent",
    )
    logger.info(
        "Agent 已构建: compaction_token_trigger=%d compaction_image_trigger=%d recursion_limit=%d checkpoint=%s",
        settings.summary_trigger_tokens,
        settings.summary_trigger_images,
        settings.recursion_limit,
        "enabled" if checkpointer is not None else "disabled",
    )
    return graph, browser

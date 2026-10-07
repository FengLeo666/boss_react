"""Configuration loading for the BOSS ReAct agent."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentSettings:
    config_path: Path
    task: str
    system_prompt_path: Path
    resume_text_path: Path
    resume_image_path: Path | None
    model_env_path: Path
    browser_profile_path: Path
    browser_action_timeout_seconds: float
    request_timeout_seconds: float | None
    summary_trigger_tokens: int
    summary_trigger_images: int
    checkpoint_database_path: Path
    checkpoint_thread_id: str
    recursion_limit: int
    debug: bool
    log_level: str
    log_file: Path

    @property
    def project_root(self) -> Path:
        return self.config_path.parent.parent

    @property
    def artifacts_dir(self) -> Path:
        return self.config_path.parent.parent / "artifacts" / "agent"


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"[{name}] must be a TOML table")
    return value


def _resolve(config_dir: Path, raw: Any, *, field_name: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise ValueError(f"{field_name} must be configured")
    path = Path(value).expanduser()
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _resolve_optional(config_dir: Path, raw: Any) -> Path | None:
    value = str(raw or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    resolved = (config_dir / path).resolve() if not path.is_absolute() else path.resolve()
    return resolved if resolved.is_file() else None


def _positive_int(raw: Any, *, field_name: str, default: int) -> int:
    value = default if raw is None else raw
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _optional_positive_float(raw: Any, *, field_name: str) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        raise ValueError(f"{field_name} must be a positive number")
    return float(raw)


def load_agent_settings(path: str | Path) -> AgentSettings:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Agent config not found: {config_path}")
    with config_path.open("rb") as stream:
        data = tomllib.load(stream)

    agent = _section(data, "agent")
    model = _section(data, "model")
    browser = _section(data, "browser")
    context = _section(data, "context")
    checkpoint = _section(data, "checkpoint")
    runtime = _section(data, "runtime")
    config_dir = config_path.parent
    trigger = _positive_int(
        context.get("summary_trigger_tokens"), field_name="summary_trigger_tokens", default=32_000
    )
    image_trigger = _positive_int(
        context.get("summary_trigger_images"), field_name="summary_trigger_images", default=249
    )
    checkpoint_thread_id = str(
        checkpoint.get("thread_id", "boss-react-default")
    ).strip()
    if not checkpoint_thread_id:
        raise ValueError("checkpoint.thread_id must not be empty")
    log_level = str(runtime.get("log_level", "INFO")).strip().upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("log_level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")

    return AgentSettings(
        config_path=config_path,
        task=str(agent.get("task", "")).strip(),
        system_prompt_path=_resolve(
            config_dir, agent.get("system_prompt_path"), field_name="system_prompt_path"
        ),
        resume_text_path=_resolve(
            config_dir, agent.get("resume_text_path"), field_name="resume_text_path"
        ),
        resume_image_path=_resolve_optional(config_dir, agent.get("resume_image_path")),
        model_env_path=_resolve(
            config_dir, model.get("env_file", "../.env"), field_name="model.env_file"
        ),
        browser_profile_path=_resolve(
            config_dir,
            browser.get("profile_dir", "../browser-profile/nodriver"),
            field_name="browser.profile_dir",
        ),
        browser_action_timeout_seconds=(
            _optional_positive_float(
                browser.get("action_timeout_seconds"),
                field_name="browser.action_timeout_seconds",
            )
            or 120.0
        ),
        request_timeout_seconds=_optional_positive_float(
            model.get("request_timeout_seconds"), field_name="request_timeout_seconds"
        ),
        summary_trigger_tokens=trigger,
        summary_trigger_images=image_trigger,
        checkpoint_database_path=_resolve(
            config_dir,
            checkpoint.get("database_path", "../data/checkpoints.sqlite"),
            field_name="checkpoint.database_path",
        ),
        checkpoint_thread_id=checkpoint_thread_id,
        recursion_limit=_positive_int(
            runtime.get("recursion_limit"), field_name="recursion_limit", default=120
        ),
        debug=bool(runtime.get("debug", False)),
        log_level=log_level,
        log_file=_resolve(
            config_dir,
            runtime.get("log_file", "logs/boss-react.log"),
            field_name="log_file",
        ),
    )


def read_required_text(path: Path, *, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{label} file is empty: {path}")
    return text

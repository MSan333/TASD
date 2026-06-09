"""Skill 系统 — 加载 .md 格式的分析 Skill，组装进 Agent prompt.

Skill 是引导 LLM 如何分析商家数据的 prompt 模板（.md 文件），不是 Python 代码。
SkillLoader 负责加载、按 ctx 数据动态选择启用、拼接成 prompt。

用法：
    from skills import SkillLoader

    loader = SkillLoader("skills/")
    analysis_prompt = loader.build_analysis_prompt(ctx)
    matching_prompt = loader.get_matching_prompt()
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _parse_frontmatter(content: str) -> tuple:
    """解析 .md 文件的 YAML frontmatter，返回 (metadata_dict, body_text).

    不依赖 pyyaml，手动解析简单的 key: value 格式。
    支持列表格式（以 - 开头的行）。
    """
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    yaml_text = parts[1].strip()
    body = parts[2].strip()

    metadata: Dict[str, Any] = {}
    current_key = None
    current_list: Optional[List[str]] = None

    for line in yaml_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("- "):
            if current_key and current_list is not None:
                current_list.append(line[2:].strip())
            continue

        if current_key and current_list is not None:
            metadata[current_key] = current_list
            current_list = None
            current_key = None

        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            if not value:
                current_key = key
                current_list = []
            elif value.lower() == "true":
                metadata[key] = True
            elif value.lower() == "false":
                metadata[key] = False
            else:
                metadata[key] = value

    if current_key and current_list is not None:
        metadata[current_key] = current_list

    return metadata, body


class SkillLoader:
    """加载 .md Skill 文件，按 ctx 数据动态选择启用."""

    def __init__(self, skills_dir: str = "skills/"):
        self.skills_dir = skills_dir
        self.config = self._load_config()
        self.skills: Dict[str, List[Dict[str, Any]]] = {
            "analysis": [],
            "matching": [],
            "decision": [],
        }
        self._load_all()

    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(self.skills_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"enabled": True}

    def _load_all(self):
        if not self.config.get("enabled", True):
            logger.info("Skill 系统已禁用 (config.enabled=false)")
            return

        for layer in ["analysis", "matching", "decision"]:
            layer_dir = os.path.join(self.skills_dir, layer)
            if not os.path.isdir(layer_dir):
                continue

            for filename in sorted(os.listdir(layer_dir)):
                if not filename.endswith(".md"):
                    continue

                filepath = os.path.join(layer_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    raw = f.read()

                metadata, body = _parse_frontmatter(raw)

                if not metadata.get("enabled", True):
                    continue

                skill = {
                    "name": metadata.get("name", filename.replace(".md", "")),
                    "description": metadata.get("description", ""),
                    "layer": metadata.get("layer", layer),
                    "requires_input": metadata.get("requires_input", []),
                    "depends_on": metadata.get("depends_on", []),
                    "version": metadata.get("version", "v1"),
                    "content": body,
                    "file": filepath,
                }

                self.skills[layer].append(skill)
                logger.debug(f"加载 Skill: {layer}/{filename} ({skill['name']})")

        total = sum(len(v) for v in self.skills.values())
        logger.info(
            f"Skill 系统已加载: "
            f"analysis={len(self.skills['analysis'])}, "
            f"matching={len(self.skills['matching'])}, "
            f"decision={len(self.skills['decision'])}, "
            f"总计={total}"
        )

    def build_analysis_prompt(self, ctx: Dict[str, Any]) -> tuple:
        """按 ctx 有哪些数据，选择性启用 analysis Skill，拼成分析 prompt.

        Returns:
            (prompt_text, selected_skill_names) 元组。
            prompt_text 为空字符串时表示没有适用的 Skill。
        """
        sections = []
        selected_names = []
        for skill in self.skills.get("analysis", []):
            required = skill.get("requires_input", [])
            if required and not all(ctx.get(field) for field in required):
                continue
            sections.append(skill["content"])
            selected_names.append(skill["name"])

        if not sections:
            return "", []

        header = (
            "你是广告排序预分析助手。请按以下分析框架，逐项分析当前商家的状态。\n"
            "每项分析输出一句话结论，格式严格按照各项的输出格式要求。\n"
            "所有分析完成后，不要输出其他内容。\n"
        )

        return header + "\n\n---\n\n".join(sections), selected_names

    def get_matching_prompt(self) -> str:
        """获取 matching 层 Skill 内容，注入 system_prompt."""
        parts = [s["content"] for s in self.skills.get("matching", [])]
        return "\n\n".join(parts) if parts else ""

    def get_decision_prompt(self) -> str:
        """获取 decision 层 Skill 内容，注入 system_prompt."""
        parts = [s["content"] for s in self.skills.get("decision", [])]
        return "\n\n".join(parts) if parts else ""

    def get_all_skill_summaries(self) -> str:
        """获取所有 Skill 的摘要（用于诊断）."""
        lines = []
        for layer, skills in self.skills.items():
            for s in skills:
                lines.append(f"[{layer}] {s['name']}: {s['description']}")
        return "\n".join(lines)

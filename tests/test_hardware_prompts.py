"""Tests for the hardware design prompt profile."""

from prompts import (
    report_writer_instructions,
    task_summarizer_instructions,
    todo_planner_system_prompt,
)


def test_planner_prompt_targets_hardware_design():
    assert "硬件系统架构师" in todo_planner_system_prompt
    assert "主控/SoC/处理器选型" in todo_planner_system_prompt
    assert "hardware_design" in todo_planner_system_prompt


def test_summarizer_prompt_requires_engineering_outputs():
    assert "硬件模块方案分析专家" in task_summarizer_instructions
    assert "候选器件" in task_summarizer_instructions
    assert "需进一步确认" in task_summarizer_instructions


def test_report_prompt_uses_hardware_report_template():
    assert "硬件方案设计报告" in report_writer_instructions
    assert "核心器件选型" in report_writer_instructions
    assert "风险矩阵" in report_writer_instructions
    assert "待用户确认的问题" in report_writer_instructions

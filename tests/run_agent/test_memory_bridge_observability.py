import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from run_agent import AIAgent


RAW_CONTENT = "secret raw memory content"
OLD_TEXT = "secret old memory text"
RAW_RESULT_SECRET = "secret result payload"


def _agent(memory_manager=None, session_id="session-sensitive-1234567890"):
    agent = AIAgent.__new__(AIAgent)
    agent._memory_store = object()
    agent._memory_manager = memory_manager
    agent.session_id = session_id
    return agent


def _patch_memory_tool(monkeypatch, result):
    calls = []

    def fake_memory_tool(*, action, target, content, old_text, store):
        calls.append({
            "action": action,
            "target": target,
            "content": content,
            "old_text": old_text,
            "store": store,
        })
        return result

    monkeypatch.setattr("tools.memory_tool.memory_tool", fake_memory_tool)
    return calls


def _messages_from(caplog):
    return "\n".join(record.getMessage() for record in caplog.records)


def test_successful_add_notifies_once_logs_notified_and_preserves_result(monkeypatch, caplog):
    result = json.dumps({"success": True, "message": RAW_RESULT_SECRET})
    _patch_memory_tool(monkeypatch, result)
    manager = MagicMock()
    manager.on_memory_write.return_value = {"attempted": 1, "delivered": 1, "failed": 0, "providers": ["ext"]}
    agent = _agent(manager)

    with caplog.at_level(logging.DEBUG):
        returned = agent._invoke_builtin_memory_tool_with_bridge(
            {"action": "add", "target": "memory", "content": RAW_CONTENT, "old_text": OLD_TEXT},
            task_id="task-abcdef123456",
            tool_call_id="call-abcdef123456",
            execution_path="direct",
        )

    assert returned == result
    manager.on_memory_write.assert_called_once_with("add", "memory", RAW_CONTENT)
    messages = _messages_from(caplog)
    assert "event=memory_bridge.notified" in messages
    assert "content_len=" in messages
    assert RAW_CONTENT not in messages
    assert OLD_TEXT not in messages
    assert RAW_RESULT_SECRET not in messages
    assert "session-sensitive-1234567890" not in messages


def test_replace_notifies(monkeypatch):
    _patch_memory_tool(monkeypatch, json.dumps({"success": True}))
    manager = MagicMock()
    manager.on_memory_write.return_value = {"attempted": 1, "delivered": 1, "failed": 0, "providers": ["ext"]}
    agent = _agent(manager)

    returned = agent._invoke_builtin_memory_tool_with_bridge(
        {"action": "replace", "target": "user", "content": "updated", "old_text": OLD_TEXT},
        execution_path="direct",
    )

    assert returned == json.dumps({"success": True})
    manager.on_memory_write.assert_called_once_with("replace", "user", "updated")


@pytest.mark.parametrize(
    "args,result,reason",
    [
        ({"action": "add", "target": "memory", "content": RAW_CONTENT}, json.dumps({"success": False}), "result_success_not_true"),
        ({"action": "add", "target": "memory", "content": RAW_CONTENT}, "not-json", "result_unparseable"),
        ({"action": "add", "target": "memory", "content": RAW_CONTENT}, json.dumps([{"success": True}]), "result_not_dict"),
        ({"action": "add", "target": "memory", "content": RAW_CONTENT}, json.dumps({"message": "missing"}), "result_success_not_true"),
        ({"action": "remove", "target": "memory", "content": RAW_CONTENT}, json.dumps({"success": True}), "action_not_bridged"),
        ({"action": "read", "target": "memory", "content": RAW_CONTENT}, json.dumps({"success": True}), "action_not_bridged"),
        ({"action": "unknown", "target": "memory", "content": RAW_CONTENT}, json.dumps({"success": True}), "action_not_bridged"),
    ],
)
def test_skip_conditions_do_not_notify_and_log_sanitized(monkeypatch, caplog, args, result, reason):
    _patch_memory_tool(monkeypatch, result)
    manager = MagicMock()
    agent = _agent(manager)

    with caplog.at_level(logging.DEBUG):
        returned = agent._invoke_builtin_memory_tool_with_bridge(args, execution_path="direct")

    assert returned == result
    manager.on_memory_write.assert_not_called()
    messages = _messages_from(caplog)
    assert "event=memory_bridge.skipped" in messages
    assert f"reason={reason}" in messages
    assert RAW_CONTENT not in messages
    assert result not in messages


def test_no_manager_skips_without_notify(monkeypatch, caplog):
    result = json.dumps({"success": True})
    _patch_memory_tool(monkeypatch, result)
    agent = _agent(None)

    with caplog.at_level(logging.DEBUG):
        assert agent._invoke_builtin_memory_tool_with_bridge(
            {"action": "add", "target": "memory", "content": RAW_CONTENT},
            execution_path="direct",
        ) == result

    messages = _messages_from(caplog)
    assert "event=memory_bridge.skipped" in messages
    assert "reason=no_manager" in messages
    assert RAW_CONTENT not in messages


def test_manager_summary_failed_logs_no_misleading_notified(monkeypatch, caplog):
    result = json.dumps({"success": True})
    _patch_memory_tool(monkeypatch, result)
    manager = MagicMock()
    manager.on_memory_write.return_value = {"attempted": 1, "delivered": 0, "failed": 1, "providers": ["bad"]}
    agent = _agent(manager)

    with caplog.at_level(logging.DEBUG):
        assert agent._invoke_builtin_memory_tool_with_bridge(
            {"action": "add", "target": "memory", "content": RAW_CONTENT},
            execution_path="direct",
        ) == result

    messages = _messages_from(caplog)
    assert "event=memory_bridge.notified" not in messages
    assert "event=memory_bridge.skipped" in messages
    assert "reason=manager_summary_not_delivered" in messages
    assert RAW_CONTENT not in messages


def test_manager_exception_warning_nonfatal_sanitized(monkeypatch, caplog):
    result = json.dumps({"success": True})
    _patch_memory_tool(monkeypatch, result)
    manager = MagicMock()
    manager.on_memory_write.side_effect = RuntimeError("provider leaked secret exception")
    agent = _agent(manager)

    with caplog.at_level(logging.DEBUG):
        returned = agent._invoke_builtin_memory_tool_with_bridge(
            {"action": "add", "target": "memory", "content": RAW_CONTENT},
            execution_path="direct",
        )

    assert returned == result
    messages = _messages_from(caplog)
    assert "event=memory_bridge.failed" in messages
    assert "exception_type=RuntimeError" in messages
    assert "provider leaked secret exception" not in messages
    assert RAW_CONTENT not in messages


def test_invoke_tool_memory_path_uses_bridge_helper(monkeypatch):
    agent = _agent(MagicMock())
    agent._memory_manager.has_tool.return_value = False
    calls = []

    def fake_helper(function_args, *, task_id="", tool_call_id="", execution_path=""):
        calls.append((function_args, task_id, tool_call_id, execution_path))
        return "bridge-result"

    monkeypatch.setattr(agent, "_invoke_builtin_memory_tool_with_bridge", fake_helper)

    assert agent._invoke_tool("memory", {"action": "add", "content": RAW_CONTENT}, "task-123456", tool_call_id="call-123456") == "bridge-result"
    assert calls == [({"action": "add", "content": RAW_CONTENT}, "task-123456", "call-123456", "direct")]


def test_sequential_memory_path_uses_bridge_helper(monkeypatch):
    agent = _agent(MagicMock())
    agent._interrupt_requested = False
    agent._turns_since_memory = 5
    agent.quiet_mode = True
    agent.verbose_logging = False
    agent.log_prefix = ""
    agent.log_prefix_chars = 80
    agent.tool_progress_callback = None
    agent.tool_start_callback = None
    agent.tool_complete_callback = None
    agent._checkpoint_mgr = SimpleNamespace(enabled=False)
    agent.valid_tool_names = {"memory"}
    agent._subdirectory_hints = SimpleNamespace(check_tool_call=lambda name, args: "")
    agent._current_tool = None
    agent._touch_activity = lambda *_args, **_kwargs: None
    agent._vprint = lambda *_args, **_kwargs: None
    agent._should_emit_quiet_tool_messages = lambda: False
    agent.tool_delay = 0

    calls = []

    def fake_helper(function_args, *, task_id="", tool_call_id="", execution_path=""):
        calls.append((function_args, task_id, tool_call_id, execution_path))
        return "bridge-result"

    monkeypatch.setattr(agent, "_invoke_builtin_memory_tool_with_bridge", fake_helper)

    tool_call = SimpleNamespace(
        id="call-sequential-123456",
        function=SimpleNamespace(
            name="memory",
            arguments=json.dumps({"action": "add", "target": "memory", "content": RAW_CONTENT}),
        ),
    )
    assistant_message = SimpleNamespace(tool_calls=[tool_call])
    messages = []

    agent._execute_tool_calls_sequential(assistant_message, messages, "task-sequential-123456")

    assert calls == [({"action": "add", "target": "memory", "content": RAW_CONTENT}, "task-sequential-123456", "call-sequential-123456", "sequential")]
    assert messages[-1] == {"role": "tool", "content": "bridge-result", "tool_call_id": "call-sequential-123456"}


def test_flush_memories_path_uses_bridge_helper(monkeypatch):
    agent = _agent(MagicMock())
    agent._memory_flush_min_turns = 1
    agent.valid_tool_names = {"memory"}
    agent._user_turn_count = 3
    agent.api_mode = "chat_completions"
    agent._cached_system_prompt = ""
    agent.tools = [{"type": "function", "function": {"name": "memory", "parameters": {}}}]
    agent.model = "test-model"
    agent.quiet_mode = True

    tool_call = SimpleNamespace(
        function=SimpleNamespace(
            name="memory",
            arguments=json.dumps({"action": "add", "target": "memory", "content": RAW_CONTENT}),
        )
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tool_call]))]
    )
    monkeypatch.setattr("agent.auxiliary_client.call_llm", lambda **_kwargs: response)
    monkeypatch.setattr("agent.auxiliary_client._fixed_temperature_for_model", lambda _model: None)

    # If flush_memories still calls tools.memory_tool directly, this fake prevents
    # real memory writes but the assertion below will fail because the bridge
    # helper was bypassed.
    _patch_memory_tool(monkeypatch, json.dumps({"success": True}))

    calls = []

    def fake_helper(function_args, *, task_id="", tool_call_id="", execution_path=""):
        calls.append((function_args, task_id, tool_call_id, execution_path))
        return json.dumps({"success": True})

    monkeypatch.setattr(agent, "_invoke_builtin_memory_tool_with_bridge", fake_helper)

    messages = [
        {"role": "user", "content": "remember this"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "next"},
    ]

    agent.flush_memories(messages, min_turns=0)

    assert calls == [({"action": "add", "target": "memory", "content": RAW_CONTENT}, "flush_memories", "", "flush")]
    assert all("_flush_sentinel" not in msg for msg in messages)

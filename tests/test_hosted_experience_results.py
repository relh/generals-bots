import pytest

from integrations.softmax.experience_results import panel_is_complete


def request(prefix, count):
    return dict(
        status="completed",
        completed_count=count,
        pending_count=0,
        running_count=0,
        failed_count=0,
        episodes=[dict(status="completed", episode_id=f"{prefix}-{i}") for i in range(count)],
    )


def test_terminal_counters_with_only_63_records_are_not_complete():
    states = [request(str(i), 16) for i in range(4)]
    states[-1]["episodes"].pop()
    assert not panel_is_complete(states, [16] * 4)
    states[-1]["episodes"].append(dict(status="completed", episode_id="3-15"))
    assert panel_is_complete(states, [16] * 4)


def test_duplicate_episode_cannot_count_twice():
    states = [request("same", 1), request("same", 1)]
    with pytest.raises(ValueError, match="duplicate episode"):
        panel_is_complete(states, [1, 1])


def test_failed_episode_is_not_a_loss_or_completion():
    state = request("a", 1)
    state["episodes"][0]["status"] = "failed"
    with pytest.raises(ValueError, match="execution failure"):
        panel_is_complete([state], [1])


def test_incomplete_startup_cannot_open_the_main_panel():
    first, second = request("a", 1), request("b", 1)
    second["status"] = "running"
    assert not panel_is_complete([first, second], [1, 1])
    assert not panel_is_complete([first], [1, 1])

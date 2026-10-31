from encore.agent import run_agent
from encore.memory import Memory
from encore.tools import Tools


class FakeLB:
    token = ""
    def count(self):
        return {"count": 42}


class FakeOllama:
    def __init__(self):
        self.calls = 0
        self.observations = []

    def chat(self, messages, tools):
        self.calls += 1
        if messages[-1]["role"] == "tool":
            self.observations.append(messages[-1]["content"])
        if self.calls == 1:
            return {"message": {"tool_calls": [{"function": {"name": "get_listen_count", "arguments": {}}}]},
                    "prompt_eval_count": 100, "eval_count": 10}
        return {"message": {"tool_calls": [{"function": {"name": "final_answer",
                    "arguments": {"answer": "You have 42 listens", "status": "answered"}}}]},
                "prompt_eval_count": 100, "eval_count": 10}


def test_react_observes_tool_and_finalizes(tmp_path):
    mem = Memory(tmp_path / "m.db")
    try:
        model = FakeOllama()
        tools = Tools(FakeLB(), None, mem, approve=lambda _: False,
                      current_goal="How many songs have I listened to?")
        result = run_agent("How many songs have I listened to?", tools, model)
        assert result.status == "answered"
        assert result.steps == 2
        assert result.prompt_tokens == 200
        assert '42' in model.observations[0]
    finally:
        mem.close()


class NoToolModel:
    def chat(self, messages, tools):
        return {"message": {"content": "Maybe answer without data"}}


def test_agent_does_not_confuse_freeform_guess_with_final(tmp_path):
    mem = Memory(tmp_path / "m.db")
    try:
        tools = Tools(FakeLB(), None, mem, approve=lambda _: False, current_goal="How many listens?")
        result = run_agent("How many listens?", tools, NoToolModel(), max_steps=2)
        assert result.status == "budget_exceeded"
    finally:
        mem.close()

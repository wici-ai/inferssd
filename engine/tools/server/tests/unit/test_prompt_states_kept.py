# SPDX-License-Identifier: Apache-2.0
# A conversation of a model whose memory cannot be rolled back (sliding window or hybrid, such as Gemma 4 or Qwen3.6)
# goes on from its own state after another conversation behind the same system prompt and tools took the slot,
# checked against a running server of such a model with a prompt cache, as `inferssd serve` starts it:
#   DEBUG_EXTERNAL=1 PORT=<port> pytest unit/test_prompt_states_kept.py
# Without DEBUG_EXTERNAL the tests are skipped: there is no such model among the test presets.

import uuid

import pytest

from utils import *

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    if "DEBUG_EXTERNAL" not in os.environ:
        pytest.skip("needs a running server of a sliding-window or hybrid model with a prompt cache (DEBUG_EXTERNAL, PORT)")
    server = ServerProcess()


def system_prompt() -> str:
    # New to the server on every run, so that no earlier state serves it.
    rules = "\n".join(f"Rule {i}: answer the user's question {i} in plain words and keep each answer short."
                      for i in range(1, 61))
    return f"Session {uuid.uuid4()}. You are a careful assistant.\n{rules}"


def ask(messages: list) -> dict:
    body = {"messages": messages, "max_tokens": 8, "cache_prompt": True}
    reasoning = server.make_request("GET", "/props").body.get("reasoning") or {}
    if not reasoning or reasoning.get("can_turn_off"):
        body["reasoning_effort"] = "none"
    res = server.make_request("POST", "/v1/chat/completions", data=body, timeout=600)
    assert res.status_code == 200, res.body
    return res.body


def test_a_conversation_left_for_another_goes_on_from_its_state():
    system = {"role": "system", "content": system_prompt()}
    # a first question of a few hundred tokens: the next conversation keeps more than half of the slot (the system
    # prompt), so upstream's rule would let the first conversation's own part go
    question = "Here are my notes: " + " ".join(f"note {i} says the value is {i * 7}." for i in range(1, 26)) + \
        " What is the value of note 3?"
    first_messages = [system, {"role": "user", "content": question}]
    first = ask(first_messages)
    reply = first["choices"][0]["message"]["content"]
    # another conversation behind the same system prompt takes the slot
    ask([system, {"role": "user", "content": "Name one primary color."}])
    # the first conversation goes on
    nxt = ask(first_messages + [{"role": "assistant", "content": reply},
                                {"role": "user", "content": "And the value of note 4?"}])
    timings = nxt["timings"]
    assert timings["cache_n"] >= first["timings"]["prompt_n"] - 64, timings

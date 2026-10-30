# SPDX-License-Identifier: Apache-2.0
"""Small streaming terminal client for the server owned by this invocation."""

from __future__ import annotations

import json
import urllib.request
from typing import Optional


def completion(port: int, messages: list[dict[str, str]], max_tokens: int) -> str:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(
            {
                "messages": messages,
                "stream": True,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode(),
        {"Content-Type": "application/json"},
    )
    result: list[str] = []
    # Ignore proxy environment variables for the private loopback server.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=7200) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                print(flush=True)
                return "".join(result)
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            choices = event.get("choices", [])
            if choices:
                content = choices[0].get("delta", {}).get("content") or ""
                result.append(content)
                print(content, end="", flush=True)
    raise RuntimeError("server closed the response before completing the answer")


def chat(port: int, prompt: Optional[str], max_tokens: int) -> int:
    messages: list[dict[str, str]] = []
    if prompt is not None:
        completion(port, [{"role": "user", "content": prompt}], max_tokens)
        return 0
    print("InferSSD chat. /exit to quit, /clear to start a new conversation.")
    while True:
        try:
            text = input("\n> ").strip()
        except EOFError:
            print()
            return 0
        if text == "/exit":
            return 0
        if text == "/clear":
            messages.clear()
            continue
        if not text:
            continue
        messages.append({"role": "user", "content": text})
        answer = completion(port, messages, max_tokens)
        messages.append({"role": "assistant", "content": answer})

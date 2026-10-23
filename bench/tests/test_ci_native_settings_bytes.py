"""Parse the reserved settings line strictly without decoding unrelated logs."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.staged_checkpoint import native_settings
from ci.stage_session import BrokerError


PREFIX = b"[ssd.settings] "
VALID = PREFIX + b'{"values":{"vram.budget":32}}\n'


def test_truncated_tokenizer_display_is_not_a_settings_encoding_error(tmp_path):
    # The real DeepSeek loader truncates its token-array display mid-codepoint.
    # Preserve the original log bytes, including that unrelated malformed UTF-8.
    raw = b'llama_model_loader: tokenizer.ggml.tokens = ["<\xef...\n' + VALID + b"other\xff log\n"
    path = tmp_path / "stderr.log"
    path.write_bytes(raw)
    assert native_settings(path) == dict(values={"vram.budget": 32})
    assert path.read_bytes() == raw


@pytest.mark.parametrize("raw", [
    PREFIX + b'{"values":{"x":"\xff"}}\n',
    PREFIX + b'{"values":\n',
    PREFIX + b'{"values":{},"values":{}}\n',
    VALID + VALID,
    b"non-target\xff log only\n",
    PREFIX + b'{"values":null}\n',
])
def test_malformed_target_duplicate_or_missing_snapshot_is_rejected(tmp_path, raw):
    path = tmp_path / "stderr.log"
    path.write_bytes(raw)
    with pytest.raises((ValueError, BrokerError)):
        native_settings(path)
    assert path.read_bytes() == raw

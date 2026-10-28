"""Complete settings output consumed by the reference launcher."""

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_composed_original_input_is_frozen_settings_command_output():
    settings = json.loads((FIXTURES / "settings-command-goldens.json").read_text())["cases"][0]
    launch = json.loads((FIXTURES / "launch-settings-composed-reference.json").read_text())
    for runtime, observed in zip(("3.9", "3.12"), launch):
        assert (
            observed["case"]["settings"]
            == settings["expected"][runtime]["writes"]["/case/settings.env"]
        )
        assert observed["runs"][0]["code"] == 0
        assert "ARG --ctx-checkpoints\nARG 1\n" in observed["runs"][0]["stdout"]

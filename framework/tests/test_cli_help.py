"""The top-level help lists the public commands; the others still run."""

import pytest

from inferssd.cli import main
from inferssd.configuration.settings import host_platform


def test_help_lists_public_commands_only(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    text = capsys.readouterr().out
    assert error.value.code == 0 and "{launch,configuration,source,profile}" in text
    assert "launch-recorded" not in text and "bootstrap-records" not in text


def test_unlisted_commands_still_parse(capsys):
    with pytest.raises(SystemExit) as error:
        main(["plan", "--help"])
    assert error.value.code == 0 and " plan [-h] --prompts" in capsys.readouterr().out


def test_configuration_platform_defaults_to_this_host(tmp_path):
    host = host_platform()
    implicit, explicit = tmp_path / "implicit.ssd", tmp_path / "explicit.ssd"
    assert main(["configuration", "write", "--output", str(implicit)]) == 0
    assert main(["configuration", "write", "--platform", host, "--output", str(explicit)]) == 0
    assert implicit.read_text() == explicit.read_text()

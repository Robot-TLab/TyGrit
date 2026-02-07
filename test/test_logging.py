"""Tests for TyGrit.logging -- level filtering, VIZ level, per-module overrides."""

from __future__ import annotations

from TyGrit.logging import configure, log


def _capture(capsys, level: str, modules=None):
    """Reconfigure, emit one message at each level, return captured stderr."""
    configure(level=level, modules=modules)

    log.debug("msg_debug")
    log.log("VIZ", "msg_viz")
    log.info("msg_info")
    log.warning("msg_warning")
    log.error("msg_error")

    return capsys.readouterr().err


class TestLevelFiltering:
    def test_debug_shows_all(self, capsys):
        out = _capture(capsys, "DEBUG")
        assert "msg_debug" in out
        assert "msg_viz" in out
        assert "msg_info" in out
        assert "msg_warning" in out
        assert "msg_error" in out

    def test_viz_hides_debug(self, capsys):
        out = _capture(capsys, "VIZ")
        assert "msg_debug" not in out
        assert "msg_viz" in out
        assert "msg_info" in out

    def test_info_hides_debug_and_viz(self, capsys):
        out = _capture(capsys, "INFO")
        assert "msg_debug" not in out
        assert "msg_viz" not in out
        assert "msg_info" in out
        assert "msg_warning" in out
        assert "msg_error" in out

    def test_warning_hides_info(self, capsys):
        out = _capture(capsys, "WARNING")
        assert "msg_info" not in out
        assert "msg_warning" in out
        assert "msg_error" in out

    def test_error_only(self, capsys):
        out = _capture(capsys, "ERROR")
        assert "msg_warning" not in out
        assert "msg_error" in out


class TestModuleOverrides:
    def test_override_makes_module_verbose(self, capsys):
        configure(level="ERROR", modules={"test_logging": "DEBUG"})
        log.debug("from_test_module")
        out = capsys.readouterr().err
        # The test runs inside module "test_logging", so the override applies
        assert "from_test_module" in out

    def test_override_does_not_leak(self, capsys):
        # Global is ERROR, only "some.other.module" is DEBUG
        configure(level="ERROR", modules={"some.other.module": "DEBUG"})
        log.debug("should_be_hidden")
        out = capsys.readouterr().err
        # This test's module name is "test_logging", not "some.other.module"
        assert "should_be_hidden" not in out


class TestReconfigure:
    def test_reconfigure_replaces(self, capsys):
        configure(level="ERROR")
        log.info("hidden")
        out1 = capsys.readouterr().err
        assert "hidden" not in out1

        configure(level="DEBUG")
        log.info("visible")
        out2 = capsys.readouterr().err
        assert "visible" in out2


class TestVizLevel:
    def test_viz_method(self, capsys):
        configure(level="DEBUG")
        log.log("VIZ", "viz_message_here")
        out = capsys.readouterr().err
        assert "viz_message_here" in out
        assert "VIZ" in out

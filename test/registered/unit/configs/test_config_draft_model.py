"""Unit tests for ModelConfig._config_draft_model EAGLE3 alias handling."""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import ModelConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def _make_config(arch: str, is_draft_model: bool = True) -> ModelConfig:
    cfg = ModelConfig.__new__(ModelConfig)  # bypass heavy __init__
    cfg.is_draft_model = is_draft_model
    cfg.hf_config = SimpleNamespace(architectures=[arch])
    return cfg


class TestConfigDraftModel(unittest.TestCase):
    def test_eagle3_alias_is_normalized(self):
        cfg = _make_config("Eagle3LlamaForCausalLM")
        cfg._config_draft_model()
        self.assertEqual(cfg.hf_config.architectures[0], "LlamaForCausalLMEagle3")

    def test_non_draft_model_is_untouched(self):
        cfg = _make_config("Eagle3LlamaForCausalLM", is_draft_model=False)
        cfg._config_draft_model()
        self.assertEqual(cfg.hf_config.architectures[0], "Eagle3LlamaForCausalLM")

    def test_unknown_arch_is_untouched(self):
        cfg = _make_config("SomeRandomNewArchForCausalLM")
        cfg._config_draft_model()
        self.assertEqual(cfg.hf_config.architectures[0], "SomeRandomNewArchForCausalLM")


if __name__ == "__main__":
    unittest.main()

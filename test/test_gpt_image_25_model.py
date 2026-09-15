from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("CHATGPT2API_AUTH_KEY", "test-auth")

from services.config import config
from services.openai_backend_api import OpenAIBackendAPI
from services.protocol import openai_v1_models
from services.protocol.conversation import ImageGenerationError, ConversationRequest, stream_image_outputs_with_pool
from utils.helper import IMAGE_MODELS, is_codex_image_model, is_supported_image_model, split_image_model


class GptImage25ModelTests(unittest.TestCase):
    def test_split_image_model_accepts_gpt_image_25(self) -> None:
        self.assertEqual(split_image_model("gpt-image-2.5"), (None, "gpt-image-2.5"))
        self.assertEqual(split_image_model("GPT-Image-2.5"), (None, "gpt-image-2.5"))
        self.assertEqual(split_image_model(" gpt-image-2.5 "), (None, "gpt-image-2.5"))

    def test_gpt_image_25_is_supported_web_model(self) -> None:
        self.assertTrue(is_supported_image_model("gpt-image-2.5"))
        self.assertFalse(is_codex_image_model("gpt-image-2.5"))
        self.assertIn("gpt-image-2.5", IMAGE_MODELS)

    def test_unknown_image_model_still_rejected(self) -> None:
        self.assertFalse(is_supported_image_model("gpt-image-3"))
        self.assertFalse(is_supported_image_model("gpt-image-2.6"))

    def test_image_model_settings_maps_gpt_image_25_to_upstream_model(self) -> None:
        backend = OpenAIBackendAPI()
        try:
            with mock.patch.object(config, "data", {
                **config.data,
                "default_upstream_model_name": "gpt-5-5-extended",
                "default_thinking_effort": "auto",
            }):
                self.assertEqual(backend._image_model_settings("gpt-image-2.5"), ("gpt-5-5", "extended"))
            with mock.patch.object(config, "data", {
                **config.data,
                "default_upstream_model_name": "gpt-5-5",
                "default_thinking_effort": "auto",
            }):
                self.assertEqual(backend._image_model_settings("gpt-image-2.5"), ("gpt-5-5", ""))
        finally:
            backend.close()

    def test_list_models_exposes_gpt_image_25_for_web_accounts(self) -> None:
        with (
            mock.patch.object(
                openai_v1_models.model_catalog_service,
                "list_models",
                return_value={"object": "list", "data": []},
            ),
            mock.patch.object(
                openai_v1_models.account_service,
                "list_accounts",
                return_value=[
                    {"access_token": "token-web-plus", "type": "Plus", "source_type": "web"},
                ],
            ),
        ):
            result = openai_v1_models.list_models()

        ids = {item["id"] for item in result["data"]}
        self.assertIn("gpt-image-2", ids)
        self.assertIn("gpt-image-2.5", ids)
        self.assertNotIn("codex-gpt-image-2", ids)

    def test_image_pool_accepts_gpt_image_25_model(self) -> None:
        """模型校验应放行 gpt-image-2.5，随后才因无可用账号报错。"""
        request = ConversationRequest(prompt="一张测试图", model="gpt-image-2.5")
        with mock.patch.object(
            openai_v1_models.account_service,
            "get_available_access_token",
            side_effect=RuntimeError("no available image quota"),
        ):
            with self.assertRaises(ImageGenerationError) as ctx:
                list(stream_image_outputs_with_pool(request))
        self.assertNotIn("unsupported image model", str(ctx.exception))

        unsupported = ConversationRequest(prompt="一张测试图", model="gpt-image-9.9")
        with self.assertRaises(ImageGenerationError) as ctx:
            list(stream_image_outputs_with_pool(unsupported))
        self.assertIn("unsupported image model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

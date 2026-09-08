"""Client initialization and credential management for Sarvam AI SDK."""
from __future__ import annotations

import logging
from typing import Optional

from backend.app.core.config import Settings, get_settings
from backend.app.services.meeting_pipeline.sarvam.config import SarvamConfig
from backend.app.services.meeting_pipeline.sarvam.exceptions import (
    SarvamAPIKeyMissingError,
    SarvamClientError,
)

logger = logging.getLogger(__name__)


class SarvamClientManager:
    """Manages the lifecycle and initialization of the official SarvamAI client."""

    def __init__(
        self,
        config: Optional[SarvamConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()

        if config is None:
            self.config = SarvamConfig(
                api_key=self.settings.sarvam_api_key,
                model=self.settings.sarvam_model,
                language_code=self.settings.sarvam_language_code,
                mode=self.settings.sarvam_mode,
                with_diarization=self.settings.sarvam_with_diarization,
                poll_interval_seconds=self.settings.sarvam_poll_interval_seconds,
                max_wait_seconds=self.settings.sarvam_max_wait_seconds,
            )
        else:
            self.config = config

        self._client = None

    def get_client(self):
        """Return the initialized SarvamAI SDK client.
        
        Validates API key and initializes SarvamAI instance.
        """
        if self._client is not None:
            return self._client

        if not self.config.api_key:
            raise SarvamAPIKeyMissingError(
                "Sarvam API key is not configured. Please set SARVAM_API_KEY in your environment or .env."
            )

        api_key_str = self.config.api_key.get_secret_value().strip()
        if not api_key_str:
            raise SarvamAPIKeyMissingError(
                "Sarvam API key is empty. Please provide a valid SARVAM_API_KEY."
            )

        try:
            from sarvamai import SarvamAI

            # Official SDK accepts api_subscription_key
            self._client = SarvamAI(api_subscription_key=api_key_str)
            logger.info("Initialized SarvamAI SDK client successfully")
            return self._client
        except Exception as exc:
            raise SarvamClientError(f"Failed to initialize SarvamAI client: {exc}") from exc

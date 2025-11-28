"""Gemini native API client with structured output support.

This module provides a client for Google's Gemini API that guarantees
structured JSON output conforming to Pydantic schemas.

Usage:
    from xhs_seo_optimizer.utils.gemini_client import GeminiStructuredClient
    from xhs_seo_optimizer.models.reports import SuccessProfileReport

    client = GeminiStructuredClient()
    report = client.generate_structured(
        prompt="Analyze competitor notes...",
        output_model=SuccessProfileReport
    )
"""

from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Type, TypeVar, Optional
import os

T = TypeVar('T', bound=BaseModel)


class GeminiStructuredClient:
    """Gemini client with native structured output support.

    This client uses Gemini's native `response_mime_type` and `response_schema`
    to guarantee JSON output that conforms to a Pydantic model schema.

    Attributes:
        api_key: Google API key for Gemini
        model: Model name (default: gemini-2.5-flash)
        client: Underlying genai.Client instance
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash"
    ):
        """Initialize Gemini client.

        Args:
            api_key: Google API key. If None, uses GOOGLE_API_KEY env var.
            model: Model name (default: gemini-2.5-flash)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY env var or pass api_key param."
            )
        self.model = model
        self.client = genai.Client(api_key=self.api_key)

    def generate_structured(
        self,
        prompt: str,
        output_model: Type[T],
        temperature: float = 0.1,
        max_output_tokens: int = 8192
    ) -> T:
        """Generate structured output conforming to Pydantic model.

        Uses Gemini's native structured output feature to guarantee
        valid JSON conforming to the provided schema.

        Args:
            prompt: The prompt to send to the model
            output_model: Pydantic model class defining the output schema
            temperature: Generation temperature (default 0.1 for consistency)
            max_output_tokens: Maximum output tokens (default 8192)

        Returns:
            Validated Pydantic model instance

        Raises:
            ValueError: If response cannot be parsed as valid JSON
            ValidationError: If JSON doesn't conform to schema
        """
        # Get JSON schema from Pydantic model
        json_schema = output_model.model_json_schema()

        # Create generation config with structured output
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=json_schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        # Generate content
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        # Parse and validate response
        return output_model.model_validate_json(response.text)

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 4096
    ) -> str:
        """Generate plain text response (no structured output).

        Use this for intermediate reasoning tasks where JSON is not required.

        Args:
            prompt: The prompt to send to the model
            temperature: Generation temperature (default 0.7)
            max_output_tokens: Maximum output tokens (default 4096)

        Returns:
            Generated text string
        """
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        return response.text

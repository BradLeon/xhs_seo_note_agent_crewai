"""Image Generator Tool - 图像生成工具.

Generates images using Gemini model via OpenRouter API.
Supports reference image input for style/subject consistency.
"""

import os
import json
import base64
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import OpenAI
from crewai.tools import BaseTool
from pydantic import Field, ConfigDict, BaseModel


class ImageGeneratorInput(BaseModel):
    """Input schema for ImageGeneratorTool."""

    prompt: str = Field(
        description="图片生成描述 (详细的中文描述，50-500字符)"
    )
    reference_image_url: Optional[str] = Field(
        default=None,
        description="参考图URL (可选，用于保持风格/主体一致性)"
    )
    must_preserve_elements: Optional[List[str]] = Field(
        default=None,
        description="必须包含的元素列表 (e.g., ['产品外观', '品牌logo'])"
    )
    image_type: str = Field(
        default="cover",
        description="图片类型: cover | inner_1 | inner_2 | ..."
    )
    style_reference: Optional[str] = Field(
        default=None,
        description="风格参考描述 (e.g., '小红书爆款育儿笔记风格')"
    )


class ImageGeneratorTool(BaseTool):
    """图像生成工具 (Image Generator Tool).

    Generates images using Google Gemini model via OpenRouter API.
    Supports:
    - Text-to-image generation with detailed Chinese prompts
    - Reference image input for style/subject consistency
    - Must-preserve elements enforcement
    - Automatic saving to local files

    Output:
    - success: bool
    - image_url: Base64 data URL or error message
    - local_path: Path to saved image file
    - error: Error message if failed
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "image_generator"
    description: str = (
        "根据prompt生成图片，支持参考图输入以保持主体一致性。"
        "输出包括Base64图片URL和本地保存路径。"
        "使用场景：为优化后的笔记生成封面图和内页图。"
    )
    args_schema: type[BaseModel] = ImageGeneratorInput

    # OpenRouter configuration
    api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    model: str = Field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_IMAGE_MODEL",
            "google/gemini-2.0-flash-exp:free"  # Free tier for image generation
        )
    )

    # Output directory
    output_dir: str = Field(default="outputs/images")

    def _run(
        self,
        prompt: str,
        reference_image_url: Optional[str] = None,
        must_preserve_elements: Optional[List[str]] = None,
        image_type: str = "cover",
        style_reference: Optional[str] = None
    ) -> str:
        """Generate image based on prompt and optional reference.

        Args:
            prompt: Image generation description
            reference_image_url: Optional reference image for consistency
            must_preserve_elements: Elements that must appear in generated image
            image_type: Type of image (cover, inner_1, etc.)
            style_reference: Style description for the image

        Returns:
            JSON string with generation results
        """
        # Validate API key
        if not self.api_key:
            return json.dumps({
                "success": False,
                "image_type": image_type,
                "error": "OPENROUTER_API_KEY not configured",
                "prompt_used": prompt
            }, ensure_ascii=False)

        # Build enhanced prompt
        full_prompt = self._build_full_prompt(
            prompt=prompt,
            must_preserve_elements=must_preserve_elements,
            style_reference=style_reference
        )

        try:
            # Generate image via OpenRouter
            result = self._generate_image(
                prompt=full_prompt,
                reference_image_url=reference_image_url
            )

            if result.get("success") and result.get("image_data"):
                # Save to local file
                local_path = self._save_image(
                    image_data=result["image_data"],
                    image_type=image_type
                )
                result["local_path"] = local_path

            result["image_type"] = image_type
            result["prompt_used"] = full_prompt
            result["reference_image_used"] = reference_image_url

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "image_type": image_type,
                "error": str(e),
                "prompt_used": full_prompt,
                "reference_image_used": reference_image_url
            }, ensure_ascii=False)

    def _build_full_prompt(
        self,
        prompt: str,
        must_preserve_elements: Optional[List[str]] = None,
        style_reference: Optional[str] = None
    ) -> str:
        """Build enhanced prompt with constraints.

        Args:
            prompt: Base prompt
            must_preserve_elements: Elements to preserve
            style_reference: Style reference

        Returns:
            Enhanced prompt string
        """
        parts = [prompt]

        if style_reference:
            parts.append(f"\n\n风格参考: {style_reference}")

        if must_preserve_elements:
            elements_str = "、".join(must_preserve_elements)
            parts.append(f"\n\n必须包含以下元素: {elements_str}")

        # Add XHS-specific styling hints
        parts.append("\n\n图片要求: 适合小红书平台，高清、明亮、有质感，比例适合手机屏幕浏览。")

        return "".join(parts)

    def _generate_image(
        self,
        prompt: str,
        reference_image_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Call OpenRouter API to generate image.

        Args:
            prompt: Image generation prompt
            reference_image_url: Optional reference image

        Returns:
            Dict with success status and image data
        """
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

        # Build messages
        messages = []

        if reference_image_url:
            # Multi-modal message with reference image
            content = [
                {
                    "type": "text",
                    "text": f"请参考这张图片的风格和主体，生成一张新图片。\n\n生成要求:\n{prompt}"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": reference_image_url}
                }
            ]
            messages.append({"role": "user", "content": content})
        else:
            # Text-only message
            messages.append({
                "role": "user",
                "content": f"请生成一张图片:\n\n{prompt}"
            })

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"modalities": ["image", "text"]}
            )

            # Extract image from response
            message = response.choices[0].message

            # Check for images in response
            # The response format may vary - handle different structures
            if hasattr(message, 'images') and message.images:
                # Direct images attribute
                for image in message.images:
                    image_url = image.get('image_url', {}).get('url', '')
                    if image_url:
                        return {
                            "success": True,
                            "image_url": image_url,
                            "image_data": image_url  # Base64 data URL
                        }

            # Check content for image blocks
            if hasattr(message, 'content'):
                content = message.content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get('type') == 'image_url':
                                url = block.get('image_url', {}).get('url', '')
                                if url:
                                    return {
                                        "success": True,
                                        "image_url": url,
                                        "image_data": url
                                    }
                            elif block.get('type') == 'image':
                                # Some models return base64 directly
                                data = block.get('data', '')
                                if data:
                                    return {
                                        "success": True,
                                        "image_url": f"data:image/png;base64,{data}",
                                        "image_data": f"data:image/png;base64,{data}"
                                    }

            # If no image found, return text response as fallback info
            text_content = message.content if isinstance(message.content, str) else str(message.content)
            return {
                "success": False,
                "error": f"No image generated. Model response: {text_content[:200]}",
                "model_response": text_content
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"API call failed: {str(e)}"
            }

    def _save_image(self, image_data: str, image_type: str) -> str:
        """Save base64 image data to local file.

        Args:
            image_data: Base64 data URL (data:image/png;base64,...)
            image_type: Type of image for filename

        Returns:
            Path to saved file
        """
        # Create output directory
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(image_data[:100].encode()).hexdigest()[:8]
        filename = f"{image_type}_{timestamp}_{hash_suffix}.png"
        filepath = output_path / filename

        # Extract base64 data
        if image_data.startswith("data:"):
            # Remove data URL prefix
            base64_data = image_data.split(",", 1)[1] if "," in image_data else image_data
        else:
            base64_data = image_data

        # Decode and save
        try:
            image_bytes = base64.b64decode(base64_data)
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            return str(filepath)
        except Exception as e:
            print(f"⚠️ Failed to save image: {e}")
            return ""

    def generate_multiple(
        self,
        prompts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate multiple images from a list of prompts.

        Convenience method for batch generation.

        Args:
            prompts: List of prompt configs, each with:
                - prompt: str
                - image_type: str
                - reference_image_url: Optional[str]
                - must_preserve_elements: Optional[List[str]]
                - style_reference: Optional[str]

        Returns:
            List of generation results
        """
        results = []
        for config in prompts:
            result_json = self._run(
                prompt=config.get("prompt", ""),
                reference_image_url=config.get("reference_image_url"),
                must_preserve_elements=config.get("must_preserve_elements"),
                image_type=config.get("image_type", "inner"),
                style_reference=config.get("style_reference")
            )
            results.append(json.loads(result_json))
        return results

"""
Provider-aware API client with retry logic and error handling.

Supports:
- Azure OpenAI chat completions + image edits
- Standard OpenAI chat completions + image edits
"""

import base64
import io
import json
import time
from typing import Any, Dict, List, Optional

import requests

from config.config import AzureConfig, ProcessingConfig


class VisionClient:
    """Client for vision-capable chat completions."""

    def __init__(self, config: AzureConfig, processing_config: ProcessingConfig):
        self.config = config
        self.processing_config = processing_config

        if config.is_azure:
            self.endpoint = (
                f"{config.vision_endpoint}/openai/deployments/"
                f"{config.vision_deployment}/chat/completions"
                f"?api-version={config.vision_api_version}"
            )
        else:
            self.endpoint = f"{config.openai_base_url}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.is_azure:
            headers["api-key"] = self.config.vision_api_key
        else:
            headers["Authorization"] = f"Bearer {self.config.vision_api_key}"
        return headers

    def _payload(
        self,
        image_data_urls: List[str],
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        messages = [{"role": "system", "content": system_prompt}]

        user_content = [{"type": "text", "text": user_prompt}]
        for data_url in image_data_urls:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })
        messages.append({"role": "user", "content": user_content})

        payload: Dict[str, Any] = {
            "messages": messages,
            "max_completion_tokens": self.processing_config.max_tokens,
        }

        if self.config.is_openai:
            payload["model"] = self.config.vision_model

        if response_format:
            payload["response_format"] = response_format

        return payload

    def analyze_images(
        self,
        image_data_urls: List[str],
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze images with retries."""
        payload = self._payload(image_data_urls, system_prompt, user_prompt, response_format)

        for attempt in range(self.processing_config.max_retries):
            try:
                print(
                    f"[Vision API:{self.config.provider}] Sending request to analyze "
                    f"{len(image_data_urls)} image(s)... (timeout: 120s)"
                )
                start_time = time.time()

                response = requests.post(
                    self.endpoint,
                    headers=self._headers(),
                    json=payload,
                    timeout=120,
                )

                elapsed = time.time() - start_time
                print(f"[Vision API:{self.config.provider}] Response: {response.status_code} (took {elapsed:.1f}s)")

                if response.status_code == 200:
                    return response.json()

                print(
                    f"Vision API error (attempt {attempt + 1}): "
                    f"{response.status_code} - {response.text}"
                )

                if response.status_code == 429:
                    time.sleep(self.processing_config.rate_limit_delay * (attempt + 1))
                elif response.status_code >= 500:
                    time.sleep(2 ** attempt)
                else:
                    raise Exception(f"Client error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as exc:
                print(f"Request error (attempt {attempt + 1}): {exc}")
                if attempt < self.processing_config.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise Exception(f"Vision API failed after {self.processing_config.max_retries} attempts")

    def get_augmentation_suggestions(
        self,
        image_data_urls: List[str],
        domain: str = "general object detection",
    ) -> Dict[str, List[Dict[str, str]]]:
        """Get augmentation suggestions for images."""
        system_prompt = (
            f"You are an expert in data augmentation for {domain}.\n"
            "Your task is to analyze images and suggest realistic augmentations "
            "that would improve model robustness."
        )

        user_prompt = f"""Analyze each image and suggest 2-3 realistic augmentations for {domain} that would create valuable training variations.

Focus on:
1. Environmental variations (lighting, weather, seasons, time of day)
2. Camera variations (angles, quality, different camera types)
3. Natural challenging conditions (fog, rain, shadows, glare)
4. Edge cases that improve model robustness

IMPORTANT: Return ONLY a JSON object with this EXACT structure (no additional nesting):
{{
  "image_0": [
    {{"prompt": "specific detailed augmentation prompt", "category": "environmental"}},
    {{"prompt": "another specific prompt", "category": "camera"}},
    {{"prompt": "third specific prompt", "category": "edge_case"}}
  ],
  "image_1": [
    {{"prompt": "specific detailed augmentation prompt", "category": "environmental"}},
    {{"prompt": "another specific prompt", "category": "camera"}},
    {{"prompt": "third specific prompt", "category": "edge_case"}}
  ]
}}

Each prompt must be:
- Specific and detailed (describe exact changes to make)
- Realistic and achievable via image editing
- Focused on one clear augmentation
- 10-30 words describing the transformation

Provide suggestions for ALL images (image_0, image_1, image_2, etc.)"""

        response = self.analyze_images(
            image_data_urls,
            system_prompt,
            user_prompt,
            response_format={"type": "json_object"},
        )

        try:
            content = response["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )

            print(f"[Vision API:{self.config.provider}] Raw response content: {content[:500]}...")

            if not content or content.strip() == "":
                print(f"[Vision API:{self.config.provider}] ERROR: Empty content received")
                print(f"[Vision API:{self.config.provider}] Full response: {json.dumps(response, indent=2)}")
                raise Exception("Empty response content from Vision API")

            parsed = json.loads(content)
            print(f"[Vision API:{self.config.provider}] Parsed JSON keys: {list(parsed.keys())}")
        except (KeyError, IndexError) as exc:
            print(f"[Vision API:{self.config.provider}] ERROR: Unexpected response structure: {exc}")
            print(f"[Vision API:{self.config.provider}] Full response: {json.dumps(response, indent=2)}")
            raise
        except json.JSONDecodeError as exc:
            print(f"[Vision API:{self.config.provider}] ERROR: JSON decode error: {exc}")
            print(f"[Vision API:{self.config.provider}] Content that failed to parse: {repr(content)}")
            print(f"[Vision API:{self.config.provider}] Content length: {len(content) if content else 0}")
            raise

        suggestions = parsed["suggestions"] if "suggestions" in parsed else parsed

        print(f"[Vision API:{self.config.provider}] Number of image keys in suggestions: {len(suggestions)}")
        if suggestions:
            first_key = list(suggestions.keys())[0]
            print(f"[Vision API:{self.config.provider}] First image has {len(suggestions[first_key])} suggestions")

        return suggestions


class ImageEditClient:
    """Client for image editing APIs."""

    def __init__(self, config: AzureConfig, processing_config: ProcessingConfig):
        self.config = config
        self.processing_config = processing_config

        if config.is_azure:
            self.endpoint = (
                f"{config.image_edit_endpoint}/openai/deployments/"
                f"{config.image_edit_deployment}/images/edits"
                f"?api-version={config.image_edit_api_version}"
            )
        else:
            self.endpoint = f"{config.openai_base_url}/images/edits"

    def _headers(self) -> Dict[str, str]:
        if self.config.is_azure:
            return {"api-key": self.config.image_edit_api_key}
        return {"Authorization": f"Bearer {self.config.image_edit_api_key}"}

    def _extract_image_bytes(self, result: Dict[str, Any]) -> bytes:
        data = result.get("data") or []
        if not data:
            raise Exception("No image data in response")

        image_item = data[0]
        image_b64 = image_item.get("b64_json")
        image_url = image_item.get("url")

        if image_b64:
            return base64.b64decode(image_b64)

        if image_url:
            if image_url.startswith("data:"):
                return base64.b64decode(image_url.split(",", 1)[1])

            image_response = requests.get(image_url, timeout=120)
            image_response.raise_for_status()
            return image_response.content

        raise Exception("No supported image payload in response")

    def edit_image(
        self,
        image_data_url: str,
        prompt: str,
        mask: Optional[str] = None,
    ) -> bytes:
        """Edit an image with retries."""
        print(f"[ImageEditClient:{self.config.provider}] Preparing to edit image with prompt: '{prompt[:80]}...'")

        if "," in image_data_url:
            image_b64 = image_data_url.split(",", 1)[1]
        else:
            image_b64 = image_data_url
        image_bytes = base64.b64decode(image_b64)
        print(f"[ImageEditClient:{self.config.provider}] Decoded image: {len(image_bytes)} bytes")

        mask_bytes = None
        if mask:
            mask_b64 = mask.split(",", 1)[1] if "," in mask else mask
            mask_bytes = base64.b64decode(mask_b64)

        data = {"prompt": prompt}
        if self.config.is_openai:
            data["model"] = self.config.image_edit_model

        for attempt in range(self.processing_config.max_retries):
            try:
                print(f"[Image Edit:{self.config.provider}] Sending edit request: '{prompt[:60]}...' (timeout: 120s)")

                files = {
                    "image": ("image.png", io.BytesIO(image_bytes), "image/png")
                }
                if mask_bytes:
                    files["mask"] = ("mask.png", io.BytesIO(mask_bytes), "image/png")

                start_time = time.time()
                response = requests.post(
                    self.endpoint,
                    headers=self._headers(),
                    files=files,
                    data=data,
                    timeout=120,
                )
                elapsed = time.time() - start_time
                print(f"[Image Edit:{self.config.provider}] Response: {response.status_code} (took {elapsed:.1f}s)")

                if response.status_code == 200:
                    return self._extract_image_bytes(response.json())

                print(
                    f"Image Edit API error (attempt {attempt + 1}): "
                    f"{response.status_code} - {response.text}"
                )

                if response.status_code == 429:
                    time.sleep(self.processing_config.rate_limit_delay * (attempt + 1))
                elif response.status_code >= 500:
                    time.sleep(2 ** attempt)
                else:
                    raise Exception(f"Client error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as exc:
                print(f"Request error (attempt {attempt + 1}): {exc}")
                if attempt < self.processing_config.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise Exception(f"Image Edit API failed after {self.processing_config.max_retries} attempts")


class APIClientFactory:
    """Factory for creating provider-aware API clients."""

    @staticmethod
    def create_vision_client(
        api_config: AzureConfig,
        processing_config: ProcessingConfig,
    ) -> VisionClient:
        return VisionClient(api_config, processing_config)

    @staticmethod
    def create_image_edit_client(
        api_config: AzureConfig,
        processing_config: ProcessingConfig,
    ) -> ImageEditClient:
        return ImageEditClient(api_config, processing_config)


# Backward-compatible aliases for existing imports.
AzureVisionClient = VisionClient
AzureImageEditClient = ImageEditClient
AzureClientFactory = APIClientFactory

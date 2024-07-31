import base64
from pathlib import Path

import aiohttp

# Constants
API_URL = "http://192.168.1.100:5002"
INTERROGATE_ENDPOINT = f"{API_URL}/sdapi/v1/interrogate"


class Connector:
    """Class for generating and analyzing images using stable diffusion with session management"""

    def __init__(self):
        self.session = None

    async def initialize_session(self) -> None:
        """Initialize the aiohttp Client Session if not already initialized"""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def cleanup_session(self) -> None:
        """Close the aiohttp Client Session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def make_post_request(self, endpoint: str, data: dict = None, headers: dict = None) -> dict:
        """Makes an HTTP POST request to the specified endpoint"""
        await self.initialize_session()
        try:
            async with self.session.post(url=endpoint, json=data, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"An aiohttp error has occurred: {e}")
            raise

    async def interrogate_clip(self, image_path: Path) -> str:
        """Returns an image description using CLIP"""
        encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode()
        data = {"image": encoded_image, "model": "clip"}
        response_dict = await self.make_post_request(INTERROGATE_ENDPOINT, data=data)
        return response_dict['caption']

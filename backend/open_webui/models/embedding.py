from langchain.embeddings import Embeddings
import openai
import numpy as np
from typing import List
import aiohttp

class AsyncVLLMEmbeddings(Embeddings):
    def __init__(self, api_base: str, api_key: str = "Dummy", model: str = "dragonkue/BGE-m3-ko"):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """비동기적으로 여러 텍스트를 임베딩."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "input": texts,
                "model": self.model
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            async with session.post(
                f"{self.api_base}/embeddings",
                json=payload,
                headers=headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"API call failed: {await response.text()}")
                result = await response.json()
                return [data["embedding"] for data in result["data"]]

    async def aembed_query(self, text: str) -> List[float]:
        """비동기적으로 단일 쿼리 텍스트를 임베딩."""
        return (await self.aembed_documents([text]))[0]
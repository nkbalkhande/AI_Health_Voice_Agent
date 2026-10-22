from abc import ABC, abstractmethod
from typing import Any


class EvaluationRunner(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def run(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        pass

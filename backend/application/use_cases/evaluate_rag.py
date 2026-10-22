from typing import Any

from backend.domain.interfaces.evaluation_runner import EvaluationRunner


class EvaluateRagUseCase:
    def __init__(self, *, runner: EvaluationRunner) -> None:
        self._runner = runner

    async def execute(
        self,
        dataset: list[dict[str, Any]],
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._runner.run(dataset, config=config)

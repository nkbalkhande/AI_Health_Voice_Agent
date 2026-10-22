from __future__ import annotations

import json
from pathlib import Path

from backend.application.use_cases.evaluate_rag_pipeline import (
    EvalCase,
    EvaluateRagPipelineUseCase,
)
from backend.core.config import settings
from backend.core.job_store import jobs
from backend.core.logger import logger
from datetime import datetime


async def run_evaluation_job(
    job_id: str,
    use_case: EvaluateRagPipelineUseCase,
    cases: list[EvalCase],
) -> None:
    try:
        jobs[job_id]["status"] = "RUNNING"
        logger.info("Evaluation started: job_id=%s", job_id)
        result = await use_case.execute(cases, run_id=job_id)
        # Sep 2 2026: judge traces are a side channel (see RagasRunner). Pull
        # them out before the result is stored/served so the report keeps its
        # shape; they go to a *_traces.jsonl sidecar next to the artifact.
        judge_traces = result.pop("judge_traces", None)
        jobs[job_id]["status"] = "COMPLETED"
        jobs[job_id]["result"] = result
        # R-53: results used to live only in an in-process dict, so a restart
        # after a multi-hour judge run discarded them and every circulating
        # result file was a hand-saved HTTP response. Write the full result -
        # manifest included - to evaluation.artifacts_dir.
        artifact_path: Path | None = None
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(settings.evaluation_artifacts_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = out_dir / f"eval_{job_id}-{timestamp}.json"
            artifact_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            jobs[job_id]["artifact_path"] = str(artifact_path)
            if judge_traces:
                traces_path = artifact_path.with_name(
                    artifact_path.stem + "_traces.jsonl"
                )
                with traces_path.open("w", encoding="utf-8") as fh:
                    for entry in judge_traces:
                        fh.write(json.dumps(entry, ensure_ascii=False, default=str))
                        fh.write("\n")
                jobs[job_id]["traces_path"] = str(traces_path)
        except OSError:
            logger.exception(
                "Could not persist evaluation artifact: job_id=%s", job_id
            )
        logger.info(
            "Evaluation completed: job_id=%s (artifact: %s)", job_id, artifact_path
        )
    except Exception as exc:
        logger.exception("Evaluation failed: job_id=%s", job_id)
        jobs[job_id]["status"] = "FAILED"
        jobs[job_id]["error"] = str(exc)

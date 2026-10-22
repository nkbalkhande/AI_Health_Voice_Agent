import uuid

from fastapi import APIRouter, BackgroundTasks, Depends

from backend.api.deps import get_evaluate_pipeline_use_case
from backend.api.schemas import (
    EvaluationJobCreatedResponse,
    EvaluationJobNotFoundResponse,
    EvaluationJobResultResponse,
    EvaluationJobStatusResponse,
    RagasEvalRequest,
    RagasEvalResponse,
)
from backend.application.evaluation_background import run_evaluation_job
from backend.application.use_cases.evaluate_rag_pipeline import (
    EvalCase,
    EvaluateRagPipelineUseCase,
)
from backend.core.job_store import jobs
from backend.core.logger import logger

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/evaluate", response_model=EvaluationJobCreatedResponse)
async def run_evaluation(
    body: RagasEvalRequest,
    background_tasks: BackgroundTasks,
    use_case: EvaluateRagPipelineUseCase = Depends(get_evaluate_pipeline_use_case),
) -> EvaluationJobCreatedResponse:
    """Start RAG pipeline + RAGAS evaluation in the background.

    Returns a ``job_id`` immediately. Poll ``GET /evaluate/{job_id}/status`` and
    fetch results from ``GET /evaluate/{job_id}/result`` when complete.
    """
    logger.info("Evaluation request received")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "PENDING",
        "result": None,
        "error": None,
    }
    logger.info("Job created: job_id=%s", job_id)

    cases = [
        EvalCase(question=case.question, ground_truth=case.ground_truth or "")
        for case in body.cases
    ]

    background_tasks.add_task(
        run_evaluation_job,
        job_id,
        use_case,
        cases,
    )
    logger.info("Background task started: job_id=%s", job_id)

    return EvaluationJobCreatedResponse(job_id=job_id, status="PENDING")


@router.get(
    "/evaluate/{job_id}/status",
    response_model=EvaluationJobStatusResponse | EvaluationJobNotFoundResponse,
)
async def get_evaluation_status(
    job_id: str,
) -> EvaluationJobStatusResponse | EvaluationJobNotFoundResponse:
    job = jobs.get(job_id)
    if not job:
        return EvaluationJobNotFoundResponse(error="Job not found")

    return EvaluationJobStatusResponse(job_id=job_id, status=job["status"])


@router.get(
    "/evaluate/{job_id}/result",
    response_model=EvaluationJobResultResponse | EvaluationJobNotFoundResponse,
)
async def get_evaluation_result(
    job_id: str,
) -> EvaluationJobResultResponse | EvaluationJobNotFoundResponse:
    job = jobs.get(job_id)
    if not job:
        return EvaluationJobNotFoundResponse(error="Job not found")

    return EvaluationJobResultResponse(
        status=job["status"],
        result=job.get("result"),
        error=job.get("error"),
    )

@router.post("/evaluate-rag", response_model=RagasEvalResponse)
async def run_evaluation_sync(
    body: RagasEvalRequest,
    use_case: EvaluateRagPipelineUseCase = Depends(get_evaluate_pipeline_use_case),
) -> RagasEvalResponse:
    """Run the RAG pipeline for each question, then score it with ragas.

    Provide only ``question`` (+ optional ``ground_truth``). The pipeline
    retrieves, reranks and generates the answer, and ragas returns the metrics.
    """
    cases = [
        EvalCase(question=case.question, ground_truth=case.ground_truth or "")
        for case in body.cases
    ]
    results = await use_case.execute(cases)
    # R-10: carry the same provenance the background-job path returns. Files
    # saved off this endpoint used to have no manifest at all, which is how
    # unmanifested result files entered circulation.
    return RagasEvalResponse(
        runner=results.get("runner", "ragas"),
        status=results.get("status", "unknown"),
        samples=results.get("samples", 0),
        metrics=results.get("metrics", {}),
        per_sample=results.get("per_sample", []),
        contexts=results.get("contexts", []),
        manifest=results.get("manifest", {}),
        metrics_coverage=results.get("metrics_coverage", {}),
        headline_metrics=results.get("headline_metrics", {}),
        diagnostic_metrics=results.get("diagnostic_metrics", {}),
        evaluated_samples=results.get("evaluated_samples"),
        skipped_samples=results.get("skipped_samples"),
        judge_model=results.get("judge_model"),
        embedding_model=results.get("embedding_model"),
    )

"""HTTP API: `POST /ask`, `GET /health`. Run with `make serve`."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from filingsqa import __version__
from filingsqa.config import get_settings
from filingsqa.llm.base import LLMError
from filingsqa.rag.pipeline import RAGPipeline
from filingsqa.rag.schemas import AskRequest, AskResponse

log = logging.getLogger(__name__)

DISCLAIMER = "Educational project. Not investment advice."


def _default_factory() -> RAGPipeline:
    from filingsqa.rag.factory import build_pipeline

    return build_pipeline(get_settings())


def create_app(pipeline_factory: Callable[[], RAGPipeline] = _default_factory) -> FastAPI:
    """The pipeline is injected so tests can run the API with fakes and no network."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = pipeline_factory()
        yield

    app = FastAPI(title="FilingsQA", version=__version__, description=DISCLAIMER, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        pipeline: RAGPipeline = request.app.state.pipeline
        try:
            return pipeline.ask(body)
        except LLMError as exc:
            log.exception("LLM failure")
            raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc

    return app

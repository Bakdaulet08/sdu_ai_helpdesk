"""HTTP interface for the existing retrieval/generation service."""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .moderation import ModerationService
from .audit import RequestJournal
from .generation import AnswerService, ChatClient, GenerationError

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    question: str = Field(min_length=1, max_length=2000)
    language: Literal['ru', 'kk', 'en'] = 'ru'

    @field_validator('question')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Question must not be blank')
        return value.strip()


class AnswerResponse(BaseModel):
    request_id: str
    status: Literal['answered', 'fallback']
    answer: str = ""
    forum_answer: str | None
    ai_explanation: str
    language: Literal['ru', 'kk', 'en']
    basis: Literal['retrieved', 'context', 'general', 'none']
    reason: str | None
    source: dict | None
    source_type: str | None
    retrieval_score: float | None


class ErrorResponse(BaseModel):
    request_id: str
    status: Literal['error'] = 'error'
    error: str


def build_service():
    from .embedding import E5
    from .retrieval import postgres_retriever, validate_threshold
    url = os.getenv('KB_DATABASE_URL', '').strip()
    if not url:
        raise ValueError('Set KB_DATABASE_URL in ai_ml/.env')
    client = ChatClient.from_env()
    context_path = Path(os.getenv('KB_CONTEXT_FILE', 'config/university_context.txt'))
    if not context_path.is_absolute():
        context_path = ROOT / context_path
    context = context_path.read_text(encoding='utf-8')
    if len(context) > 20000:
        raise ValueError('University context exceeds 20000 characters')
    threshold = validate_threshold(os.getenv('KB_MIN_SIMILARITY', '0.80'))
    encoder = E5(os.getenv('KB_DEVICE', 'cpu'), int(os.getenv('KB_BATCH_SIZE', '16')))
    return AnswerService(postgres_retriever(url, encoder, threshold, os.getenv('KB_DATASET', 'faq')), client, context)


class ModerationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    text: str = Field(min_length=1, max_length=10000)

    @field_validator('text')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Text must not be blank')
        return value


class ModerationResponse(BaseModel):
    request_id: str
    decision: Literal['allow', 'block', 'review']
    publish_allowed: bool
    categories: list[Literal['harassment', 'hate', 'threat', 'sexual_harassment']]
    reason: str
    policy_version: str


class ModerationError(BaseModel):
    request_id: str
    decision: Literal['review'] = 'review'
    publish_allowed: bool = False
    error: str


def create_app(service=None, journal=None, moderator=None):
    # Dependency injection keeps API tests independent of model downloads and paid APIs.
    @asynccontextmanager
    async def lifespan(app):
        from dotenv import load_dotenv
        load_dotenv(ROOT / '.env')
        log_path = Path(os.getenv('KB_REQUEST_LOG', 'data/logs/requests.sqlite3'))
        if not log_path.is_absolute():
            log_path = ROOT / log_path
        app.state.journal = journal if journal is not None else RequestJournal(log_path)
        app.state.service = service if service is not None else build_service()
        yield
        del app.state.service

    app = FastAPI(title='SDU AI Helpdesk', version='0.1.0', lifespan=lifespan)
    gate = Lock()
    moderation_gate = Lock()
    moderation_service = moderator if moderator is not None else ModerationService()

    @app.post('/generate-answer', response_model=AnswerResponse,
              responses={502: {'model': ErrorResponse}, 503: {'model': ErrorResponse}})
    def generate(payload: AnswerRequest):
        request_id = str(uuid4())
        started = perf_counter()

        def failure(code):
            return {'request_id': request_id, 'status': 'error', 'error': code}

        try:
            app.state.journal.start(request_id, payload.question, payload.language)
        except Exception:
            LOGGER.error('Request journal unavailable: %s', request_id)
            return JSONResponse(failure('logging_unavailable'), status_code=503)

        if not gate.acquire(blocking=False):
            result, status = failure('service_busy'), 503
        else:
            try:
                raw = app.state.service.answer(payload.question, payload.language)
                result = AnswerResponse(request_id=request_id, **raw).model_dump()
                status = 200
            except GenerationError as exc:
                result = failure(exc.code)
                status = 503 if exc.code in {'llm_overloaded', 'llm_rate_limited'} else 502
            except Exception:
                # Do not expose DB URLs, credentials or exception bodies to clients/logs.
                LOGGER.error('Answer pipeline failed: %s', request_id)
                result, status = failure('pipeline_unavailable'), 503
            finally:
                gate.release()
        try:
            app.state.journal.finish(request_id, result, status, round((perf_counter()-started)*1000))
        except Exception:
            LOGGER.error('Request journal update failed: %s', request_id)
            return JSONResponse(failure('logging_unavailable'), status_code=503)
        return JSONResponse(result, status_code=status)

    @app.post('/moderate', response_model=ModerationResponse,
              responses={502: {'model': ModerationError}, 503: {'model': ModerationError}})
    def moderate(payload: ModerationRequest):
        request_id = str(uuid4())

        def failure(code, status):
            return JSONResponse(ModerationError(request_id=request_id, error=code).model_dump(), status_code=status)

        if not moderation_gate.acquire(blocking=False):
            return failure('service_busy', 503)
        try:
            result = ModerationResponse(request_id=request_id, **moderation_service.moderate(payload.text))
            LOGGER.info('Moderation %s: %s', request_id, result.decision)
            return result
        except GenerationError as exc:
            return failure(exc.code, 503 if exc.code in {'llm_overloaded', 'llm_rate_limited'} else 502)
        except Exception:
            LOGGER.error('Moderation failed: %s', request_id)
            return failure('moderation_unavailable', 503)
        finally:
            moderation_gate.release()

    return app


app = create_app()

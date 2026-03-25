from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Dict, Optional

from app.api.cache import get_redis
from app.api.config import SESSION_TTL_SECONDS


@dataclass
class SessionState:
    session_id: str
    program: Optional[str] = None
    study_year: Optional[str] = None
    pending_slot: Optional[str] = None
    pending_intent: Optional[str] = None
    active_topic: Optional[str] = None
    last_effective_question: Optional[str] = None
    last_user_question: Optional[str] = None
    last_intent: Optional[str] = None
    turn_count: int = 0
    updated_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def clone(self) -> "SessionState":
        return SessionState(**deepcopy(self.to_dict()))


async def load_session_state(session_id: str) -> SessionState:
    r = get_redis()
    raw = await r.get(_key(session_id))
    if not raw:
        return SessionState(session_id=session_id)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return SessionState(session_id=session_id)

    payload["session_id"] = session_id
    payload.setdefault("metadata", {})
    return SessionState(**payload)


async def save_session_state(state: SessionState) -> None:
    r = get_redis()
    state.updated_at = _utc_now()
    await r.set(_key(state.session_id), json.dumps(state.to_dict()), ex=int(SESSION_TTL_SECONDS))


async def clear_session_state(session_id: str) -> None:
    r = get_redis()
    await r.delete(_key(session_id))


async def touch_session_state(session_id: str) -> None:
    r = get_redis()
    await r.expire(_key(session_id), int(SESSION_TTL_SECONDS))


async def session_exists(session_id: str) -> bool:
    r = get_redis()
    return bool(await r.exists(_key(session_id)))


async def session_ttl(session_id: str) -> int:
    r = get_redis()
    return int(await r.ttl(_key(session_id)))


def _key(session_id: str) -> str:
    return f"session:{session_id}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

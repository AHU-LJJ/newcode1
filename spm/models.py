from __future__ import annotations
import hashlib
import json
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def norm_entity(value):
    if not value:
        return None
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w+# -]", " ", value).split()) or None


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Message(Model):
    role: Literal["user", "assistant"]
    content: str


class Episode(Model):
    id: str
    order: int = Field(ge=0)
    time: str | None = None
    messages: list[Message]


class Condition(Model):
    entity: str | None = None
    audience: str | None = None
    formality: Literal["casual", "neutral", "formal"] | None = None
    task_type: str | None = None
    channel: str | None = None
    context: str | None = None

    @field_validator("entity")
    @classmethod
    def normalize(cls, value):
        return norm_entity(value)

    def is_global(self):
        return not any(self.model_dump().values())


class Evidence(Model):
    episode_id: str
    turn_idx: int = Field(ge=0)
    span: str = Field(min_length=1)
    expression: Literal["explicit", "incidental", "choice", "feedback", "event"]
    time: str | None = None


class RawUnit(Model):
    content: str = Field(min_length=1, max_length=600)
    polarity: Literal["prefer", "avoid", "require"]
    scope: str = Field(min_length=1, max_length=600)
    domain: str = Field(min_length=1, max_length=80)
    nature: Literal["constraint", "taste", "style"]
    expression: Literal["explicit", "incidental", "choice", "feedback", "event"]
    condition: Condition = Field(default_factory=Condition)
    span: str = Field(min_length=1)
    turn_idx: int = Field(ge=0)


class Extraction(Model):
    units: list[RawUnit] = Field(default_factory=list, max_length=24)


class PreferenceFields(Model):
    """Semantic fields only; the host binds evidence from an allowed source."""
    content: str = Field(min_length=1, max_length=600)
    polarity: Literal["prefer", "avoid", "require"]
    scope: str = Field(min_length=1, max_length=600)
    domain: str = Field(min_length=1, max_length=80)
    nature: Literal["constraint", "taste", "style"]
    condition: Condition = Field(default_factory=Condition)


class SourceObservation(PreferenceFields):
    source_id: int = Field(ge=0, strict=True)
    expression: Literal["explicit", "incidental", "choice", "feedback", "event"]


class SourceExtraction(Model):
    units: list[SourceObservation] = Field(default_factory=list, max_length=24)


class IndexedGate(Model):
    class Item(Model):
        index: int = Field(ge=0, strict=True)
        p_apply: float = Field(ge=0, le=1, allow_inf_nan=False)
        reason: str = Field(min_length=1, max_length=500)
    decisions: list[Item]


class IndexedRelations(Model):
    class Item(Model):
        index: int = Field(ge=0, strict=True)
        relation: Literal['same','contradicts','different_context','compatible','unrelated']
    relations: list[Item]


class ScalarGate(Model):
    """One host-bound candidate: the model never has to copy an identifier."""
    p_apply: float = Field(ge=0, le=1, allow_inf_nan=False)
    reason: str = Field(min_length=1, max_length=500)


class UpdateChecks(Model):
    """Evidence check for potentially destructive graph operations."""
    class Item(Model):
        index: int = Field(ge=0, strict=True)
        same_target: bool = Field(strict=True)
        same_facet: bool = Field(strict=True)
        equivalent: bool = Field(strict=True)
        mutually_exclusive: bool = Field(strict=True)
        explicit_revision: bool = Field(strict=True)
        revision_source: int | None = Field(default=None, ge=0, strict=True)
        revision_quote: str | None = Field(default=None, min_length=1, max_length=200)
        reason: str = Field(min_length=1, max_length=300)
    checks: list[Item]


class PrefUnit(Model):
    id: str
    content: str
    polarity: Literal["prefer", "avoid", "require"]
    scope: str
    domain: str
    nature: Literal["constraint", "taste", "style"]
    condition: Condition = Field(default_factory=Condition)
    evidence: list[Evidence]
    confidence: float = Field(ge=0, le=1)
    n_support: int = Field(default=1, ge=1)
    n_contra: int = Field(default=0, ge=0)
    status: Literal["candidate", "active", "pending", "superseded"] = "active"
    first_order: int = 0
    last_order: int = 0
    valid_to: str | None = None
    superseded_by: str | None = None
    challenges: list[str] = Field(default_factory=list)

    def embed_text(self):
        return f"{self.content}. Applies to: {self.scope}"

    def key(self):
        return fingerprint([self.content.casefold().strip(), self.polarity, self.scope.casefold().strip(),
                            self.nature, self.condition.model_dump()])


class ParsedQuery(Model):
    situation: Condition = Field(default_factory=Condition)
    entities: list[str] = Field(default_factory=list, max_length=12)
    probes: list[str] = Field(default_factory=list, max_length=6)


class Relations(Model):
    class Item(Model):
        unit_id: str
        relation: Literal["same", "contradicts", "different_context", "compatible", "unrelated"]
    relations: list[Item]


class BatchRelations(Model):
    class Item(Model):
        new_id: str
        unit_id: str
        relation: Literal['same','contradicts','different_context','compatible','unrelated']
    relations: list[Item]


class GateScores(Model):
    class Item(Model):
        unit_id: str
        p_apply: float = Field(ge=0, le=1, allow_inf_nan=False)
        reason: str = Field(min_length=1, max_length=500)
    decisions: list[Item]


class Decision(Model):
    unit_id: str
    decision: Literal["apply", "suppress", "uncertain"]
    p_apply: float = Field(ge=0, le=1)
    reason: str


class PreferenceInference(Model):
    preference: str
    checklist: list[str] = Field(max_length=20)


class Selection(Model):
    labels: list[Literal["A", "B", "C", "D", "E", "F", "G", "H"]] = Field(min_length=1, max_length=8)

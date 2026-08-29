from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReviewTimeRange(BaseModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec must be greater than start_sec")
        return self


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    generation_id: str = Field(min_length=1, max_length=100)
    feedback: str = Field(min_length=1, max_length=8000)
    time_range: ReviewTimeRange | None = None
    target_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class Observation(BaseModel):
    text: str = Field(min_length=1)
    start_sec: float | None = Field(default=None, ge=0)
    end_sec: float | None = Field(default=None, ge=0)


class Localization(BaseModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self):
        if self.end_sec < self.start_sec:
            raise ValueError("localization end_sec must not precede start_sec")
        return self


class CriticResult(BaseModel):
    issue_type: Literal[
        "appearance",
        "identity_consistency",
        "motion_timing",
        "motion_quality",
        "pose",
        "object_consistency",
        "camera_motion",
        "shot_timing",
        "action_order",
        "spatial_relation",
        "audio_sync",
        "dialogue",
        "overall_style",
        "composition",
        "other",
    ]
    issue_confirmed: bool
    confidence: float = Field(ge=0, le=1)
    localization: Localization | None = None
    observations: list[Observation] = Field(default_factory=list)
    inferences: list[str] = Field(default_factory=list)
    root_cause: Literal[
        "PROMPT_UNDERSPECIFIED",
        "PROMPT_AMBIGUOUS",
        "PROMPT_CONFLICT",
        "PROMPT_TIMING_INSUFFICIENT",
        "MODEL_STOCHASTICITY",
        "MODEL_CAPABILITY_LIMIT",
        "REFERENCE_CONFLICT",
        "INSUFFICIENT_EVIDENCE",
    ]
    reason: str = Field(min_length=1)


class PatchOperation(BaseModel):
    op: Literal["replace"]
    path: str = Field(min_length=1)
    value: str = Field(min_length=1)


class PatchPlan(BaseModel):
    action: Literal[
        "PATCH_PROMPT",
        "REGENERATE_SAME_PROMPT",
        "CHANGE_REFERENCE",
        "MANUAL_REVIEW",
        "UNRESOLVED",
    ]
    reason: str = Field(min_length=1)
    operations: list[PatchOperation] = Field(default_factory=list)
    changes: list[str] = Field(default_factory=list)
    preserved: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_operations(self):
        if self.action == "PATCH_PROMPT" and not self.operations:
            raise ValueError("PATCH_PROMPT requires at least one operation")
        if self.action != "PATCH_PROMPT" and self.operations:
            raise ValueError("Only PATCH_PROMPT may contain operations")
        return self

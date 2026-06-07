"""Finn agent state definitions.

State flows through LangGraph nodes. Each node reads and writes these fields.

Extract phase data models: intent / requirements / hard constraints / soft constraints.
Pydantic models + ExtractResult aggregate with validation:
- is_sufficient() — data complete enough to enter plan phase
- missing_fields() — list of fields still needing clarification
- clarification_question() — generate next question by priority
- apply_update() — incremental merge for multi-turn extraction
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from langgraph.graph import MessagesState


def _merge_bookings(
    left: dict[str, "BookingResult"],
    right: dict[str, "BookingResult"],
) -> dict[str, "BookingResult"]:
    """Merge two booking dicts — right-side keys take precedence."""
    merged = dict(left)
    merged.update(right)
    return merged


# ── Enums ──────────────────────────────────────────────────


# ── Legacy: kept for memory/profile backward compatibility ──────


class PartyMember(BaseModel):
    """A person or category of people in the party.

    Used by the memory system for stored companions.
    New extract phase uses GroupProfile.tags instead for POI filtering.
    """

    role: str = ""
    age: int | None = None
    constraints: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)


# ── Enums ──────────────────────────────────────────────────


class SceneType(StrEnum):
    """出行场景类型 — only family and friends."""
    FAMILY = "family"
    FRIENDS = "friends"


class BudgetPreference(StrEnum):
    """预算偏好"""
    ECONOMY = "economy"
    MID = "mid"
    LUXURY = "luxury"


class TravelPace(StrEnum):
    """出行节奏"""
    RELAXED = "relaxed"
    BALANCED = "balanced"
    FAST = "fast"


class TransportMode(StrEnum):
    """交通方式"""
    WALK = "walk"
    TRANSIT = "transit"
    DRIVE = "drive"


# ── Extract Phase: Sub-models ──────────────────────────────


class UserIntent(BaseModel):
    """用户出行意图 —— 从对话中提取的核心参数。

    字段逐一对应下游求解器输入，null 表示尚未提取。
    """

    activity_summary: str | None = Field(
        default=None, description="一句话总结用户想做什么，如'喝咖啡然后看电影'"
    )
    city: str | None = Field(default=None, description="目标城市")
    plan_date: str | None = Field(default=None, description="出行日期 YYYY-MM-DD")
    time_window_start: str | None = Field(default=None, description="开始时间 HH:MM")
    time_window_hours: float | None = Field(
        default=None, description="可用时长（小时）", ge=0
    )
    guest_count: int | None = Field(default=None, description="参与人数", ge=1)
    scene: SceneType | None = Field(default=None, description="场景类型")
    raw_utterance: str | None = Field(
        default=None, description="用户原始自然语言输入"
    )


class UserRequirements(BaseModel):
    """用户明确提出的需求——应被满足但非硬约束。

    为空的 list 表示用户未提出此类需求。
    """

    must_visit_pois: list[str] = Field(
        default_factory=list, description="用户指定的必去地点名称"
    )
    must_have_cuisine: list[str] = Field(
        default_factory=list, description="必须包含的菜系"
    )
    must_have_activity_type: list[str] = Field(
        default_factory=list, description="必须包含的活动类型"
    )
    special_requests: list[str] = Field(
        default_factory=list, description="特殊需求（生日蛋糕、鲜花、纪念日等）"
    )
    notes: str | None = Field(default=None, description="其他自由文本备注")


class HardConstraints(BaseModel):
    """硬约束 —— 数学意义上的约束条件，不可协商放松。

    求解器用这些字段做 feasibility check，违反则直接 infeasible。
    """

    budget_max_cny: float | None = Field(
        default=None, description="预算上限（元）", ge=0
    )
    dietary_restrictions: list[str] = Field(
        default_factory=list, description="饮食限制（清真、素食、过敏原等）"
    )
    child_age: int | None = Field(
        default=None, description="儿童年龄（影响亲子友好筛选）", ge=0
    )
    accessibility_needed: bool = Field(default=False, description="无障碍需求")
    time_deadline: str | None = Field(
        default=None, description="必须在此时间前结束 HH:MM"
    )
    must_include_poi_ids: list[str] = Field(
        default_factory=list, description="必须包含的 POI ID 列表"
    )


class SoftConstraints(BaseModel):
    """软约束/偏好 —— 可协商放松。

    求解无解时按以下优先级放松:
    1. preferred_poi_types（放宽类型偏好）
    2. travel_pace（放宽节奏限制）
    3. budget_preference（放宽预算偏好）
    4. max_transit_minutes（放宽转场时间）
    """

    budget_preference: BudgetPreference | None = Field(
        default=None, description="预算偏好"
    )
    travel_pace: TravelPace | None = Field(default=None, description="出行节奏")
    preferred_poi_types: list[str] = Field(
        default_factory=list, description="偏好 POI 类型"
    )
    preferred_cuisines: list[str] = Field(
        default_factory=list, description="偏好菜系"
    )
    preferred_transport: TransportMode | None = Field(
        default=None, description="偏好交通方式"
    )
    max_transit_minutes: float | None = Field(
        default=None, description="单程最大转场时间（分钟）"
    )
    avoid_poi_types: list[str] = Field(
        default_factory=list, description="不想去的类型"
    )


class GroupProfile(BaseModel):
    """同行人群画像 —— 结构化标签供下游 POI 过滤使用。

    tags 是连接意图提取与 POI 搜索的核心桥梁：
    - 人群标签: child_5yo, elderly_70s, friends_4_mix
    - 饮食标签: diet_low_calorie, diet_halal, diet_vegan
    - 约束标签: child_safe, wheelchair_accessible, non_smoking
    - 偏好标签: indoor_if_rain, quiet, photo_friendly, pet_friendly
    """

    type: SceneType = Field(default=SceneType.FAMILY, description="场景类型")
    tags: list[str] = Field(
        default_factory=list,
        description="结构化人群标签，供下游 POI 过滤使用",
    )
    hard_constraints: list[str] = Field(
        default_factory=list,
        description="群组硬约束，如 child_safe, wheelchair_accessible, non_smoking",
    )
    soft_preferences: list[str] = Field(
        default_factory=list,
        description="群组软偏好，如 indoor_if_rain, quiet, photo_friendly",
    )


class TimeWindow(BaseModel):
    """时间窗口约束。"""

    date: str | None = Field(default=None, description="YYYY-MM-DD")
    start_time: str | None = Field(default=None, description="HH:MM 出发时间")
    end_time: str | None = Field(default=None, description="HH:MM 推断结束时间")
    budget_hours: dict = Field(
        default_factory=lambda: {"min": 4, "max": 6},
        description="可用时长范围",
    )
    flexibility: Literal["strict", "normal", "high"] = "normal"

    @field_validator("date", "start_time", "end_time", mode="before")
    @classmethod
    def _coerce_none_to_empty(cls, v):
        """Coerce None to empty string for downstream compatibility."""
        return v if v is not None else ""

    @model_validator(mode="after")
    def _validate_time_order(self):
        """Ensure end_time > start_time when both are set."""
        st = self.start_time
        et = self.end_time
        if st and et and st != "" and et != "":
            try:
                st_m = int(st.split(":")[0]) * 60 + int(st.split(":")[1])
                et_m = int(et.split(":")[0]) * 60 + int(et.split(":")[1])
                if et_m <= st_m:
                    # Auto-fix: set end_time 8h after start
                    end_m = st_m + 480  # 8h default
                    end_h = end_m // 60
                    end_min = end_m % 60
                    object.__setattr__(self, "end_time", f"{end_h:02d}:{end_min:02d}")
            except (ValueError, IndexError):
                pass
        return self


class GeoConstraint(BaseModel):
    """空间约束 —— 搜索范围与交通容忍度。"""

    center_address: str = Field(default="", description="出发地址或区域")
    center_location: str | None = Field(
        default=None, description="经纬度 lng,lat（后续地理编码填充）"
    )
    radius_m: int = Field(
        default=5000, ge=1000, description="搜索半径，单位米"
    )
    max_transit_time_min: int = Field(
        default=30, description="单程最大转场时间（分钟）"
    )
    constraint_desc: str = Field(
        default="", description="原始距离描述，如'别太远'"
    )

    @field_validator("radius_m", mode="before")
    @classmethod
    def _coerce_radius(cls, v):
        """Coerce None to default; clamp to [1000, 100000]."""
        if v is None:
            return 5000
        return max(1000, min(100000, int(v)))

    @field_validator("center_location", mode="after")
    @classmethod
    def _validate_location(cls, v):
        """Validate location is 'lng,lat' format."""
        if v is None:
            return v
        if isinstance(v, str) and v.strip():
            parts = v.strip().split(",")
            if len(parts) != 2:
                return v  # Don't break on malformed, downstream handles it
            try:
                float(parts[0].strip())
                float(parts[1].strip())
            except (ValueError, TypeError):
                pass  # Non-numeric, leave as-is
        return v


class ChainTemplate(BaseModel):
    """活动链模板 —— 出行节奏与节点数量约束。"""

    template: list[Literal["play", "eat", "follow_up", "rest", "lunch", "dinner"]] = Field(
        default_factory=lambda: ["play", "eat", "follow_up"]
    )
    min_nodes: int = 2
    max_nodes: int = 4
    preferred_activity_types: list[str] = Field(
        default_factory=list,
        description="活动类型偏好，如 amusement_park, museum, hotpot, cafe",
    )


# ── Constraint Profile ─────────────────────────────────────


class ConstraintProfile(BaseModel):
    """Structured constraint extraction — bridge between clarify and planner.

    Extracted by LLM in clarify_intent and used by poi_search (filter + score),
    planner (insertion cost weights), and verify_and_adjust (checking).
    """

    # Hard constraints — POI pool filtering
    budget_max_cny: float | None = Field(default=None, description="总预算上限（元）")
    dietary_restrictions: list[str] = Field(
        default_factory=list, description="饮食限制（清真、素食、过敏原等）"
    )
    child_age: int | None = Field(default=None, description="儿童年龄", ge=0)
    accessibility_needed: bool = Field(default=False, description="无障碍需求")
    time_deadline: str | None = Field(default=None, description="必须在此时间前结束 HH:MM")
    must_visit_poi_names: list[str] = Field(
        default_factory=list, description="必须访问的 POI 名称列表"
    )
    must_have_cuisine: list[str] = Field(
        default_factory=list, description="必须包含的菜系"
    )

    # Soft constraints — scoring weights
    preferred_poi_types: list[str] = Field(
        default_factory=list, description="偏好 POI 类型"
    )
    preferred_cuisines: list[str] = Field(
        default_factory=list, description="偏好菜系"
    )
    preferred_transport: str | None = Field(
        default=None, description="偏好交通方式 walk/transit/drive"
    )
    max_transit_minutes: float | None = Field(
        default=None, description="单程最大转场时间（分钟）"
    )
    avoid_poi_types: list[str] = Field(
        default_factory=list, description="不想去的 POI 类型"
    )
    budget_preference: str | None = Field(
        default=None, description="预算偏好 economy/mid/luxury"
    )
    travel_pace: str | None = Field(
        default=None, description="出行节奏 relaxed/balanced/fast"
    )

    # Strategy weights (α, β, γ, δ) — computed by compute_strategy_weights()
    alpha: float = Field(default=1.0, description="transit cost weight")
    beta: float = Field(default=1.0, description="wait/idle cost weight")
    gamma: float = Field(default=1.0, description="match mismatch cost weight")
    delta: float = Field(default=1.0, description="distance-from-center cost weight")

    # Derived meal slot assignments
    meal_slots: list[dict] = Field(
        default_factory=list,
        description="用餐时段分配，如 [{slot:'lunch', window:'11:00-13:00', casual:true}, ...]",
    )

    # Multi-user parallel activity detection
    parallel_activities: list[dict] = Field(
        default_factory=list,
        description=(
            "并行活动列表，如 [{group:'kids', slot:'play', activity_hint:'儿童乐园', "
            "cluster_required:true}, {group:'adults', slot:'play', activity_hint:'购物'}]"
        ),
    )

    # Distance preference
    distance_preference: str | None = Field(
        default=None,
        description="距离偏好描述，如 'prefer_optimal_range'/'nearby_only'/'no_limit'",
    )
    preferred_transit_range_min: int | None = Field(
        default=None, description="偏好转场时间下限（分钟）"
    )
    preferred_transit_range_max: int | None = Field(
        default=None, description="偏好转场时间上限（分钟）"
    )


# ── Extract Phase: Aggregate ───────────────────────────────


class ExtractResult(BaseModel):
    """Extract 阶段的完整输出 —— 意图 + 需求 + 硬约束 + 软约束。

    核心校验逻辑:
    - is_sufficient()  → True == 数据完备，可流转至 poi_search 节点
    - missing_fields() → 缺失字段列表（用于引导 LLM 澄清）
    - clarification_question() → 按优先级生成下一轮提问
    - apply_update() → 增量合并新提取的字段（多轮对话）
    """

    intent: UserIntent = Field(default_factory=UserIntent)
    requirements: UserRequirements = Field(default_factory=UserRequirements)
    hard_constraints: HardConstraints = Field(default_factory=HardConstraints)
    soft_constraints: SoftConstraints = Field(default_factory=SoftConstraints)
    group: GroupProfile = Field(default_factory=GroupProfile)
    time: TimeWindow = Field(default_factory=TimeWindow)
    geo: GeoConstraint = Field(default_factory=GeoConstraint)
    chain: ChainTemplate = Field(default_factory=ChainTemplate)
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="提取置信度 0-1"
    )
    constraint_profile: "ConstraintProfile | None" = Field(
        default=None, description="结构化约束画像，连接 clarify 与 planner"
    )
    follow_up_question: str | None = Field(
        default=None, description="下一轮自然语言追问，完备时为 null"
    )

    # ── 必要字段 —— 缺一不可 ──
    REQUIRED_INTENT_FIELDS: tuple[str, ...] = (
        "activity_summary",
        "city",
        "plan_date",
        "time_window_start",
        "time_window_hours",
        "guest_count",
    )
    REQUIRED_HARD_FIELDS: tuple[str, ...] = ()

    def is_sufficient(self) -> bool:
        """判定 extract 阶段是否完成。

        规则:
        1. intent 必要字段必须全部非空
        2. budget 由 LLM 按画像推断 + 常识兜底，不作为阻塞字段
        3. requirements / soft_constraints 全可选——空 = 无偏好
        """
        for field in self.REQUIRED_INTENT_FIELDS:
            if getattr(self.intent, field, None) is None:
                return False
        for field in self.REQUIRED_HARD_FIELDS:
            if getattr(self.hard_constraints, field, None) is None:
                return False
        return True

    def missing_fields(self) -> list[str]:
        """返回当前缺失的必要字段列表（dotted path 形式）。"""
        missing: list[str] = []
        for field in self.REQUIRED_INTENT_FIELDS:
            if getattr(self.intent, field, None) is None:
                missing.append(f"intent.{field}")
        for field in self.REQUIRED_HARD_FIELDS:
            if getattr(self.hard_constraints, field, None) is None:
                missing.append(f"hard_constraints.{field}")
        return missing

    def clarification_question(self) -> str | None:
        """根据缺失字段生成下一轮澄清问题。

        按优先级返回最关键缺失字段对应的问题。
        全部完备时返回 None。
        """
        missing = self.missing_fields()
        if not missing:
            return None

        _questions: dict[str, str] = {
            "intent.activity_summary": "请问您想做什么？比如喝咖啡、看电影、吃火锅？",
            "intent.city": "请问您想去哪个城市？",
            "intent.plan_date": "请问您计划哪天出行？",
            "intent.time_window_start": "请问您计划几点开始？",
            "intent.time_window_hours": "请问您有多少时间可用？",
            "intent.guest_count": "请问一共几个人？",
        }
        for m in missing:
            if m in _questions:
                return _questions[m]
        return f"还需要了解: {', '.join(missing)}"

    def apply_update(self, update: "UpdateExtractResultInput") -> list[str]:
        """增量合并更新字段。只更新非 None 值，返回被更新的字段名列表。

        不会覆盖用户已确认的非 None 值——新值仅在新值与旧值不同时写入。
        这是"增量合并"语义：LLM 每次只传本次对话新提取的字段。
        """
        updated: list[str] = []

        if update.intent is not None:
            intent_updates = update.intent.model_dump(exclude_none=True)
            for k, v in intent_updates.items():
                current = getattr(self.intent, k, None)
                if current is None or current != v:
                    setattr(self.intent, k, v)
                    updated.append(f"intent.{k}")

        if update.requirements is not None:
            req_updates = update.requirements.model_dump(exclude_none=True)
            for k, v in req_updates.items():
                current = getattr(self.requirements, k, None)
                if isinstance(v, list) and isinstance(current, list):
                    merged = list(dict.fromkeys(current + v))
                    if merged != current:
                        setattr(self.requirements, k, merged)
                        updated.append(f"requirements.{k}")
                elif current is None or current != v:
                    setattr(self.requirements, k, v)
                    updated.append(f"requirements.{k}")

        if update.hard_constraints is not None:
            hc_updates = update.hard_constraints.model_dump(exclude_none=True)
            for k, v in hc_updates.items():
                current = getattr(self.hard_constraints, k, None)
                if isinstance(v, list) and isinstance(current, list):
                    merged = list(dict.fromkeys(current + v))
                    if merged != current:
                        setattr(self.hard_constraints, k, merged)
                        updated.append(f"hard_constraints.{k}")
                elif current is None or current != v:
                    setattr(self.hard_constraints, k, v)
                    updated.append(f"hard_constraints.{k}")

        if update.soft_constraints is not None:
            sc_updates = update.soft_constraints.model_dump(exclude_none=True)
            for k, v in sc_updates.items():
                current = getattr(self.soft_constraints, k, None)
                if isinstance(v, list) and isinstance(current, list):
                    merged = list(dict.fromkeys(current + v))
                    if merged != current:
                        setattr(self.soft_constraints, k, merged)
                        updated.append(f"soft_constraints.{k}")
                elif current is None or current != v:
                    setattr(self.soft_constraints, k, v)
                    updated.append(f"soft_constraints.{k}")

        if update.group is not None:
            group_updates = update.group.model_dump(exclude_none=True)
            for k, v in group_updates.items():
                current = getattr(self.group, k, None)
                if isinstance(v, list) and isinstance(current, list):
                    merged = list(dict.fromkeys(current + v))
                    if merged != current:
                        setattr(self.group, k, merged)
                        updated.append(f"group.{k}")
                elif current is None or current != v:
                    setattr(self.group, k, v)
                    updated.append(f"group.{k}")

        if update.time is not None:
            time_updates = update.time.model_dump(exclude_none=True)
            for k, v in time_updates.items():
                current = getattr(self.time, k, None)
                if current is None or current != v:
                    setattr(self.time, k, v)
                    updated.append(f"time.{k}")

        if update.geo is not None:
            geo_updates = update.geo.model_dump(exclude_none=True)
            for k, v in geo_updates.items():
                current = getattr(self.geo, k, None)
                if current is None or current != v:
                    setattr(self.geo, k, v)
                    updated.append(f"geo.{k}")

        if update.chain is not None:
            chain_updates = update.chain.model_dump(exclude_none=True)
            for k, v in chain_updates.items():
                current = getattr(self.chain, k, None)
                if current is None or current != v:
                    setattr(self.chain, k, v)
                    updated.append(f"chain.{k}")

        if update.constraint_profile is not None:
            cp_updates = update.constraint_profile.model_dump(exclude_none=True)
            if self.constraint_profile is None:
                self.constraint_profile = ConstraintProfile(**cp_updates)
                updated.append("constraint_profile")
            else:
                # Fields that are list[dict] — replace, don't merge
                _LIST_OF_DICT_FIELDS = {"meal_slots", "parallel_activities"}
                for k, v in cp_updates.items():
                    current = getattr(self.constraint_profile, k, None)
                    if k in _LIST_OF_DICT_FIELDS:
                        # list[dict]: LLM always sends complete list, just replace
                        if current != v:
                            setattr(self.constraint_profile, k, v)
                            updated.append(f"constraint_profile.{k}")
                    elif isinstance(v, list) and isinstance(current, list):
                        merged = list(dict.fromkeys(current + v))
                        if merged != current:
                            setattr(self.constraint_profile, k, merged)
                            updated.append(f"constraint_profile.{k}")
                    elif current is None or current != v:
                        setattr(self.constraint_profile, k, v)
                        updated.append(f"constraint_profile.{k}")

        if update.confidence is not None and update.confidence != self.confidence:
            self.confidence = update.confidence
            updated.append("confidence")

        self.follow_up_question = update.follow_up_question

        return updated


class UpdateExtractResultInput(BaseModel):
    """update_extract_result 工具输入 —— 增量更新 ExtractResult。

    LLM 每次只传本次对话新提取的字段。
    工具 handler 负责增量合并到已有 ExtractResult。
    """

    intent: UserIntent | None = Field(
        default=None, description="意图字段增量（只传本次新提取的字段）"
    )
    requirements: UserRequirements | None = Field(
        default=None, description="需求字段增量（list 会 merge 非覆盖）"
    )
    hard_constraints: HardConstraints | None = Field(
        default=None, description="硬约束字段增量（list 会 merge）"
    )
    soft_constraints: SoftConstraints | None = Field(
        default=None, description="软约束字段增量（list 会 merge）"
    )
    group: GroupProfile | None = Field(
        default=None, description="人群画像增量（list 会 merge）"
    )
    time: TimeWindow | None = Field(default=None, description="时间窗口增量")
    geo: GeoConstraint | None = Field(default=None, description="空间约束增量")
    chain: ChainTemplate | None = Field(default=None, description="活动链增量")
    constraint_profile: ConstraintProfile | None = Field(
        default=None, description="约束画像增量"
    )
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="当前提取置信度 0-1"
    )
    follow_up_question: str | None = Field(
        default=None, description="下一轮追问，完备时为 null"
    )


# ── POI Search Phase ───────────────────────────────────────


class POICandidate(BaseModel):
    """A single POI from search results."""

    id: str = ""
    name: str = ""
    address: str = ""
    location: str = ""  # "lng,lat"
    type: str = ""  # POI type code from Amap
    rating: float | None = None
    distance_m: int | None = None
    price_level: str | None = None
    tags: list[str] = Field(default_factory=list)
    source: str = ""  # which search strategy produced this candidate

    # ── Enhanced fields (populated by poi_search detail fetch) ──
    open_time: str | None = None       # "09:00"
    close_time: str | None = None      # "22:00"
    stay_min: int = 60                 # minimum stay minutes
    stay_base: int = 120               # default stay minutes
    stay_max: int = 240                # maximum before diminishing returns
    elastic_coef: float = 1.0          # elasticity for budget reflow (0=rigid, 2=elastic)
    match_score: float = 0.0           # composite constraint match (0-1, by poi_search)
    price_per_person: float | None = None  # from biz_ext.cost

    # ── Macro category ──
    macro_category: str = ""  # "餐饮"|"景点"|"文化"|"购物"|"休闲娱乐"|"运动"|""

    # ── Sub-user assignment (for multi-user parallel activities) ──
    assigned_user_group: str = ""  # "adults"|"kids"|"elderly"|"" (empty = shared)


class POISearchResult(BaseModel):
    """Aggregated POI search results with cache metadata."""

    candidates: list[POICandidate] = Field(default_factory=list)
    cache_key: str = ""
    strategy_used: str = ""
    from_cache: bool = False


# ── Plan Phase ─────────────────────────────────────────────


class SubTask(BaseModel):
    """A single unit of work within a plan."""

    id: str
    type: Literal["search", "compare", "book"]
    target: str
    dependencies: list[str] = Field(default_factory=list)
    params: dict = Field(default_factory=dict)
    compensatory: str | None = None


class Plan(BaseModel):
    """Decomposed plan: a DAG of sub-tasks with estimated cost."""

    sub_tasks: list[SubTask] = Field(default_factory=list)
    total_cost_estimate: float | None = None
    notes: str = ""


class Verification(BaseModel):
    """Result of plan verification."""

    score: float
    issues: list[str] = Field(default_factory=list)
    status: Literal["pass", "fix", "reject"] = "fix"


# ── Execution Phase ────────────────────────────────────────


class BookingResult(BaseModel):
    """Outcome of a single booking sub-task."""

    task_id: str
    status: Literal["pending", "success", "failed", "cancelled", "compensated"]
    order_id: str | None = None
    error: str | None = None
    error_type: Literal["transient", "recoverable", "fatal"] | None = None
    retries: int = 0


# ═══════════════════════════════════════════════════════════════════════
# New models — Planner/Checker/Presentation
# ═══════════════════════════════════════════════════════════════════════


class WeatherContext(BaseModel):
    """Weather for the planning date, fetched by context_agent."""
    date: str = ""
    condition: str = ""             # 晴 / 多云 / 雨 / 雪
    temp_high: float | None = None  # Celsius
    temp_low: float | None = None
    wind: str = ""
    indoor_recommended: bool = False


class PlanCard(BaseModel):
    """Pre-assembled display card for one stop in the itinerary."""
    step: int = 0
    poi_id: str = ""
    poi_name: str = ""
    activity_type: str = ""       # play / eat / transit
    start_time: str = ""          # HH:MM
    end_time: str = ""            # HH:MM
    detail_line: str = ""         # "评分4.5 | 人均80元 | 09:00-22:00"
    address: str = ""
    transport_from_prev: str = "" # "驾车15分钟 (12km)"
    amap_nav_url: str = ""        # Amap navigation deep link
    taxi_estimate: str = ""       # "打车约12元"


class CheckerIssue(BaseModel):
    """Issue found by rule-based checker."""
    level: Literal["L1_swap", "L2_compress", "L3_degrade"] = "L3_degrade"
    description: str = ""
    suggested_action: str = ""
    affected_poi_ids: list[str] = Field(default_factory=list)


# ── Planner Algorithm Models ───────────────────────────────


class TimeAlloc(BaseModel):
    """Time allocation for a single activity slot, computed by forward/backward pass."""

    slot: str = ""              # play/lunch/dinner/eat/follow_up
    duration_min: int = 120     # allocated minutes (after elastic reflow)
    earliest_start: str = ""    # HH:MM from forward pass
    earliest_end: str = ""
    latest_start: str = ""      # HH:MM from backward pass
    latest_end: str = ""
    slack_min: int = 0          # total float (latest_start - earliest_start)


class ActivityNode(BaseModel):
    """A single stop in the planned itinerary with its time allocation."""

    slot: str = ""                           # play/lunch/dinner/eat/follow_up
    poi: POICandidate | None = None          # selected POI
    time: TimeAlloc | None = None            # assigned time window
    transit_from_prev_min: int = 0           # minutes from previous stop
    transit_distance_m: int = 0
    transport_mode: str = "drive"            # walk/transit/drive (auto-selected)
    stay_duration_min: int = 120
    cost_estimate: float = 0.0


class PlannedPath(BaseModel):
    """A complete planned itinerary — output of one strategy."""

    nodes: list[ActivityNode] = Field(default_factory=list)
    total_transit_min: int = 0
    total_cost: float = 0.0
    coverage_score: float = 0.0          # composite score (0-1)
    insertion_cost_total: float = 0.0    # sum of insertion costs
    notes: str = ""
    strategy: str = ""                   # "time_critical" | "geo_cluster" | "match_driven"


class StrategyResult(BaseModel):
    """Result from a single strategy — for asyncio.gather collection."""

    path: PlannedPath | None = None
    strategy: str = ""
    score: float = 0.0
    error: str | None = None


# ── New: Search Strategy & Category Pools ───────────────────


class CategorySearch(BaseModel):
    """LLM-formulated search parameters for one macro-category."""

    category: str = ""  # "餐饮" | "购物" | "文化" | "景点" | "休闲娱乐" | "运动" | "户外" | "室内"
    keywords: list[str] = Field(default_factory=list)
    search_type: Literal["around", "text", "both"] = "both"
    target_count: int = 10
    priority: float = 1.0  # 0-1 importance weight


class SearchStrategy(BaseModel):
    """Overall search strategy formulated by LLM from user constraints."""

    categories: list[CategorySearch] = Field(default_factory=list)
    radius_m: int = 5000
    reasoning: str = ""
    round_number: int = 1
    coverage_sufficient: bool = False


class POICategoryPool(BaseModel):
    """POI candidates grouped by macro-category."""

    category: str = ""
    candidates: list[POICandidate] = Field(default_factory=list)
    coverage_score: float = 0.0


# ── New: Multi-Agent Plan & Fusion ──────────────────────────


class AgentPlan(BaseModel):
    """One agent's complete itinerary plan with its strategy."""

    agent_name: str = ""  # "time_efficient" | "cost_optimal" | "experience_rich" | "balanced"
    strategy_desc: str = ""
    nodes: list[ActivityNode] = Field(default_factory=list)
    total_transit_min: int = 0
    total_cost: float = 0.0
    coverage_score: float = 0.0
    reasoning: str = ""


class FusionResult(BaseModel):
    """Weighted voting fusion result across multiple agent plans."""

    plan: Plan | None = None  # final fused plan in SubTask format
    path: PlannedPath | None = None  # final fused path with timing
    votes: dict[str, int] = Field(default_factory=dict)  # agent_name → vote count
    fusion_score: float = 0.0
    reasoning: str = ""


# ── Agent State ────────────────────────────────────────────


class AgentState(MessagesState):
    """State carried through every LangGraph node.

    Inherits ``messages`` with ``add_messages`` reducer from ``MessagesState``.
    """

    # IP-based user location (injected by CLI, used as POI search center)
    user_coords: str = ""  # "lng,lat"

    # Extract phase
    extract_result: "ExtractResult | None" = None
    clarify_iterations: int = 0

    # POI search phase (legacy — kept for backward compat; new path uses category_pools)
    poi_candidates: list[POICandidate] = Field(default_factory=list)
    distance_matrix: dict[str, int] = Field(default_factory=dict)
    # ^ key="poi_a_id|poi_b_id", value=distance_m

    # New search phase
    search_strategy: SearchStrategy | None = None
    search_round: int = 0
    category_pools: dict[str, POICategoryPool] = Field(default_factory=dict)
    # ^ key=category name ("餐饮","购物","文化",...), value=pool

    # Weather (fetched by context_agent)
    weather: WeatherContext | None = None

    # Plan phase
    plan: Plan | None = None
    verification: Verification | None = None
    plan_iterations: int = 0
    modify_feedback: str = ""
    plan_cards: list[PlanCard] = Field(default_factory=list)
    checker_issues: list[CheckerIssue] = Field(default_factory=list)
    planned_paths: list[PlannedPath] = Field(default_factory=list)
    selected_path: PlannedPath | None = None

    # New multi-agent plan + fusion
    agent_plans: list[AgentPlan] = Field(default_factory=list)
    fusion_result: FusionResult | None = None
    modify_count: int = 0  # guard against infinite modify→qa_check loops

    # Execution phase
    next_action: str = ""
    bookings: Annotated[
        dict[str, BookingResult], _merge_bookings
    ] = Field(default_factory=dict)
    execution_status: Literal[
        "idle", "running", "partial", "done", "failed", "compensated"
    ] = "idle"
    retry_count: int = 0

    # Fan-out context (set via Send.arg for book_worker)
    current_task_id: str = ""
    current_retry_count: int = 0

"""Deterministic preference extraction from completed trips.

No LLM call — uses curated keyword dictionaries to identify cuisine
and activity preferences from the Intent and Plan text.
"""

from __future__ import annotations

from finn.memory.models import PreferenceCategory
from finn.state import Intent, Plan

# ── Keyword dictionaries ────────────────────────────────────────────────

CUISINE_KEYWORDS: dict[str, str] = {
    "火锅": "火锅",
    "川菜": "川菜",
    "粤菜": "粤菜",
    "日料": "日料",
    "日式": "日料",
    "烧烤": "烧烤",
    "烤肉": "烤肉",
    "西餐": "西餐",
    "韩餐": "韩餐",
    "韩国料理": "韩餐",
    "海鲜": "海鲜",
    "素食": "素食",
    "小吃": "小吃",
    "早茶": "早茶",
    "湘菜": "湘菜",
    "本帮菜": "本帮菜",
    "西北菜": "西北菜",
    "面馆": "面食",
    "面食": "面食",
    "米线": "米线",
    "咖啡": "咖啡厅",
    "奶茶": "奶茶",
    "酒吧": "酒吧",
    "甜点": "甜点",
    "甜品": "甜点",
    "面包": "面包甜点",
    "披萨": "披萨",
    "汉堡": "汉堡",
    "炸鸡": "炸鸡",
    "串串": "串串",
    "麻辣烫": "麻辣烫",
    "小龙虾": "小龙虾",
    "烤鸭": "烤鸭",
    "卤味": "卤味",
    "饺子": "饺子",
    "拉面": "拉面",
    "泰国菜": "泰国菜",
    "越南菜": "越南菜",
    "印度菜": "印度菜",
}

ACTIVITY_KEYWORDS: dict[str, str] = {
    "电影": "看电影",
    "爬山": "爬山",
    "逛街": "逛街",
    "购物": "逛街",
    "逛商场": "逛街",
    "公园": "逛公园",
    "博物馆": "博物馆",
    "展览": "看展",
    "看展": "看展",
    "密室": "密室逃脱",
    "剧本杀": "剧本杀",
    "KTV": "唱歌",
    "唱歌": "唱歌",
    "温泉": "泡温泉",
    "滑雪": "滑雪",
    "游泳": "游泳",
    "健身": "健身",
    "骑行": "骑行",
    "徒步": "徒步",
    "游乐园": "游乐园",
    "动物园": "动物园",
    "桌游": "桌游",
    "保龄球": "保龄球",
    "台球": "台球",
    "射箭": "射箭",
    "卡丁车": "卡丁车",
    "蹦床": "蹦床",
    "攀岩": "攀岩",
    "冲浪": "冲浪",
    "潜水": "潜水",
    "滑冰": "滑冰",
    "看演出": "看演出",
    "演唱会": "看演出",
    "话剧": "看演出",
}


def _match_keywords(text: str, keyword_map: dict[str, str]) -> list[str]:
    """Return deduplicated mapped values for any keyword found in *text*."""
    found = []
    for kw, canonical in keyword_map.items():
        if kw in text:
            found.append(canonical)
    return list(dict.fromkeys(found))  # deduplicate, preserve order


def extract_learnings_from_trip(
    intent: Intent,
    plan: Plan | None = None,
) -> list[PreferenceCategory]:
    """Extract preference signals from a completed trip.

    Parameters
    ----------
    intent:
        The clarified Intent from the trip.
    plan:
        The final Plan (may be None if trip was cancelled early).

    Returns
    -------
    list[PreferenceCategory]
        Items ready to merge into the user's preference profile.
    """
    now = ""  # filled by caller in manager; we return items without timestamps
    items: list[PreferenceCategory] = []

    activity_text = intent.activity or ""

    # ── Cuisine keywords ───────────────────────────────────────────
    for cuisine in _match_keywords(activity_text, CUISINE_KEYWORDS):
        items.append(
            PreferenceCategory(
                category="dining",
                value=cuisine,
                confidence=0.55,
                source="inferred",
                occurrences=1,
            )
        )

    # ── Activity keywords ──────────────────────────────────────────
    for activity in _match_keywords(activity_text, ACTIVITY_KEYWORDS):
        items.append(
            PreferenceCategory(
                category="activity",
                value=activity,
                confidence=0.55,
                source="inferred",
                occurrences=1,
            )
        )

    # ── Area ───────────────────────────────────────────────────────
    if intent.area:
        items.append(
            PreferenceCategory(
                category="area",
                value=intent.area,
                confidence=0.55,
                source="inferred",
                occurrences=1,
            )
        )

    # ── Budget ─────────────────────────────────────────────────────
    if intent.budget_per_person is not None:
        items.append(
            PreferenceCategory(
                category="budget",
                value=str(intent.budget_per_person),
                confidence=0.50,
                source="inferred",
                occurrences=1,
            )
        )

    # ── Hard constraints (explicit, high confidence) ────────────────
    for constraint in intent.hard_constraints:
        items.append(
            PreferenceCategory(
                category="constraint",
                value=constraint,
                confidence=0.85,
                source="explicit",
                occurrences=1,
            )
        )

    # ── Preferences (explicit, medium-high confidence) ──────────────
    for pref in intent.preferences:
        items.append(
            PreferenceCategory(
                category="general",
                value=pref,
                confidence=0.75,
                source="explicit",
                occurrences=1,
            )
        )

    # ── Scenario tracking (count only, handled in manager) ─────────
    # (This is done in manager.save_trip when it updates favorite_scenarios)

    # ── Plan notes (free-text hints) ────────────────────────────────
    if plan and plan.notes:
        for cuisine in _match_keywords(plan.notes, CUISINE_KEYWORDS):
            items.append(
                PreferenceCategory(
                    category="dining",
                    value=cuisine,
                    confidence=0.50,
                    source="inferred",
                    occurrences=1,
                )
            )
        for activity in _match_keywords(plan.notes, ACTIVITY_KEYWORDS):
            items.append(
                PreferenceCategory(
                    category="activity",
                    value=activity,
                    confidence=0.50,
                    source="inferred",
                    occurrences=1,
                )
            )

    return items

"""ProfileBuilder — guided profile setup for first-time users.

Runs as a synchronous Q&A sequence in the CLI before the first graph
invocation. Not a LangGraph node — this is a simple sequential form.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

from finn.logger import logger
from finn.memory.models import PreferenceCategory, UserProfile
from finn.state import PartyMember

if TYPE_CHECKING:
    from finn.cli import CLI

# Beijing timezone
CST = timezone(timedelta(hours=8))

# ── Simple keyword parser for party description (no LLM needed for MVP) ──

ROLE_KEYWORDS: dict[str, list[str]] = {
    "self": ["自己", "我"],
    "spouse": ["老婆", "老公", "妻子", "丈夫", "太太", "先生", "爱人", "对象", "女朋友", "男朋友", "女友", "男友"],
    "child": ["孩子", "小孩", "儿子", "女儿", "宝宝", "小朋友", "娃"],
    "parent": ["爸爸", "妈妈", "父亲", "母亲", "爸", "妈", "父母", "老人"],
    "friend": ["朋友", "闺蜜", "兄弟", "哥们", "同事", "同学"],
}

AGE_PATTERNS: list[tuple[str, int]] = [
    # (regex-like keyword, age)
    ("1岁", 1), ("2岁", 2), ("3岁", 3), ("4岁", 4), ("5岁", 5),
    ("6岁", 6), ("7岁", 7), ("8岁", 8), ("9岁", 9), ("10岁", 10),
    ("11岁", 11), ("12岁", 12), ("13岁", 13), ("14岁", 14), ("15岁", 15),
    ("25岁", 25), ("30岁", 30), ("35岁", 35), ("40岁", 40),
    ("45岁", 45), ("50岁", 50),
]


def _find_age_for_role(text: str, role_keyword: str) -> int | None:
    """Find age attached to a specific role in the text.

    Looks for patterns like "X岁的孩子", "X岁孩子", or age near the role keyword.
    Returns None if not found.
    """
    import re
    # Pattern: "N岁的role" or "N岁role" right near the role keyword
    for m in re.finditer(r"(\d+)岁[的]?" + re.escape(role_keyword), text):
        return int(m.group(1))
    # Pattern: "role N岁" or "role，N岁"
    for m in re.finditer(re.escape(role_keyword) + r"[，,的]?(\d+)岁", text):
        return int(m.group(1))
    # Fallback: age near the role keyword, but only if no OTHER role keyword
    # sits between the role and the age (avoid attributing child's age to parent)
    idx = text.find(role_keyword)
    if idx >= 0:
        after = text[idx:idx + len(role_keyword) + 10]
        for pattern, value in AGE_PATTERNS:
            age_pos = after.find(pattern)
            if age_pos < 0:
                continue
            # Check if another role keyword sits between this role and the age
            segment = after[:age_pos]
            other_role_found = False
            for other_role, other_kws in ROLE_KEYWORDS.items():
                if other_role == role_keyword:
                    continue
                for kw in other_kws:
                    if kw in segment:
                        other_role_found = True
                        break
                if other_role_found:
                    break
            if not other_role_found:
                return value
    return None


def _parse_party_description(text: str) -> list[PartyMember]:
    """Heuristic parser for a free-text party description.

    Returns a list of PartyMember extracted from the text.
    """
    if not text.strip():
        return []

    members: list[PartyMember] = []
    text_lower = text.lower()

    # Detect roles
    detected_roles: dict[str, str] = {}  # role -> matched keyword
    for role, keywords in ROLE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                detected_roles[role] = kw
                break

    # Global constraints
    all_constraints: list[str] = []
    if "不吃辣" in text:
        all_constraints.append("不吃辣")
    if "海鲜过敏" in text:
        all_constraints.append("海鲜过敏")
    if "素食" in text:
        all_constraints.append("素食")
    if "清真" in text:
        all_constraints.append("清真饮食")
    if "减肥" in text or "减脂" in text:
        all_constraints.append("控制热量")
    has_child_seat = "需儿童座椅" in text or "儿童座椅" in text or "安全座椅" in text

    # Build members with per-role age detection
    if "spouse" in detected_roles:
        kw = detected_roles["spouse"]
        spouse_age = _find_age_for_role(text, kw) or 30
        m = PartyMember(
            role="spouse",
            age=spouse_age,
            constraints=all_constraints.copy(),
            preferences=[],
        )
        members.append(m)

    if "child" in detected_roles:
        kw = detected_roles["child"]
        child_age = _find_age_for_role(text, kw) or 5
        child_constraints = [c for c in all_constraints if c not in ("减肥", "控制热量")]
        if has_child_seat:
            child_constraints.append("需儿童座椅")
        m = PartyMember(
            role="child",
            age=child_age,
            constraints=list(set(child_constraints)),
            preferences=[],
        )
        members.append(m)

    if "friend" in detected_roles:
        kw = detected_roles["friend"]
        friend_age = _find_age_for_role(text, kw)
        m = PartyMember(
            role="friend",
            age=friend_age,
            constraints=all_constraints.copy(),
            preferences=[],
        )
        members.append(m)

    if "parent" in detected_roles:
        kw = detected_roles["parent"]
        parent_age = _find_age_for_role(text, kw) or 60
        m = PartyMember(
            role="parent",
            age=parent_age,
            constraints=all_constraints.copy(),
            preferences=[],
        )
        members.append(m)

    return members


BUDGET_MAP: dict[str, tuple[float, float]] = {
    "100以下": (0, 100),
    "100-300": (100, 300),
    "300-500": (300, 500),
    "不限": (0, 99999),
    "": (0, 99999),
}


class ProfileBuilder:
    """Guided profile setup for first-time users.

    Usage::

        builder = ProfileBuilder()
        profile = await builder.run(cli)
        MemoryManager().save_profile(profile)
    """

    QUESTIONS = [
        {
            "id": "name",
            "prompt": (
                "你好！我是 Finn，你的周末出行助手。先简单认识一下，怎么称呼你？\n"
                "（直接回车跳过）"
            ),
            "field": "name",
            "optional": True,
        },
        {
            "id": "home",
            "prompt": (
                "你住在哪个区或者哪个商圈？这样我推荐附近的好去处会更精准。\n"
                "（比如：朝阳区国贸、海淀区中关村。直接回车跳过）"
            ),
            "field": "home_location",
            "optional": True,
        },
        {
            "id": "party",
            "prompt": (
                "你通常会跟谁一起出去玩？告诉我同行人的角色、年龄和特别需要注意的地方。\n"
                "（比如：带老婆和5岁的孩子，孩子需要儿童座椅，老婆海鲜过敏。\n"
                "直接回车跳过）"
            ),
            "field": "party_description",
            "optional": True,
        },
        {
            "id": "taste",
            "prompt": (
                "口味上有什么偏好吗？喜欢哪类菜？有什么忌口的？\n"
                "（比如：喜欢川菜火锅、不吃香菜、海鲜过敏。直接回车跳过）"
            ),
            "field": "taste_prefs",
            "optional": True,
        },
        {
            "id": "budget",
            "prompt": (
                "出去吃喝玩乐，人均预算一般在什么范围？\n"
                "（输入选项：100以下 / 100-300 / 300-500 / 不限。直接回车选\"不限\"）"
            ),
            "field": "budget_range",
            "optional": True,
        },
    ]

    def __init__(self):
        self._profile = UserProfile(
            created_at=datetime.now(CST).isoformat(),
            updated_at=datetime.now(CST).isoformat(),
        )

    async def run(self, cli: CLI) -> UserProfile:
        """Run the guided setup flow.

        Returns a completed UserProfile ready to save.
        """
        cli._console.print()
        cli._console.print(
            "欢迎首次使用 Finn！我先简单了解一下你的偏好，大概需要 1 分钟。\n"
            "（所有问题都可以按回车跳过，以后还能补充）\n"
        )

        for q in self.QUESTIONS:
            answer = await self._ask(cli, q["prompt"])
            self._process_answer(q["id"], q["field"], answer.strip())

        self._profile.setup_complete = True
        self._profile.updated_at = datetime.now(CST).isoformat()

        cli._console.print("\n[bold #00d7af]设置完成！[/bold #00d7af] 以后每次出行我都会参考你的画像。\n"
                           "随时可以输入 [bold]/profile[/bold] 查看，或手动编辑 [bold]~/.finn/profile.json[/bold]\n")
        return self._profile

    async def _ask(self, cli: CLI, prompt: str) -> str:
        """Print a prompt and await user input."""
        cli._console.print(f"[bold]{prompt}[/bold]")

        if hasattr(cli, "_session") and cli._session:
            try:
                answer = await cli._session.prompt_async(
                    [("class:prompt", "> ")]
                )
                return answer
            except (EOFError, KeyboardInterrupt):
                return ""
        else:
            try:
                return input("> ")
            except (EOFError, KeyboardInterrupt):
                return ""

    def _process_answer(self, qid: str, field: str, answer: str) -> None:
        """Parse and store an answer into the profile."""
        if not answer:
            return

        if qid == "name":
            self._profile.name = answer
            logger.debug("Profile: name=%s", answer)

        elif qid == "home":
            self._profile.home_location = answer
            logger.debug("Profile: home=%s", answer)

        elif qid == "party":
            members = _parse_party_description(answer)
            if members:
                self._profile.saved_party_members = members
                self._profile.default_party_size = len(members) + 1  # +1 for self
                logger.debug("Profile: party=%d members", len(members))

        elif qid == "taste":
            # Parse taste preferences into PreferenceCategory items
            items = _parse_taste_prefs(answer)
            if items:
                self._profile.preferences.learned_items = items
                logger.debug("Profile: %d taste preferences", len(items))
                # Rebuild denormalised fields
                from finn.memory.manager import _rebuild_denormalised
                _rebuild_denormalised(self._profile.preferences)

        elif qid == "budget":
            budget = BUDGET_MAP.get(answer, (0, 99999))
            self._profile.preferences.budget_range = budget
            if budget != (0, 99999):
                lo, hi = budget
                self._profile.preferences.learned_items.append(
                    PreferenceCategory(
                        category="budget",
                        value=f"{int(lo)}-{int(hi)}元",
                        confidence=0.9,
                        source="explicit",
                        occurrences=1,
                        first_seen=datetime.now(CST).isoformat(),
                        last_seen=datetime.now(CST).isoformat(),
                    )
                )
            logger.debug("Profile: budget=%s", answer)


# ── Taste preference parser ──────────────────────────────────────────────

CUISINE_WORDS = {
    "川菜", "粤菜", "日料", "日式", "韩餐", "韩国料理", "西餐", "烧烤",
    "烤肉", "火锅", "海鲜", "素食", "早茶", "湘菜", "本帮菜", "西北菜",
    "面食", "面馆", "米线", "小吃", "咖啡", "奶茶", "酒吧", "甜点",
    "甜品", "披萨", "汉堡", "炸鸡", "串串", "麻辣烫", "小龙虾",
    "烤鸭", "卤味", "饺子", "拉面", "泰国菜", "越南菜", "印度菜",
    "面包", "甜点",
}

CONSTRAINT_WORDS = {
    "不吃辣": "不吃辣",
    "不吃香菜": "不吃香菜",
    "海鲜过敏": "海鲜过敏",
    "素食": "素食",
    "清真": "清真饮食",
    "不吃蒜": "不吃蒜",
    "不吃葱": "不吃葱",
    "不吃姜": "不吃姜",
    "不吃牛": "不吃牛肉",
    "不吃羊": "不吃羊肉",
    "不吃猪": "不吃猪肉",
    "减肥": "控制热量",
    "减脂": "控制热量",
    "控糖": "控糖",
}

# Words from CUISINE_WORDS that are substrings of constraint keywords
_CUISINE_EXCLUSIONS: dict[str, str] = {
    "海鲜": "海鲜过敏",  # "海鲜" in "海鲜过敏" → don't count as cuisine
    "素食": "素食",      # Listed as both cuisine and constraint; handled below
}


def _parse_taste_prefs(text: str) -> list[PreferenceCategory]:
    """Parse free-text taste preferences into PreferenceCategory items."""
    now = datetime.now(CST).isoformat()
    items: list[PreferenceCategory] = []

    # Detect constraints first — they take priority over cuisine matches
    found_constraint_phrases: set[str] = set()
    for keyword, constraint in CONSTRAINT_WORDS.items():
        if keyword in text:
            items.append(PreferenceCategory(
                category="constraint",
                value=constraint,
                confidence=0.9,
                source="explicit",
                occurrences=1,
                first_seen=now,
                last_seen=now,
            ))
            found_constraint_phrases.add(keyword)

    # Extract cuisine preferences — skip words that appear only within constraint phrases
    for word in CUISINE_WORDS:
        if word not in text:
            continue
        # Check if this cuisine word is only appearing in a constraint context
        skip = False
        for cuisine_kw, constraint_kw in _CUISINE_EXCLUSIONS.items():
            if word == cuisine_kw and constraint_kw in text and word in constraint_kw:
                skip = True
                break
        if skip:
            continue
        items.append(PreferenceCategory(
            category="dining",
            value=word,
            confidence=0.9,
            source="explicit",
            occurrences=1,
            first_seen=now,
            last_seen=now,
        ))

    return items

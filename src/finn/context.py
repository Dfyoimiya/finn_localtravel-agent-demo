"""ContextAgent — parallel weather + regeocode fetch before clarify_intent.

Runs via MCP, no LLM. Sets state.weather for downstream nodes.
"""

from __future__ import annotations

import asyncio

from finn.logger import logger
from finn.state import AgentState, WeatherContext
from finn.mcp import mcp_session


async def context_agent(state: AgentState) -> dict:
    """Fetch weather and enrich geo context in parallel via MCP.

    Runs AFTER clarify_intent so plan_date and city are known from the
    user's natural-language input. Weather is always for the trip date.
    Graceful degradation: returns weather=None on MCP failure.
    """
    extract = state.get("extract_result")
    city = extract.intent.city if extract else None
    plan_date = extract.intent.plan_date if extract else None
    user_coords = state.get("user_coords", "")

    weather: WeatherContext | None = None
    geo_district: str | None = None

    logger.info("→ context_agent | city=%s date=%s coords=%s",
                city or "?", plan_date or "?", user_coords[:20] if user_coords else "?")

    try:
        async with mcp_session() as (tools, _):
            tasks = []
            tasks.append(_fetch_weather(tools, city, plan_date))

            if user_coords:
                tasks.append(_regeocode(tools, user_coords))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, WeatherContext):
                    weather = r
                elif isinstance(r, str) and r:
                    geo_district = r

    except Exception as exc:
        logger.warning("context_agent MCP failed: %s", exc)

    if weather:
        logger.info("← context_agent | weather=%s %.0f~%.0f°C indoor=%s",
                    weather.condition, weather.temp_low or 0,
                    weather.temp_high or 0, weather.indoor_recommended)
    if geo_district:
        logger.info("← context_agent | district=%s", geo_district)

    return {"weather": weather}


async def _fetch_weather(tools, city: str | None, date: str | None) -> WeatherContext | None:
    """Fetch weather from Amap and build WeatherContext.

    Tries to match the plan_date in the forecast; falls back to the first
    available day if the plan date is beyond the forecast window.
    The ``date`` field on WeatherContext always reflects the plan_date so
    downstream display shows the correct trip date.
    """
    weather_tool = next((t for t in tools if t.name == "maps_weather"), None)
    if not weather_tool:
        return None

    target_city = city or "重庆"
    try:
        result = await weather_tool.ainvoke({"city": target_city})
        data = _extract_json(str(result)) if "{" in str(result) else {}
    except Exception:
        return None

    forecasts = data.get("forecasts", [])
    if not forecasts:
        return None

    # Find forecast for the plan date; if beyond forecast window use
    # the last available day (closest to the plan date)
    target = None
    for f in forecasts:
        if f.get("date") == date:
            target = f
            break
    if target is None:
        target = forecasts[-1]  # furthest-out forecast, nearest to plan date

    condition = target.get("dayweather", "")
    indoor = any(w in condition for w in ["雨", "雪", "高温", "沙尘"])

    try:
        temp_high = float(target.get("daytemp_float", target.get("daytemp", 0)))
    except (ValueError, TypeError):
        temp_high = None
    try:
        temp_low = float(target.get("nighttemp_float", target.get("nighttemp", 0)))
    except (ValueError, TypeError):
        temp_low = None

    return WeatherContext(
        date=date or target.get("date", ""),
        condition=condition,
        temp_high=temp_high,
        temp_low=temp_low,
        wind=target.get("daywind", ""),
        indoor_recommended=indoor,
    )


async def _regeocode(tools, coords: str) -> str | None:
    """Reverse geocode coordinates to address string."""
    regeo_tool = next((t for t in tools if t.name == "maps_regeocode"), None)
    if not regeo_tool:
        return None

    try:
        result = await regeo_tool.ainvoke({"location": coords})
        data = _extract_json(str(result)) if "{" in str(result) else {}
        regeocode = data.get("regeocode", {})
        address_component = regeocode.get("addressComponent", {})
        district = address_component.get("district", "")
        township = address_component.get("township", "")
        return f"{district}{township}" if district else None
    except Exception:
        return None


def _extract_json(text: str) -> dict:
    """Extract first JSON object from text."""
    import json
    import re
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}

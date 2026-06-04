"""Auto-detect user location via Amap IP API and current time."""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone, timedelta

from finn.config import config

# Beijing timezone
_CST = timezone(timedelta(hours=8))


def get_current_context() -> dict:
    """Return current time and IP-based location for context injection.

    Returns a dict with:
        current_time: str like "2026-06-03 14:30 (周三)"
        location_city: str like "重庆市"
        location_district: str or None
        location_coords: str like "106.42,29.83" or None
        location_province: str or None
    """
    now = datetime.now(_CST)
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    ctx = {
        "current_time": now.strftime(f"%Y-%m-%d %H:%M ({weekdays[now.weekday()]})"),
        "location_city": None,
        "location_district": None,
        "location_coords": None,
        "location_province": None,
    }

    try:
        ip_info = _get_ip_location()
        if ip_info:
            ctx["location_province"] = ip_info.get("province")
            ctx["location_city"] = ip_info.get("city")
            ctx["location_district"] = ip_info.get("district")
            # Amap IP API returns a rectangle string "min_lng,min_lat;max_lng,max_lat"
            rect = ip_info.get("rectangle")
            if rect:
                parts = rect.split(";")
                if len(parts) == 2:
                    min_lng, min_lat = parts[0].split(",")
                    max_lng, max_lat = parts[1].split(",")
                    center_lng = (float(min_lng) + float(max_lng)) / 2
                    center_lat = (float(min_lat) + float(max_lat)) / 2
                    ctx["location_coords"] = f"{center_lng:.4f},{center_lat:.4f}"
    except Exception:
        pass

    return ctx


def format_context(ctx: dict) -> str:
    """Format context dict as a system message for the LLM."""
    parts = [f"当前时间: {ctx['current_time']}"]
    loc_parts = []
    if ctx["location_province"]:
        loc_parts.append(ctx["location_province"])
    if ctx["location_city"]:
        loc_parts.append(ctx["location_city"])
    if ctx["location_district"]:
        loc_parts.append(ctx["location_district"])
    if loc_parts:
        parts.append(f"用户当前位置: {''.join(loc_parts)} (基于IP自动定位)")
    if ctx["location_coords"]:
        parts.append(f"用户当前坐标: {ctx['location_coords']}")
    return "\n".join(parts)


def _get_ip_location(ip: str | None = None) -> dict | None:
    """Call Amap IP location API. Returns province/city/district/rectangle."""
    api_key = config.amap_api_key
    if not api_key:
        return None

    url = f"https://restapi.amap.com/v3/ip?key={api_key}"
    if ip:
        url += f"&ip={ip}"

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None

    if data.get("status") != "1":
        return None

    return {
        "province": data.get("province") or None,
        "city": data.get("city") or None,
        "district": data.get("district") or None,
        "adcode": data.get("adcode") or None,
        "rectangle": data.get("rectangle") or None,
    }

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo


# 计算项目根目录，后面读取 data 文件时会用到。
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_DESCRIPTIONS = {
    0: "晴朗",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴天",
    45: "有雾",
    48: "冻雾",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "强毛毛雨",
    56: "小冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "小冻雨",
    67: "大冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪",
    80: "小阵雨",
    81: "中阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


def fetch_json(url: str, params: dict[str, str]) -> dict:
    """请求 JSON 接口并返回解析后的字典。"""
    query = urlencode(params)
    request_url = f"{url}?{query}"

    try:
        with urlopen(request_url, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Weather API HTTP error: {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError("Weather API network error.") from exc


def format_location_name(location: dict) -> str:
    """把地理编码结果整理成更容易读的地点名称。"""
    parts = [
        location.get("name"),
        location.get("admin1"),
        location.get("country"),
    ]
    return ", ".join(part for part in parts if part)


def describe_weather_code(code: int | None) -> str:
    """把 Open-Meteo 的天气代码转成中文描述。"""
    if code is None:
        return "未知天气"
    return WEATHER_CODE_DESCRIPTIONS.get(code, f"未知天气代码 {code}")


def get_current_time(timezone_name: str = "Asia/Shanghai") -> str:
    """
    返回指定时区的当前时间，给 agent 作为可调用工具使用。

    注意事项：
    - timezone_name 需要使用标准 IANA 时区名称，例如 Asia/Shanghai。
    - 这个工具只负责“取时间”，不要把总结、解释之类的逻辑塞进工具里。
    - 工具返回值越单一越好，这样模型更容易稳定使用。
    """
    try:
        # ZoneInfo 使用标准 IANA 时区名称，比如 Asia/Shanghai。
        current_time = datetime.now(ZoneInfo(timezone_name))
    except Exception:
        return (
            f"Invalid timezone: {timezone_name}. "
            "Use a standard IANA timezone like Asia/Shanghai or America/Los_Angeles."
        )

    return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")


def read_local_note(filename: str = "notes.txt") -> str:
    """
    读取 data 目录下的 txt 文件内容，避免 agent 直接访问任意路径。

    注意事项：
    - filename 最终只会保留文件名本身，不能跨目录读取其他文件。
    - 当前示例只支持 .txt，目的是先让你专注理解 Tool 调用流程。
    - 如果以后要支持更多格式，建议新建专门的工具，而不是把一个函数做得过于复杂。
    - 工具最好返回“可直接消费”的文本结果，避免把底层细节暴露给模型。
    """
    # 只保留文件名本身，防止传入类似 ../../secret.txt 这样的路径。
    safe_name = Path(filename).name
    note_path = DATA_DIR / safe_name

    # 这里只允许读取 txt，方便新手先聚焦在工具调用，而不是文件解析细节。
    if note_path.suffix.lower() != ".txt":
        return "Only .txt files are supported in this starter project."

    if not note_path.exists():
        # 如果文件不存在，就把当前可用的 txt 文件列出来，方便用户重试。
        available_files = sorted(path.name for path in DATA_DIR.glob("*.txt"))
        if not available_files:
            return "No note files were found in the data directory."

        return (
            f"File not found: {safe_name}. "
            f"Available files: {', '.join(available_files)}"
        )

    # 读取 UTF-8 文本内容并去掉首尾空白。
    return note_path.read_text(encoding="utf-8").strip()


def get_weather_by_city(city: str) -> str:
    """
    查询指定城市的当前天气。

    注意事项：
    - 这个工具先调用地理编码接口，把城市名解析成经纬度，再查询天气。
    - city 尽量传常见城市名，例如 Shanghai、Beijing、Tokyo。
    - 如果城市重名，工具会默认取搜索结果里的第一项。
    - 这个工具依赖网络请求；如果接口超时或不可达，会返回明确错误信息。
    """
    city = city.strip()
    if not city:
        return "城市名不能为空。"

    try:
        geocoding_data = fetch_json(
            GEOCODING_API_URL,
            {
                "name": city,
                "count": "1",
                "language": "zh",
                "format": "json",
            },
        )
    except RuntimeError as exc:
        return f"查询城市坐标失败：{exc}"

    results = geocoding_data.get("results") or []
    if not results:
        return f"没有找到城市：{city}。请尝试使用更完整的城市名。"

    location = results[0]
    latitude = location["latitude"]
    longitude = location["longitude"]
    timezone_name = location.get("timezone", "auto")
    display_name = format_location_name(location)

    try:
        forecast_data = fetch_json(
            FORECAST_API_URL,
            {
                "latitude": str(latitude),
                "longitude": str(longitude),
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "precipitation,weather_code,wind_speed_10m"
                ),
                "timezone": timezone_name,
            },
        )
    except RuntimeError as exc:
        return f"查询天气失败：{exc}"

    current = forecast_data.get("current")
    if not current:
        return f"暂时无法获取 {display_name} 的天气数据。"

    weather_code = current.get("weather_code")
    weather_text = describe_weather_code(weather_code)
    time_text = current.get("time", "未知时间")
    temperature = current.get("temperature_2m", "未知")
    apparent_temperature = current.get("apparent_temperature", "未知")
    humidity = current.get("relative_humidity_2m", "未知")
    precipitation = current.get("precipitation", "未知")
    wind_speed = current.get("wind_speed_10m", "未知")

    return (
        f"{display_name} 当前天气如下：\n"
        f"- 观测时间：{time_text}\n"
        f"- 天气：{weather_text}\n"
        f"- 气温：{temperature}°C\n"
        f"- 体感温度：{apparent_temperature}°C\n"
        f"- 相对湿度：{humidity}%\n"
        f"- 降水：{precipitation} mm\n"
        f"- 风速：{wind_speed} km/h"
    )

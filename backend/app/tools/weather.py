"""weather tool: current conditions from Open-Meteo (no key). Geocode the place, fetch the
conditions, map WMO codes to words.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import AsyncTool, ToolError
from app.tools.http import get_json

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# https://open-meteo.com/en/docs (WMO 4677 weather interpretation codes)
_WMO: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "violent showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with light hail",
    99: "thunderstorm with heavy hail",
}


def describe_code(code: Any) -> str:
    try:
        return _WMO.get(int(code), f"weather code {code}")
    except (TypeError, ValueError):
        return "unknown conditions"


class WeatherArgs(BaseModel):
    place: str = Field(
        description="A city or place name, optionally with country, e.g. 'Madrid' or "
        "'São Paulo, Brazil'.",
        min_length=2,
        max_length=100,
    )


class WeatherTool(AsyncTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "weather"
    description: str = (
        "Current weather (temperature, conditions, wind) for a city or place. Use it when "
        "the user asks what the weather is like somewhere right now."
    )
    args_schema: type[BaseModel] = WeatherArgs

    client: httpx.AsyncClient = Field(exclude=True)

    async def _arun(self, place: str) -> dict[str, Any]:
        geo = await get_json(
            self.client,
            GEOCODE_URL,
            {"name": place, "count": 1, "language": "en", "format": "json"},
            service="the place lookup",
        )
        results = geo.get("results") if isinstance(geo, dict) else None
        if not results:
            raise ToolError(f"could not find a place called {place!r}")
        spot = results[0]
        forecast = await get_json(
            self.client,
            FORECAST_URL,
            {
                "latitude": spot["latitude"],
                "longitude": spot["longitude"],
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
            service="the weather service",
        )
        current = forecast.get("current") if isinstance(forecast, dict) else None
        if not current:
            raise ToolError("the weather service returned no current conditions")
        region = ", ".join(
            p for p in (spot.get("name"), spot.get("admin1"), spot.get("country")) if p
        )
        return {
            "place": region,
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "conditions": describe_code(current.get("weather_code")),
            "wind_kmh": current.get("wind_speed_10m"),
            "observed_at": current.get("time"),
            "timezone": forecast.get("timezone"),
            "source": "Open-Meteo",
        }

"""gif_search and weather, driven through an httpx mock transport (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from app.tools.base import ToolError
from app.tools.gif import GifSearchTool
from app.tools.weather import WeatherTool, describe_code

Handler = Callable[[httpx.Request], httpx.Response]


def client_with(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=1.0)


def json_response(payload: object, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


GIPHY_HIT = {
    "data": [
        {
            "title": "Happy Cat GIF",
            "url": "https://giphy.com/gifs/abc",
            "images": {
                "fixed_height": {
                    "url": "https://media.giphy.com/x/200.gif",
                    "width": "300",
                    "height": "200",
                }
            },
        }
    ]
}


async def test_gif_search_returns_first_hit_with_key_and_rating() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return json_response(GIPHY_HIT)

    tool = GifSearchTool(api_key="k", client=client_with(handler))
    out = await tool.ainvoke({"query": "happy cat"})
    assert out == {
        "query": "happy cat",
        "title": "Happy Cat GIF",
        "url": "https://media.giphy.com/x/200.gif",
        "width": 300,
        "height": 200,
        "page": "https://giphy.com/gifs/abc",
        "source": "GIPHY",
    }
    params = dict(seen[0].url.params)
    assert params["api_key"] == "k" and params["q"] == "happy cat" and params["rating"] == "g"


async def test_gif_search_no_results() -> None:
    tool = GifSearchTool(api_key="k", client=client_with(lambda r: json_response({"data": []})))
    with pytest.raises(ToolError, match="no GIF found"):
        await tool.ainvoke({"query": "zzzz"})


@pytest.mark.parametrize(
    ("status", "message"),
    [(401, "API key"), (429, "rate limiting"), (500, "error \\(500\\)")],
)
async def test_gif_search_http_errors_are_tool_errors(status: int, message: str) -> None:
    tool = GifSearchTool(api_key="k", client=client_with(lambda r: json_response({}, status)))
    with pytest.raises(ToolError, match=message):
        await tool.ainvoke({"query": "x"})


async def test_gif_search_timeout_and_network_errors() -> None:
    def timeout(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    def down(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ToolError, match="did not answer in time"):
        await GifSearchTool(api_key="k", client=client_with(timeout)).ainvoke({"query": "x"})
    with pytest.raises(ToolError, match="unreachable"):
        await GifSearchTool(api_key="k", client=client_with(down)).ainvoke({"query": "x"})


def test_gif_schema_hides_key_and_client() -> None:
    from langchain_core.utils.function_calling import convert_to_openai_tool

    schema = convert_to_openai_tool(
        GifSearchTool(api_key="secret", client=client_with(lambda r: json_response({})))
    )
    assert "secret" not in json.dumps(schema)
    assert schema["function"]["parameters"]["required"] == ["query"]


GEO_HIT = {
    "results": [
        {
            "name": "Madrid",
            "admin1": "Madrid",
            "country": "Spain",
            "latitude": 40.4,
            "longitude": -3.7,
        }
    ]
}
FORECAST = {
    "timezone": "Europe/Madrid",
    "current": {
        "time": "2026-09-17T15:00",
        "temperature_2m": 27.3,
        "apparent_temperature": 26.1,
        "weather_code": 2,
        "wind_speed_10m": 12.5,
    },
}


async def test_weather_geocodes_then_reads_current_conditions() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if "geocoding" in req.url.host:
            return json_response(GEO_HIT)
        return json_response(FORECAST)

    out = await WeatherTool(client=client_with(handler)).ainvoke({"place": "Madrid"})
    assert out == {
        "place": "Madrid, Madrid, Spain",
        "temperature_c": 27.3,
        "feels_like_c": 26.1,
        "conditions": "partly cloudy",
        "wind_kmh": 12.5,
        "observed_at": "2026-09-17T15:00",
        "timezone": "Europe/Madrid",
        "source": "Open-Meteo",
    }
    assert [r.url.host for r in seen] == ["geocoding-api.open-meteo.com", "api.open-meteo.com"]
    assert dict(seen[1].url.params)["latitude"] == "40.4"


async def test_weather_unknown_place() -> None:
    tool = WeatherTool(client=client_with(lambda r: json_response({"results": []})))
    with pytest.raises(ToolError, match="could not find a place"):
        await tool.ainvoke({"place": "Nowheresville"})


def test_wmo_codes() -> None:
    assert describe_code(0) == "clear sky"
    assert describe_code("61") == "light rain"
    assert describe_code(12345) == "weather code 12345"
    assert describe_code(None) == "unknown conditions"

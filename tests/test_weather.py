"""Tests for the get_weather tool (Open-Meteo, no API key) — wave-g.

Every network boundary is faked: the geocoding + forecast HTTP is mocked so no
live request is ever made. Covered:

* metric vs imperial formatting (°C/km·h vs °F/mph) and the unit params sent,
* the registered tool defaults to the configured location + units but honours an
  explicit location argument,
* graceful failure: a network error / unknown place / bad response returns a clear
  "unavailable" string and never raises,
* the WMO code -> phrase mapping,
* the full urllib wiring (urlopen monkeypatched) hits the documented endpoints.
"""
# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name
# pylint: disable=import-outside-toplevel,unused-argument
# (tests import lazily to keep heavy modules out of collection; fakes mirror SDK signatures)

import json
import urllib.error

import pytest

from my_stt_tts import tools
from my_stt_tts.config import Config
from my_stt_tts.tools import ToolRegistry, default_tools, get_weather, make_weather_tool

# A representative Open-Meteo geocoding hit and forecast block.
_GEO_HIT = {
    "results": [
        {"name": "Lausanne", "country": "Switzerland", "latitude": 46.516, "longitude": 6.632}
    ]
}
_FORECAST = {
    "current": {
        "temperature_2m": 18.4,
        "apparent_temperature": 17.1,
        "weather_code": 3,
        "wind_speed_10m": 12.7,
    }
}


def _fake_http(geo: dict, forecast: dict) -> tuple[list[dict], object]:
    """Return (recorded_params, fake _http_get_json) returning geo then forecast."""
    calls: list[dict] = []

    def _http(url: str, params: dict):  # noqa: ANN202
        calls.append({"url": url, "params": params})
        return geo if "geocoding" in url else forecast

    return calls, _http


def test_metric_formatting(monkeypatch):
    _calls, http = _fake_http(_GEO_HIT, _FORECAST)
    monkeypatch.setattr(tools, "_http_get_json", http)
    out = get_weather("Lausanne", units="metric")
    assert "Lausanne, Switzerland" in out
    assert "overcast" in out  # WMO code 3
    assert "18°C" in out and "°F" not in out
    assert "feels like 17°C" in out
    assert "13 km/h" in out  # 12.7 rounded


def test_imperial_formatting_and_unit_params(monkeypatch):
    fc = {
        "current": {
            "temperature_2m": 65.1,
            "apparent_temperature": 63.0,
            "weather_code": 0,
            "wind_speed_10m": 8.0,
        }
    }
    calls, http = _fake_http(_GEO_HIT, fc)
    monkeypatch.setattr(tools, "_http_get_json", http)
    out = get_weather("Lausanne", units="imperial")
    assert "65°F" in out and "°C" not in out
    assert "8 mph" in out and "km/h" not in out
    assert "clear sky" in out  # WMO code 0
    # The forecast request asked Open-Meteo for imperial units explicitly.
    forecast_params = calls[-1]["params"]
    assert forecast_params["temperature_unit"] == "fahrenheit"
    assert forecast_params["wind_speed_unit"] == "mph"


def test_tool_defaults_to_config_location_and_units(monkeypatch):
    calls, http = _fake_http(_GEO_HIT, _FORECAST)
    monkeypatch.setattr(tools, "_http_get_json", http)
    tool = make_weather_tool(default_location="Bern, Switzerland", default_units="metric")
    # No explicit location -> the configured default is geocoded.
    tool.run({})
    assert calls[0]["params"]["name"] == "Bern, Switzerland"


def test_tool_honours_explicit_location(monkeypatch):
    calls, http = _fake_http(
        {"results": [{"name": "Tokyo", "country": "Japan", "latitude": 35.7, "longitude": 139.7}]},
        _FORECAST,
    )
    monkeypatch.setattr(tools, "_http_get_json", http)
    tool = make_weather_tool(default_location="Lausanne, Switzerland")
    out = tool.run({"location": "Tokyo"})
    assert calls[0]["params"]["name"] == "Tokyo"
    assert "Tokyo, Japan" in out


def test_network_failure_is_graceful(monkeypatch):
    def _boom(url: str, params: dict):  # noqa: ANN202, ARG001
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(tools, "_http_get_json", _boom)
    out = get_weather("Lausanne", units="metric")
    assert "unavailable" in out.lower()
    assert "Lausanne" in out  # names the place it tried


def test_unknown_place_is_graceful(monkeypatch):
    calls, http = _fake_http({"results": []}, _FORECAST)
    monkeypatch.setattr(tools, "_http_get_json", http)
    out = get_weather("Xyzzylandia", units="metric")
    assert "unavailable" in out.lower()
    assert len(calls) == 1  # bailed after geocoding found nothing


def test_blank_location_returns_error():
    assert get_weather("   ").startswith("error")


def test_empty_current_block_is_graceful(monkeypatch):
    _calls, http = _fake_http(_GEO_HIT, {"current": {}})
    monkeypatch.setattr(tools, "_http_get_json", http)
    out = get_weather("Lausanne")
    assert "unavailable" in out.lower()


def test_wmo_descriptions():
    assert tools._wmo_description(0) == "clear sky"
    assert tools._wmo_description(95) == "thunderstorm"
    assert "unsettled" in tools._wmo_description(123456)  # unknown code -> neutral


def test_weather_tool_registered_in_defaults():
    reg = ToolRegistry(default_tools(location="Lausanne, Switzerland", units="metric"))
    assert "get_weather" in reg
    schema = next(t for t in reg.anthropic_tools() if t["name"] == "get_weather")
    assert "location" in schema["input_schema"]["properties"]
    # location is optional (omit -> use the configured home).
    assert "required" not in schema["input_schema"]


def test_brain_builds_weather_tool_from_config():
    from my_stt_tts.brain import Brain

    cfg = Config(anthropic_api_key="x", location="Geneva, Switzerland", units="imperial")
    brain = Brain(cfg)
    assert brain.tools is not None and "get_weather" in brain.tools


def test_urllib_wiring_hits_open_meteo(monkeypatch):
    """End-to-end through _http_get_json: urlopen is the only thing faked."""
    seen_urls: list[str] = []

    class _Resp:
        def __init__(self, payload: dict) -> None:
            self._b = json.dumps(payload).encode("utf-8")

        def read(self):  # noqa: ANN202
            return self._b

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def _fake_urlopen(req, timeout=None):  # noqa: ANN001, ANN202, ARG001
        url = req.full_url
        seen_urls.append(url)
        payload = _GEO_HIT if "geocoding" in url else _FORECAST
        return _Resp(payload)

    monkeypatch.setattr(tools.urllib.request, "urlopen", _fake_urlopen)
    out = get_weather("Lausanne", units="metric")
    assert "Lausanne" in out
    assert any("geocoding-api.open-meteo.com" in u for u in seen_urls)
    assert any(u.startswith("https://api.open-meteo.com/v1/forecast") for u in seen_urls)


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """Safety net: a test that forgets to patch must NOT reach the network."""

    def _blocked(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("a weather test attempted a real network call")

    monkeypatch.setattr(tools.urllib.request, "urlopen", _blocked)
    yield

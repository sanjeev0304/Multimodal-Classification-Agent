"""
classification_mcp_server.py
----------------------------
MCP server exposing per-modality classification tools, each backed by
Google Gemini (via its OpenAI-compatible endpoint):

    - classify_document        -> Invoice / Report / Contract / Manual
    - classify_sensor_data     -> sensor_type (inferred) + Normal / Fault
    - classify_network_packet  -> Normal / Suspicious / Priority
    - classify_image_tools     -> list of detected industrial tools + category + condition

Run directly (stdio transport) or let streamlit_app.py spawn it.
"""

import os
from typing import List, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
from mcp.server.fastmcp import FastMCP

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _completion_kwargs() -> dict:
    return {
        "model": MODEL,
        "response_format": {"type": "json_object"},
        "max_tokens": 1024,
        "temperature": 0.0,
    }


mcp = FastMCP("ClassificationServer")


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------
class DocumentResult(BaseModel):
    category: Literal["Invoice", "Report", "Contract", "Manual"] = Field(
        description="Primary document category."
    )
    confidence_score: float = Field(ge=0.0, le=1.0)
    key_topics: List[str] = Field(description="Key terms that drove the decision.")
    justification: str = Field(description="One-sentence reasoning for the category.")


class SensorResult(BaseModel):
    sensor_type: str = Field(
        description="Best-guess sensor type inferred from the readings "
        "(e.g. Temperature, Vibration, Pressure, Humidity, RPM/Speed, "
        "Voltage, Current, Flow Rate, or Unknown if it cannot be determined)."
    )
    state: Literal["Normal", "Fault"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    indicators: List[str] = Field(description="Readings that drove the decision.")
    justification: str


class NetworkResult(BaseModel):
    classification: Literal["Normal", "Suspicious", "Priority"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    signals: List[str] = Field(description="Traffic signals that drove the decision.")
    justification: str


class ToolDetection(BaseModel):
    tool_name: str = Field(description="Name of the industrial tool (e.g. Open-End Wrench, Phillips Screwdriver).")
    category: str = Field(
        description="Tool category: Hand Tool, Power Tool, Cutting Tool, "
        "Measuring Instrument, Fastening Tool, or Other."
    )
    condition: Literal["Good", "Worn", "Damaged", "Unknown"] = Field(
        description="Visible condition of the tool based on surface wear, rust, or damage."
    )
    confidence_score: float = Field(ge=0.0, le=1.0)


class ImageToolResult(BaseModel):
    tools_detected: List[ToolDetection] = Field(
        description="All industrial tools identified in the image."
    )
    total_count: int = Field(description="Total number of tools detected.")
    scene_description: str = Field(description="One-sentence description of the overall scene.")
    justification: str = Field(description="Brief reasoning for the detections made.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _classify(system_prompt: str, user_text: str, schema: type[BaseModel]) -> dict:
    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        **_completion_kwargs(),
    )
    raw = resp.choices[0].message.content
    if not raw or not raw.strip():
        raise ValueError("Model returned empty content. Retry or adjust the prompt.")
    return schema.model_validate_json(raw).model_dump()


def _vision_classify(image_b64: str, media_type: str, schema: type[BaseModel]) -> dict:
    """Call Gemini with an image payload using the OpenAI vision content format."""
    system = (
        "You are an expert industrial tool recognition system deployed on a factory floor. "
        "Tools may be scattered, worn, partially obscured, or have faded labels — handle all of these. "
        "Identify every industrial tool visible in the image. "
        "For each tool provide:\n"
        "  - tool_name: specific name (e.g. Open-End Wrench, Phillips Screwdriver, Angle Grinder)\n"
        "  - category: Hand Tool / Power Tool / Cutting Tool / Measuring Instrument / Fastening Tool / Other\n"
        "  - condition: assess carefully from visual cues —\n"
        "      Good: no visible wear, clean surface, intact edges\n"
        "      Worn: visible scratches, minor rust, rounded edges, paint loss\n"
        "      Damaged: cracks, severe rust, broken parts, bent or deformed\n"
        "      Unknown: too obscured to assess\n"
        "  - confidence_score: 0.0 to 1.0\n\n"
        "Also provide total_count, scene_description, and justification.\n\n"
        "Respond with valid JSON only matching this schema exactly:\n"
        '{"tools_detected": [{"tool_name": str, "category": str, '
        '"condition": str, "confidence_score": float}], '
        '"total_count": int, "scene_description": str, "justification": str}\n\n'
        "If no tools are visible, return tools_detected as [] and total_count as 0."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        max_tokens=2048,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Identify and classify every industrial tool in this image. "
                            "Assess each tool's condition carefully based on visible wear, "
                            "rust, damage, or deformation."
                        ),
                    },
                ],
            },
        ],
    )
    raw = resp.choices[0].message.content
    if not raw or not raw.strip():
        raise ValueError("Model returned empty content for image classification.")
    return schema.model_validate_json(raw).model_dump()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def classify_document(text: str) -> dict:
    """Classify a business document into Invoice, Report, Contract, or Manual.
    Returns category, confidence_score (0-1), key_topics, and justification."""
    system = (
        "You are an expert document analyst. Classify the document into exactly one of: "
        "Invoice, Report, Contract, Manual. Respond with valid json only, matching this schema: "
        '{"category": str, "confidence_score": float, "key_topics": [str], "justification": str}. '
        'Example: {"category": "Invoice", "confidence_score": 0.94, '
        '"key_topics": ["billing", "amount due"], '
        '"justification": "Contains line items and a total amount due."}'
    )
    try:
        return _classify(system, text, DocumentResult)
    except Exception as e:
        return {"category": "Unknown", "confidence_score": 0.0,
                "key_topics": [], "justification": f"Classification error: {str(e)[:200]}"}


@mcp.tool()
def classify_sensor_data(readings: str) -> dict:
    """Classify sensor/time-series readings as Normal or Fault, and identify
    which sensor type the value belongs to (Temperature/Humidity/Moisture/Vibration).
    Works with labeled readings or plain numbers with no units.
    Returns sensor_type, state, confidence_score (0-1), indicators, and justification."""
    system = (
        "You are an expert industrial sensor analyst. Given a sensor reading (labeled or a plain number), "
        "perform TWO tasks:\n\n"

        "TASK 1 — Identify the sensor type using these EXACT value ranges:\n"
        "  • Temperature : 10 – 40  (degrees Celsius)\n"
        "  • Humidity    : 50 – 85  (percent %)\n"
        "  • Moisture    : 400 – 500 (raw ADC value, no unit)\n"
        "  • Vibration   : 800 – 900 (raw ADC value, no unit)\n"
        "If the input is labeled (e.g. 'temperature=92C'), trust that label over the ranges. "
        "If the value falls clearly within one range, use that sensor type. "
        "If it falls in multiple or none, pick the closest match and explain in justification.\n\n"

        "TASK 2 — Classify the state as exactly 'Normal' or 'Fault':\n"
        "  • Temperature: Normal = 20–35°C, Fault = outside this\n"
        "  • Humidity:    Normal = 55–80%, Fault = outside this\n"
        "  • Moisture:    Normal = 420–480 ADC, Fault = outside this\n"
        "  • Vibration:   Normal = 825–875 ADC, Fault = outside this\n"
        "  • For labeled readings (e.g. 'temperature=92C'), assess fault based on typical "
        "machine operating ranges.\n\n"

        "IMPORTANT: state MUST be exactly the string 'Normal' or 'Fault' — no other values.\n\n"

        "Respond with valid JSON only:\n"
        '{"sensor_type": str, "state": "Normal" or "Fault", "confidence_score": float (0-1), '
        '"indicators": [str], "justification": str}\n\n'

        "Examples:\n"
        '  Input "25.3"  → {"sensor_type":"Temperature","state":"Normal","confidence_score":0.95,'
        '"indicators":["25.3 falls in Temperature range 10-40"],"justification":"Value 25.3 matches temperature range and is within normal operating bounds."}\n'
        '  Input "65.1"  → {"sensor_type":"Humidity","state":"Normal","confidence_score":0.93,'
        '"indicators":["65.1 falls in Humidity range 50-85"],"justification":"Value 65.1 matches humidity range and is within normal bounds."}\n'
        '  Input "446.0" → {"sensor_type":"Moisture","state":"Normal","confidence_score":0.91,'
        '"indicators":["446 falls in Moisture ADC range 400-500"],"justification":"Raw ADC value 446 is a normal moisture reading."}\n'
        '  Input "851.0" → {"sensor_type":"Vibration","state":"Normal","confidence_score":0.94,'
        '"indicators":["851 falls in Vibration ADC range 800-900"],"justification":"Raw ADC value 851 is within normal vibration range."}\n'
        '  Input "98.6"  → {"sensor_type":"Temperature","state":"Fault","confidence_score":0.88,'
        '"indicators":["98.6 is in temperature range","98.6°C far exceeds normal 20-35°C"],"justification":"98.6 falls in temperature range but is well above the safe operating maximum of 35°C."}\n'
        '  Input "temperature=92C, vibration=8.4mm/s" → {"sensor_type":"Temperature/Vibration",'
        '"state":"Fault","confidence_score":0.91,"indicators":["92°C above normal 20-35°C","vibration 8.4mm/s high"],'
        '"justification":"Both labeled readings indicate fault conditions."}'
    )
    try:
        return _classify(system, readings, SensorResult)
    except Exception as e:
        return {"sensor_type": "Unknown", "state": "Normal", "confidence_score": 0.0,
                "indicators": ["could not parse input"],
                "justification": f"Classification error: {str(e)[:200]}"}


@mcp.tool()
def classify_network_packet(packet: str) -> dict:
    """Classify a network packet/log summary as Normal, Suspicious, or Priority.
    Returns classification, confidence_score (0-1), signals, and justification."""
    system = (
        "You are an expert network security analyst. Classify the traffic as exactly one of: "
        "Normal, Suspicious, Priority. Respond with valid json only: "
        '{"classification": str, "confidence_score": float, "signals": [str], "justification": str}. '
        'Example: {"classification": "Suspicious", "confidence_score": 0.88, '
        '"signals": ["SYN flood pattern", "no completed handshakes"], '
        '"justification": "High-rate SYN packets from many IPs indicate a possible DoS attempt."}'
    )
    try:
        return _classify(system, packet, NetworkResult)
    except Exception as e:
        return {"classification": "Normal", "confidence_score": 0.0,
                "signals": ["could not parse input"],
                "justification": f"Classification error: {str(e)[:200]}"}


@mcp.tool()
def classify_image_tools(image_b64: str, media_type: str = "image/jpeg") -> dict:
    """Detect and classify industrial tools in an image.
    Accepts a base64-encoded image (JPEG or PNG) and returns a list of all
    detected tools with their name, category, condition, and confidence score.
    Use this for any image input containing industrial tools."""
    try:
        return _vision_classify(image_b64, media_type, ImageToolResult)
    except Exception as e:
        return {"tools_detected": [], "total_count": 0,
                "scene_description": "Classification failed.",
                "justification": f"Error: {str(e)[:200]}"}


# ---------------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------------
@mcp.resource("info://server_status")
def get_server_status() -> str:
    return (
        f"ClassificationServer online. Model={MODEL} (Gemini OpenAI-compatible endpoint). "
        "Tools: classify_document, classify_sensor_data, "
        "classify_network_packet, classify_image_tools."
    )


if __name__ == "__main__":
    mcp.run()
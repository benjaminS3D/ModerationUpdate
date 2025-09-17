# moderation_agent.py
# LLM-only moderation agent for rendered 3D model images
# - No OCR, no heuristics
# - API selection per call: api_style="responses" or "chat"
# - Loads taxonomy & schema from files, and composes a strict system prompt
# - Always returns schema-shaped JSON (best-effort coercion if LLM deviates)

from __future__ import annotations
import base64
import io
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from PIL import Image
except ImportError:
    raise ImportError("Please install Pillow: pip install Pillow")

# Optional JSON Schema validation (recommended)
try:
    import jsonschema
    _SCHEMA_AVAILABLE = True
except Exception:
    _SCHEMA_AVAILABLE = False


@dataclass
class ModerationAgentConfig:
    base_path: str = "."                      # folder containing taxonomy.json, schema.json, vision_system_prompt.txt
    vision_model: Optional[str] = None
    validate_schema: bool = True
    max_image_px: int = 1536                  # downscale long side before encoding
    enforce_clear_over_ambiguous: bool = True # drops ambiguous flags when a clear one exists in same family


class ModerationAgent:
    """
    ModerationAgent(
        client=client,
        vision_model="gpt-4o-mini",
        config=ModerationAgentConfig(base_path="moderation_agent")
    )

    moderation_results = agent.moderate_images(
        image_dict,               # {id: "data:image/png;base64,...."}
        api_style="responses"     # or "chat"
    )
    """

    def __init__(self, client: Any = None, vision_model: Optional[str] = None, config: Optional[ModerationAgentConfig] = None):
        self.client = client
        self.config = config or ModerationAgentConfig()
        if vision_model is not None:
            self.config.vision_model = vision_model

        # Load supporting files
        self.taxonomy = self._load_json_file("taxonomy.json")
        self.schema = self._load_json_file("schema.json")
        self.base_system_prompt = self._load_text_file("vision_system_prompt.txt")
        self.taxonomy_version = self.taxonomy.get("taxonomy_version", "Secur3D-Taxonomy-v1.0")

        # Pre-render a compact taxonomy string for the prompt
        self.rendered_taxonomy = self._render_taxonomy(self.taxonomy)
        self.rendered_schema = json.dumps(self.schema, ensure_ascii=False)

    # ---------- Public API ----------

    def moderate_images(self, image_dict: Dict[str, str], api_style: str = "responses") -> Dict[str, Dict[str, Any]]:
        """
        api_style: "responses" or "chat" (binary, per call)
        """
        if api_style not in ("responses", "chat"):
            raise ValueError('api_style must be "responses" or "chat"')

        if not self.client or not self.config.vision_model:
            raise ValueError("LLM client and vision_model are required for LLM-only operation.")

        full_system_prompt = self._compose_system_prompt()

        out: Dict[str, Dict[str, Any]] = {}
        for img_id, data_url in image_dict.items():
            pil_img = self._decode_data_url_to_image(data_url)
            pil_img = self._maybe_downscale(pil_img, self.config.max_image_px)

            # Encode to base64 PNG for transport
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            user_instructions = (
                "Analyze the single image of a rendered 3D model. "
                "Follow the taxonomy, principles, and schema EXACTLY. Return ONLY the JSON object. "
                "Use pixel-space coordinates with origin at top-left."
            )

            # Call the selected API style
            if api_style == "responses":
                result_text = self._call_via_responses(full_system_prompt, user_instructions, img_b64)
            else:
                result_text = self._call_via_chat(full_system_prompt, user_instructions, img_b64)

            report = self._safe_parse_json(result_text) or self._empty_report()
            # Ensure required fields and taxonomy version
            report.setdefault("taxonomy_version", self.taxonomy_version)
            report.setdefault("summary", "No summary provided.")
            report.setdefault("flags", [])
            report.setdefault("text_fragments", [])

            # Enforce clear-over-ambiguous precedence (post-processing policy guard)
            if self.config.enforce_clear_over_ambiguous:
                report["flags"] = self._enforce_clear_over_ambiguous(report["flags"])

            # Validate or coerce
            if self.config.validate_schema and _SCHEMA_AVAILABLE:
                try:
                    jsonschema.validate(instance=report, schema=self.schema)
                except Exception as e:
                    report = self._coerce_to_schema(report, error=str(e))

            out[img_id] = report

        return out

    # ---------- LLM Calls ----------

    def _call_via_responses(self, system_prompt: str, user_instructions: str, img_b64_png: str) -> str:
        """
        Generic 'responses' style. We pass blocks with system text + user text + image bytes.
        Adjust if your SDK uses slightly different field names.
        """
        blocks = [
            {"type": "text", "text": system_prompt},
            {"type": "input_text", "text": user_instructions},
            {"type": "input_image", "image_data": img_b64_png, "mime_type": "image/png"}
        ]
        try:
            resp = self.client.responses.create(model=self.config.vision_model, input=blocks, temperature=0)
            # best-effort extraction
            if hasattr(resp, "output") and hasattr(resp.output, "text"):
                return "".join(resp.output.text)
            return str(resp)
        except Exception as e:
            return f'{{"taxonomy_version":"{self.taxonomy_version}","summary":"LLM error","flags":[],"text_fragments":[],"_error":"{type(e).__name__}"}}'

    def _call_via_chat(self, system_prompt: str, user_instructions: str, img_b64_png: str) -> str:
        """
        Generic 'chat' style. We pass a system message and a user message with text+image_url (data URL).
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_instructions},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64_png}"}}
                ]}
            ]
            resp = self.client.chat.completions.create(model=self.config.vision_model, messages=messages, temperature=0)
            if hasattr(resp, "choices") and resp.choices:
                m = resp.choices[0].message
                if hasattr(m, "content"):
                    return m.content
            return str(resp)
        except Exception as e:
            return f'{{"taxonomy_version":"{self.taxonomy_version}","summary":"LLM error","flags":[],"text_fragments":[],"_error":"{type(e).__name__}"}}'

    # ---------- Prompt assembly ----------

    def _compose_system_prompt(self) -> str:
        """Combine the base prompt + rendered taxonomy + schema into a single authoritative system message."""
        parts = [
            self.base_system_prompt.strip(),
            "\n\n# Taxonomy (authoritative)\n",
            self.rendered_taxonomy,
            "\n\n# Output JSON Schema (authoritative)\n",
            self.rendered_schema,
            "\n\n# Required behaviors\n",
            (
                "- Non-refusal: always return JSON.\n"
                "- Safety-first & multi-label: prefer false positives; flag all applicable categories.\n"
                "- Clear > Ambiguous precedence.\n"
                "- Verbatim text capture in text_fragments with pixel-space pointers.\n"
                "- Evidence required for each flag (type + region); coordinates in pixels (origin top-left).\n"
            )
        ]
        return "".join(parts)

    def _render_taxonomy(self, tx: Dict[str, Any]) -> str:
        """Render taxonomy.json into a compact, model-readable string."""
        lines = [f'taxonomy_version: "{tx.get("taxonomy_version", "Secur3D-Taxonomy-v1.0")}"', "categories:"]
        for cat in tx.get("categories", []):
            lines.append(f'- {cat["id"]}: {cat["name"]}')
            for sub in cat.get("subcategories", []):
                desc = sub.get("description", "").strip()
                lines.append(f'  - {sub["id"]}: {sub["name"]} — {desc}')
        return "\n".join(lines)

    # ---------- Decoding / utils ----------

    def _decode_data_url_to_image(self, data_url: str) -> Image.Image:
        if not data_url.startswith("data:"):
            raise ValueError("Expected data URL (data:*;base64,...)")
        try:
            b64 = data_url.split(",", 1)[1]
        except Exception:
            raise ValueError("Malformed data URL; missing comma separator.")
        img_bytes = base64.b64decode(b64, validate=False)
        return Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    def _maybe_downscale(self, img: Image.Image, max_side: int) -> Image.Image:
        w, h = img.size
        long_side = max(w, h)
        if long_side <= max_side:
            return img
        scale = max_side / float(long_side)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return img.resize(new_size, Image.LANCZOS)

    def _safe_parse_json(self, s: str) -> Optional[Dict[str, Any]]:
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            pass
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

    def _empty_report(self) -> Dict[str, Any]:
        return {"taxonomy_version": self.taxonomy_version, "summary": "No output.", "flags": [], "text_fragments": []}

    def _coerce_to_schema(self, report: Dict[str, Any], error: str = "") -> Dict[str, Any]:
        """Coerce best-effort into schema shape."""
        schema = self.schema
        out = {
            "taxonomy_version": str(report.get("taxonomy_version", self.taxonomy_version)),
            "summary": str(report.get("summary", "Autofixed to schema."))[:240],
            "flags": [],
            "text_fragments": []
        }

        # Flags
        for fl in report.get("flags", []):
            try:
                ev = fl.get("evidence", [])
                if not isinstance(ev, list) or not ev:
                    continue
                coerced_ev = []
                for e in ev:
                    region = (((e or {}).get("pointer") or {}).get("region") or {})
                    coerced_ev.append({
                        "type": e.get("type", "texture") if e.get("type") in {"texture", "geometry", "animation", "text", "metadata"} else "texture",
                        "pointer": {"region": {
                            "x": float(region.get("x", 0.0)),
                            "y": float(region.get("y", 0.0)),
                            "w": float(max(0.0, float(region.get("w", 1.0)))),
                            "h": float(max(0.0, float(region.get("h", 1.0))))
                        }}
                    })
                out["flags"].append({
                    "category_id": str(fl.get("category_id", "")),
                    "subcategory_id": str(fl.get("subcategory_id", "")),
                    "rationale": str(fl.get("rationale", "Reason not provided.")),
                    "evidence": coerced_ev,
                    "confidence": float(min(1.0, max(0.0, float(fl.get("confidence", 0.5))))),
                    "requires_human": bool(fl.get("requires_human", True))
                })
            except Exception:
                continue

        # Text fragments
        for t in report.get("text_fragments", []):
            try:
                p = t.get("pointer", {})
                out["text_fragments"].append({
                    "text": str(t.get("text", "")),
                    "pointer": {
                        "x": float(p.get("x", 0.0)),
                        "y": float(p.get("y", 0.0)),
                        "w": float(max(0.0, float(p.get("w", 0.0)))),
                        "h": float(max(0.0, float(p.get("h", 0.0))))
                    },
                    **({"lang": str(t.get("lang"))} if t.get("lang") else {})
                })
            except Exception:
                continue

        # Optional: validate again (ignore exceptions)
        if _SCHEMA_AVAILABLE:
            try:
                jsonschema.validate(instance=out, schema=schema)
            except Exception:
                pass
        return out

    def _enforce_clear_over_ambiguous(self, flags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Drop ambiguous subcategories if a clear one exists in the same family.
        Families:
          - CA11 (ambiguous cultural) vs CR10 (clear cultural/religious)
          - HR12 (mild/ambiguous harassment) vs B13 (severe/clear)
          - I9.3 (ambiguous IP) vs I9.1/I9.2 (clear IP)
        """
        # If any CR10 subcat present, drop CA11.*
        has_cr10 = any(f.get("category_id") == "CR10" for f in flags)
        if has_cr10:
            flags = [f for f in flags if f.get("category_id") != "CA11"]

        # If any B13 present, drop HR12.*
        has_b13 = any(f.get("category_id") == "B13" for f in flags)
        if has_b13:
            flags = [f for f in flags if f.get("category_id") != "HR12"]

        # If I9.1 or I9.2 present, drop I9.3
        has_i9_clear = any((f.get("category_id") == "I9" and f.get("subcategory_id") in ("I9.1", "I9.2")) for f in flags)
        if has_i9_clear:
            flags = [f for f in flags if not (f.get("category_id") == "I9" and f.get("subcategory_id") == "I9.3")]

        return flags

    # ---------- file I/O ----------

    def _load_json_file(self, name: str) -> Dict[str, Any]:
        path = f"{self.config.base_path.rstrip('/')}/{name}"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_text_file(self, name: str) -> str:
        path = f"{self.config.base_path.rstrip('/')}/{name}"
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

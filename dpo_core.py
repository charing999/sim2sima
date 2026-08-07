import json
import time
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional, Union

import torch
import sima_sft

# ============================================================
# DPO Core (v2.2) - Reasons in Korean + UI-friendly meta
# ============================================================

DRONE_ID_DEFAULT = "drone_1"

# 파싱/스키마 실패 시 떨어질 안전 기본값.
# ORBIT은 선회 = 능동 기동이라 "모델 출력을 이해하지 못한 상태"의 기본값으로는 부적절하다.
SAFE_FALLBACK_ACTION = "HOLD"

# ----- Action schema & ranges -----
ACTION_SPECS: Dict[str, Dict[str, Any]] = {
    "ORBIT": {
        "required": ["radius_m", "angular_speed_dps", "turn_rate_limit_dps"],
        "ranges": {
            "radius_m": (150, 600),
            "angular_speed_dps": (10, 60),
            "turn_rate_limit_dps": (20, 90),
        },
    },
    "CHASE": {
        "required": ["desired_distance_m", "desired_speed_mps", "turn_rate_limit_dps"],
        "ranges": {
            "desired_distance_m": (300, 2000),
            "desired_speed_mps": (5, 25),
            "turn_rate_limit_dps": (20, 90),
        },
    },
    "INTERCEPT": {
        "required": ["target_lead_time_s", "desired_speed_mps", "turn_rate_limit_dps"],
        "ranges": {
            "target_lead_time_s": (1, 6),
            "desired_speed_mps": (5, 25),
            "turn_rate_limit_dps": (20, 90),
        },
    },
    "RETREAT": {
        "required": ["retreat_distance_m", "desired_speed_mps", "turn_rate_limit_dps"],
        "ranges": {
            "retreat_distance_m": (150, 800),
            "desired_speed_mps": (5, 25),
            "turn_rate_limit_dps": (20, 90),
        },
    },
    "PATROL": {
        "required": ["intent", "desired_speed_mps", "turn_rate_limit_dps"],
        "ranges": {
            "desired_speed_mps": (5, 25),
            "turn_rate_limit_dps": (20, 90),
            "target_lead_time_s": (1, 6),
        },
        "intent_enum": ["patrol", "recon", "intercept"],
    },
}


# ============================================================
# Robust JSON Extraction Helpers
# ============================================================

def _try_json_loads(s: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(s), None
    except Exception as e:
        return None, str(e)


def _extract_codefence(text: str) -> Optional[str]:
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _extract_first_balanced(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_text(raw: Union[str, Dict[str, Any], List[Any], None]) -> Tuple[Optional[str], str]:
    if raw is None:
        return None, "none"
    if not isinstance(raw, str):
        return json.dumps(raw, ensure_ascii=False), "already_obj"
    text = raw.strip()

    obj, _ = _try_json_loads(text)
    if obj is not None:
        if isinstance(obj, dict) and "response" in obj:
            return extract_json_text(obj["response"])
        return text, "direct"

    cf = _extract_codefence(text)
    if cf:
        obj2, _ = _try_json_loads(cf)
        if obj2 is not None:
            return cf, "codefence"

    cand = _extract_first_balanced(text, "{", "}")
    if cand:
        obj4, _ = _try_json_loads(cand)
        if obj4 is not None:
            if isinstance(obj4, dict) and "response" in obj4:
                return extract_json_text(obj4["response"])
            return cand, "balanced_obj"
    return None, "extract_fail"


def parse_candidate_robust(raw_text: Union[str, Any]) -> Tuple[Optional[Any], str, Optional[str]]:
    jt, src = extract_json_text(raw_text)
    if jt is None:
        return None, src, "no_json"
    obj, err = _try_json_loads(jt)
    if err is not None:
        return None, src, "json_decode_fail"
    return obj, src, None


def coerce_actions(obj: Any) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    if obj is None:
        return None, {"reason": "no_obj"}
    if isinstance(obj, dict):
        if "response" in obj:
            return coerce_actions(obj["response"])
        drone_keys = [k for k in obj.keys() if re.fullmatch(r"drone_\d+", str(k))]
        if drone_keys:
            out: Dict[str, Dict[str, Any]] = {}
            for dk in drone_keys:
                v = obj[dk]
                if isinstance(v, str):
                    out[dk] = {"action": v, "params": {}}
                elif isinstance(v, dict):
                    act = v.get("action") or v.get("mode") or v.get("policy")
                    params = v.get("params") or v.get("parameters") or {}
                    if not isinstance(params, dict):
                        params = {}
                    out[dk] = {"action": act, "params": params}
            return out, {"reason": None}
        if "action" in obj and "params" in obj:
            return {DRONE_ID_DEFAULT: {"action": obj["action"], "params": obj["params"]}}, {"reason": "wrapped_single_action"}
    return None, {"reason": "dict_no_drone_keys"}


# ============================================================
# L1: State Builder
# ============================================================

def build_dpo_state(drone_lat, drone_lng, target_dist, target_bearing):
    return {
        "drone_id": DRONE_ID_DEFAULT,
        "timestamp": time.time(),
        "location": {"lat": float(drone_lat), "lng": float(drone_lng)},
        "target_info": {
            "dist": round(float(target_dist), 1),
            "bearing": round(float(target_bearing), 1) if not isinstance(target_bearing, str) else target_bearing,
            "closing_rate": 0.0,
        },
        "self_status": {"speed_kmh": 60.0, "battery": 85.0},
    }


# ============================================================
# Backward Compatible Wrapper for parse_action_and_params
# ============================================================

def parse_action_and_params(raw_text: str, drone_id: str = DRONE_ID_DEFAULT) -> Dict[str, Any]:
    obj, src, err = parse_candidate_robust(raw_text)
    coerced, info = coerce_actions(obj)

    warnings: List[str] = []
    if coerced is None:
        return {
            "parse_ok": False,
            "action": SAFE_FALLBACK_ACTION,
            "params": {},
            "normalized": {drone_id: {"action": SAFE_FALLBACK_ACTION, "params": {}}},
            "error": err or info.get("reason", "unknown"),
            "warnings": ["extract_failed"],
        }

    d_info = coerced.get(drone_id) or list(coerced.values())[0]
    action = str(d_info.get("action") or SAFE_FALLBACK_ACTION).upper().strip()
    params = d_info.get("params", {})

    if action not in ACTION_SPECS:
        warnings.append(f"unknown_action:{action}")
        parse_ok = False
    else:
        spec = ACTION_SPECS[action]
        missing = [k for k in spec.get("required", []) if k not in params]
        if missing:
            warnings.append("missing:" + ",".join(missing))
        if action == "PATROL" and str(params.get("intent", "")).lower().strip() == "intercept" and "target_lead_time_s" not in params:
            warnings.append("missing:target_lead_time_s")
        parse_ok = (len(warnings) == 0)

    return {
        "parse_ok": parse_ok,
        "action": action,
        "params": params,
        "normalized": coerced,
        "error": "" if parse_ok else "spec_violation",
        "warnings": warnings,
    }


# ============================================================
# L4: Candidate Generator (Using SFT Model)
# ============================================================

def _format_ui_summary(action: str, params: Dict[str, Any], parse_ok: bool, warnings: List[str]) -> str:
    if not parse_ok:
        return f"{action} (!)"
    return f"{action}"


def generate_candidates(state: Dict[str, Any], n: int = 3) -> List[Dict[str, Any]]:
    if sima_sft.SFT_MODEL is None or sima_sft.SFT_TOKENIZER is None:
        return []

    prompt = None
    if hasattr(sima_sft, "build_prompt_json"):
        try:
            prompt = sima_sft.build_prompt_json(state, profile="AUTO")
        except Exception:
            prompt = None
    if prompt is None:
        dist = float(state["target_info"]["dist"])
        prompt = sima_sft.build_prompt(state["location"]["lat"], state["location"]["lng"], dist)

    inputs = sima_sft.SFT_TOKENIZER(prompt, return_tensors="pt").to(sima_sft.SFT_MODEL.device)
    candidates: List[Dict[str, Any]] = []

    try:
        with torch.inference_mode():
            out = sima_sft.SFT_MODEL.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.85,
                top_p=0.9,
                num_return_sequences=n,
                pad_token_id=sima_sft.SFT_TOKENIZER.pad_token_id,
            )

        for i in range(n):
            seq = out[i]
            decoded = sima_sft.SFT_TOKENIZER.decode(seq, skip_special_tokens=True)
            if decoded.lstrip().startswith(prompt):
                raw_text = decoded[len(prompt) :].lstrip()
            else:
                raw_text = decoded.replace(prompt, "").strip()

            parsed = parse_action_and_params(raw_text, drone_id=state.get("drone_id", DRONE_ID_DEFAULT))
            candidates.append(
                {
                    "candidate_id": i,
                    "raw_text": raw_text,
                    **parsed,
                    "ui_summary": _format_ui_summary(parsed["action"], parsed["params"], parsed["parse_ok"], parsed["warnings"]),
                }
            )
    except Exception as e:
        print(f"[DPO] Gen Error: {e}")
        candidates.append(
            {
                "candidate_id": 0,
                "raw_text": "",
                "parse_ok": False,
                "action": SAFE_FALLBACK_ACTION,
                "params": {},
                "normalized": {state.get("drone_id", DRONE_ID_DEFAULT): {"action": SAFE_FALLBACK_ACTION, "params": {}}},
                "warnings": ["error"],
                "ui_summary": "Error",
            }
        )

    return candidates


# ============================================================
# L5: Judge & Logger (Korean reasons)
# ============================================================

def _reason_kr_parse(parse_ok: bool) -> Tuple[float, str]:
    if parse_ok:
        return 2.0, "JSON 파싱 및 필수 파라미터 검증 통과 (+2.0)"
    return -5.0, "JSON 파싱/스키마 불일치로 감점 (-5.0)"


def _reason_kr_distance(dist: float, action: str) -> Tuple[float, str]:
    a = str(action).upper().strip()
    if dist < 300:
        if a == "RETREAT":
            return 3.0, f"거리 {dist:.1f}m < 300m: 근접 위험이라 RETREAT 선호 (+3.0)"
        return -2.0, f"거리 {dist:.1f}m < 300m: RETREAT가 아니라서 감점 (-2.0)"
    if dist > 2000:
        if a in ("CHASE", "PATROL"):
            return 2.0, f"거리 {dist:.1f}m > 2000m: 접근/정찰({a}) 선호 (+2.0)"
        return -1.0, f"거리 {dist:.1f}m > 2000m: {a}은 비효율이라 감점 (-1.0)"
    # 300 ~ 2000
    if a in ("ORBIT", "CHASE", "PATROL"):
        return 1.0, f"거리 {dist:.1f}m (중거리): {a} 허용/가산 (+1.0)"
    return 0.0, f"거리 {dist:.1f}m (중거리): {a}는 특별 가산 없음 (+0.0)"


def judge_candidates(
    candidates: List[Dict[str, Any]],
    state: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    """Select best and worst candidate; also returns log entry with UI meta."""

    dist = float(state["target_info"]["dist"])

    scored: List[Dict[str, Any]] = []
    for c in candidates:
        action = c.get("action", SAFE_FALLBACK_ACTION)
        parse_ok = bool(c.get("parse_ok", False))

        score = 0.0
        reasons_kr: List[str] = []

        d1, r1 = _reason_kr_parse(parse_ok)
        score += d1
        reasons_kr.append(r1)

        d2, r2 = _reason_kr_distance(dist, str(action))
        score += d2
        reasons_kr.append(r2)

        # (optional) warnings become an explicit note
        warns = c.get("warnings", [])
        if warns:
            reasons_kr.append("경고: " + ", ".join(map(str, warns)))

        c["score"] = score
        c["reasons_kr"] = reasons_kr
        scored.append(c)

    scored.sort(key=lambda x: x.get("score", -1e9), reverse=True)
    chosen = scored[0]
    rejected = scored[-1]

    if float(chosen.get("score", 0.0)) <= float(rejected.get("score", 0.0)):
        return chosen, rejected, None

    # 1등마저 파싱에 실패했다면 후보 전체가 쓰레기다.
    # 이 쌍을 학습 데이터로 남기면 "덜 나쁜 쓰레기"를 선호하도록 배운다.
    if not bool(chosen.get("parse_ok", False)):
        return chosen, rejected, None

    prompt = None
    if hasattr(sima_sft, "build_prompt_json"):
        try:
            prompt = sima_sft.build_prompt_json(state, profile="AUTO")
        except Exception:
            prompt = None

    chosen_json = json.dumps(chosen.get("normalized", {}), ensure_ascii=False)
    rejected_json = json.dumps(rejected.get("normalized", {}), ensure_ascii=False)

    log_entry = {
        "prompt": prompt,
        "chosen": chosen_json,
        "rejected": rejected_json,
        "meta": {
            "ts": time.time(),
            "score_chosen": chosen.get("score"),
            "score_rejected": rejected.get("score"),
            "chosen_action": chosen.get("action"),
            "rejected_action": rejected.get("action"),
            "dist_m": dist,
            "chosen_reasons_kr": chosen.get("reasons_kr", []),
            "rejected_reasons_kr": rejected.get("reasons_kr", []),
            "candidates_scored": [
                {
                    "candidate_id": x.get("candidate_id"),
                    "action": x.get("action"),
                    "score": x.get("score"),
                    "reasons_kr": x.get("reasons_kr", []),
                    "parse_ok": bool(x.get("parse_ok", False)),
                    "warnings": x.get("warnings", []),
                }
                for x in scored
            ],
        },
    }

    return chosen, rejected, log_entry


def save_preference_log(entry: Optional[Dict[str, Any]], filepath: str = "dpo_preference_data_v2.jsonl") -> None:
    if not entry:
        return
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Log fail: {e}")

# sima_app.py - 경량화 버전
"""
SIMA 드론 전술 시뮬레이션 시스템 (Slim Version)
- geo_db.py: 지형 분석 (rasterio)
- sima_model.py: 로컬 LLM (Ollama)
"""

import json
import logging
import os
import queue
import re
import threading
import time
from typing import Any, Dict, Optional, Tuple

import folium
import requests
from flask import Flask, jsonify, render_template, request, send_from_directory
from folium import CustomIcon, Element
from PIL import Image, ImageDraw

from geo_db import get_terrain_analysis, init_analyzer  # 지형 분석 (Hybrid)
from sima_model import sima_chat, sima_chat_json  # 로컬 LLM
from sima_sft import init_sft_model # 모델 로더
from dpo_core_with_reasons_kr import build_dpo_state, generate_candidates, judge_candidates, save_preference_log  # DPO Core (KR reasons + UI meta)

# Flask 로그 억제
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


# ============================================================
# 상태(공유 메모리)
# ============================================================
SYSTEM_STATE_LOCK = threading.Lock()
SYSTEM_STATE: Dict[str, Any] = {
    "ENGAGEMENT_PERMISSION": False,
    "HUMAN_OVERRIDE": False,
    "CURRENT_MODE": "PATROL",
    "DPO_MODE": "AUTO",
    "IS_CHATTING": False,
    "LAST_LAT": 37.40,
    "LAST_LNG": 126.97,
    "LAST_DIST": None,
    "LAST_DIST1": None,
    "LAST_DIST2": None,
    "LAST_BEARING": None,
    "LAST_STATUS": "INIT",
    "LAST_TELEMETRY_TS": 0.0,
    "LAST_REGION_NAME": "",
    "LAST_REGION_TYPE": "",
    # --- DPO debug/UI ---
    "DPO_LAST_TS": 0.0,
    "DPO_LAST_DIST_M": None,
    "DPO_LAST_CHOSEN_ID": None,
    "DPO_LAST_CHOSEN_ACTION": None,
    "DPO_LAST_CHOSEN_SCORE": None,
    "DPO_LAST_CANDIDATES": [],  # UI-friendly scored table
    "DPO_LAST_FULL": [],        # full candidate objects for manual select
    "DPO_LAST_STATE": None,
    "DPO_LAST_PROMPT": None,
}


DRONE_COMMAND_QUEUE: "queue.Queue[str]" = queue.Queue()


# ============================================================
# 모드별 이동 맵
# ============================================================
ACTION_MAP = {
    "CHASE":   {"N": ["UP"], "S": ["DOWN"], "E": ["RIGHT"], "W": ["LEFT"],
                "NE": ["UP", "RIGHT"], "NW": ["UP", "LEFT"], "SE": ["DOWN", "RIGHT"], "SW": ["DOWN", "LEFT"]},
    "RETREAT": {"N": ["DOWN"], "S": ["UP"], "E": ["LEFT"], "W": ["RIGHT"],
                "NE": ["DOWN", "LEFT"], "NW": ["DOWN", "RIGHT"], "SE": ["UP", "LEFT"], "SW": ["UP", "RIGHT"]},
    "ORBIT":   {},  # ORBIT 모드는 JS 자율 비행 (서버 명령 없음)
}


# ============================================================
# 유닛 정보
# ============================================================
UNITS = {
    "TANK": "T-80급 전차",
    "TANK2": "T-72급 전차",
    "DRONE": "정찰용 드론",
}


# ============================================================
# Flask 앱
# ============================================================
app = Flask(__name__)


def ensure_dirs():
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)


def ensure_assets():
    """드론/탱크 아이콘 생성"""
    drone_path = os.path.join("static", "drone.png")
    tank_path = os.path.join("static", "tank.png")

    if not os.path.exists(drone_path):
        img = Image.new('RGBA', (48, 48), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.line((8, 8, 40, 40), fill='black', width=3)
        draw.line((8, 40, 40, 8), fill='black', width=3)
        for xy in [(2,2,14,14), (34,2,46,14), (2,34,14,46), (34,34,46,46)]:
            draw.ellipse(xy, fill='gray', outline='black')
        draw.ellipse((16, 16, 32, 32), fill='white', outline='blue', width=2)
        img.save(drone_path)

    if not os.path.exists(tank_path):
        img = Image.new('RGBA', (48, 48), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle((6, 4, 12, 44), fill='black')
        draw.rectangle((36, 4, 42, 44), fill='black')
        draw.rectangle((12, 8, 36, 40), fill='darkgreen')
        draw.ellipse((16, 18, 32, 34), fill='forestgreen', outline='black')
        draw.line((24, 26, 24, 0), fill='black', width=4)
        img.save(tank_path)


# ============================================================
# SHORT Fast-path (LLM 우회)
# ============================================================
def is_distance_command(msg: str) -> bool:
    m = (msg or "").lower()
    complex_kw = ["가장", "평균", "어디", "추천", "nearest", "average"]
    if any(k in m for k in complex_kw):
        return False
    keys = ["거리", "distance", "탱크", "t-80", "t-72", "bearing", "방위", "상태"]
    return any(k in m for k in keys)


def build_sensor_reply() -> str:
    with SYSTEM_STATE_LOCK:
        status = SYSTEM_STATE.get("LAST_STATUS")
        dist1 = SYSTEM_STATE.get("LAST_DIST1")
        dist2 = SYSTEM_STATE.get("LAST_DIST2")
        bearing = SYSTEM_STATE.get("LAST_BEARING")
        ts = float(SYSTEM_STATE.get("LAST_TELEMETRY_TS", 0.0))

    age = time.time() - ts if ts > 0 else 999.0
    ## kr : 드론이 목표 반경 벗어났을 때, 얼마나 강하게 다시 원래 궤도로 복귀시킬지 결정하는 값
    status_kr = {"DETECTED": "탐지됨", "SEARCHING": "탐색중", "INIT": "초기화"}.get(status, status)
    bearing_kr = {"N": "북", "S": "남", "E": "동", "W": "서", "NE": "북동", "NW": "북서", "SE": "남동", "SW": "남서"}.get(bearing, bearing)

    parts = [f"센서 갱신: {age:.1f}초 전"]
    if status: parts.append(f"상태: {status_kr}")
    if dist1 is not None: parts.append(f"T-80: {dist1:.1f}m")
    if dist2 is not None: parts.append(f"T-72: {dist2:.1f}m")
    if bearing: parts.append(f"방위: {bearing_kr}")

    return " / ".join(parts) if len(parts) > 1 else "텔레메트리 대기 중..."


# ============================================================
# 인사/잡담 처리 (LLM 우회)
# ============================================================
def handle_greeting(msg: str) -> Optional[str]:
    """인사/잡담 감지 시 간단 응답 반환, 아니면 None"""
    m = (msg or "").strip().lower()
    
    # 인사 패턴
    greetings = ["안녕", "하이", "헬로", "hello", "hi", "hey", "반가워", "반갑", "안녕하세요"]
    if any(g in m for g in greetings) and len(m) < 20:
        return "안녕하세요! SIMA 전술 지원 시스템입니다. 지형 분석이나 상황 보고가 필요하시면 말씀해 주세요."
    
    # 감사 패턴
    thanks = ["고마워", "감사", "땡큐", "thanks", "thank you", "ㄱㅅ"]
    if any(t in m for t in thanks) and len(m) < 20:
        return "천만에요! 추가 지원이 필요하시면 언제든 말씀해 주세요."
    
    return None
# ============================================================
# LLM 브리핑 (JSON 정답지 기반)
# ============================================================
def format_briefing_llm(user_msg: str) -> str:
    """지형 정답지 + 센서 정보 → LLM 브리핑"""
    import json
    from datetime import datetime

    with SYSTEM_STATE_LOCK:
        lat = SYSTEM_STATE.get("LAST_LAT", 37.40)
        lng = SYSTEM_STATE.get("LAST_LNG", 126.97)
        dist1 = SYSTEM_STATE.get("LAST_DIST1")
        dist2 = SYSTEM_STATE.get("LAST_DIST2")
        bearing = SYSTEM_STATE.get("LAST_BEARING")

    # 지형 분석
    try:
        terrain_context = get_terrain_analysis(lat, lng, radius_m=500)
    except Exception as e:
        terrain_context = {"error": str(e)}

    # 센서 정보 추가
    terrain_context["sensor"] = {
        "tank1_dist_m": dist1,
        "tank2_dist_m": dist2,
        "bearing": bearing
    }

    # 정답지 JSON 파일 저장
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"terrain_{timestamp}.json")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(terrain_context, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"정답지 저장 실패: {e}")

    system_instruction = (
        "무조건 한국어로 답변 해. "
        "Role: You are a Tactical Terrain Analyst AI. "
        "You must answer the user's request based ONLY on the provided 'Reference Context' (JSON).\n"
        "Do not hallucinate or fetch external information.\n\n"
        "## Response Guidelines\n"
        "1. **Analyze**: First, analyze the provided JSON context (Location, LULC, Elevation, Sensor data).\n"
        "2. **Reason**: Connect the data points to the user's tactical question.\n"
        "3. **Answer**: Provide a structured, step-by-step tactical briefing.\n"
        "   - Start with a direct answer or summary.\n"
        "   - Cite specific values (e.g., 'Elevation: 120m', 'Distance: 500m') to support your point.\n"
        "   - If data is missing or low quality, explicitly state it.\n\n"
    )

    prompt = f"""<start_of_turn>user
{system_instruction}

### Reference Context (Ground Truth)
```json
{json.dumps(terrain_context, ensure_ascii=False, indent=2)}
```

### User Request
{user_msg}

### Instruction
Please provide a step-by-step tactical analysis based on the context above.
<end_of_turn>
<start_of_turn>model
"""
    return sima_chat(prompt, max_tokens=1024)


# ============================================================
# Flask Routes
# ============================================================
@app.route("/toggle_dpo_mode", methods=["POST"])
def toggle_dpo_mode():
    with SYSTEM_STATE_LOCK:
        cur = SYSTEM_STATE.get("DPO_MODE", "AUTO")
        new = "MANUAL" if cur == "AUTO" else "AUTO"
        SYSTEM_STATE["DPO_MODE"] = new

        # 보통 MANUAL이면 사람 개입(override) 켜고,
        # AUTO면 override 끄는 식으로 연결합니다.
        SYSTEM_STATE["HUMAN_OVERRIDE"] = (new == "MANUAL")

        return jsonify({
            "ok": True,
            "dpo_mode": new,
            "override": SYSTEM_STATE["HUMAN_OVERRIDE"],
            "current_mode": SYSTEM_STATE.get("CURRENT_MODE", "PATROL"),
        })


@app.route("/dpo_last")
def dpo_last():
    """UI용: 최근 후보 점수표를 반환"""
    with SYSTEM_STATE_LOCK:
        return jsonify({
            "ok": True,
            "dpo_mode": SYSTEM_STATE.get("DPO_MODE", "AUTO"),
            "override": SYSTEM_STATE.get("HUMAN_OVERRIDE", False),
            "last_ts": float(SYSTEM_STATE.get("DPO_LAST_TS", 0.0) or 0.0),
            "last": {
                "ts": float(SYSTEM_STATE.get("DPO_LAST_TS", 0.0) or 0.0),
                "dist_m": SYSTEM_STATE.get("DPO_LAST_DIST_M"),
                "chosen_id": SYSTEM_STATE.get("DPO_LAST_CHOSEN_ID"),
                "chosen_action": SYSTEM_STATE.get("DPO_LAST_CHOSEN_ACTION"),
                "chosen_score": SYSTEM_STATE.get("DPO_LAST_CHOSEN_SCORE"),
                "candidates": SYSTEM_STATE.get("DPO_LAST_CANDIDATES", []),
            }
        })


@app.route("/dpo_select", methods=["POST"])
def dpo_select():
    """MANUAL 모드에서 사용자가 후보를 선택해 preference log로 저장 + 행동 반영"""
    try:
        payload = request.get_json(force=True, silent=True) or {}
        cid = int(payload.get("candidate_id"))
    except Exception:
        return jsonify({"ok": False, "error": "candidate_id가 올바르지 않습니다."})

    with SYSTEM_STATE_LOCK:
        if SYSTEM_STATE.get("DPO_MODE") != "MANUAL" or not SYSTEM_STATE.get("HUMAN_OVERRIDE"):
            return jsonify({"ok": False, "error": "MANUAL 모드에서만 선택할 수 있습니다. (DPO 모드 토글 필요)"})

        full = SYSTEM_STATE.get("DPO_LAST_FULL") or []
        state = SYSTEM_STATE.get("DPO_LAST_STATE")
        prompt = SYSTEM_STATE.get("DPO_LAST_PROMPT")

    if not full or not state:
        return jsonify({"ok": False, "error": "선택할 후보 데이터가 없습니다. (탱크가 2km 이내로 들어와야 생성됨)"})

    chosen = None
    for c in full:
        if int(c.get("candidate_id", -1)) == cid:
            chosen = c
            break
    if chosen is None:
        return jsonify({"ok": False, "error": f"candidate_id={cid} 후보를 찾을 수 없습니다."})

    # rejected: 나머지 중 최저 점수 (없으면 저장 불가)
    others = [c for c in full if int(c.get("candidate_id", -1)) != cid]
    if not others:
        return jsonify({"ok": False, "error": "후보가 1개뿐이라 preference pair를 만들 수 없습니다."})
    rejected = sorted(others, key=lambda x: float(x.get("score", -1e9)))[0]

    # preference log entry 구성
    try:
        import json as _json
        chosen_json = _json.dumps(chosen.get("normalized", {}), ensure_ascii=False)
        rejected_json = _json.dumps(rejected.get("normalized", {}), ensure_ascii=False)

        # UI-friendly scored table (이미 judge에서 계산됨)
        candidates_scored = []
        for c in full:
            candidates_scored.append({
                "candidate_id": c.get("candidate_id"),
                "action": c.get("action"),
                "score": c.get("score"),
                "reasons": c.get("reasons", []),
                "reasons_kr": c.get("reasons_kr", []),
                "parse_ok": c.get("parse_ok", False),
                "warnings": c.get("warnings", []),
            })

        log_entry = {
            "prompt": prompt,
            "chosen": chosen_json,
            "rejected": rejected_json,
            "meta": {
                "ts": time.time(),
                "user_selected": True,
                "score_chosen": chosen.get("score"),
                "score_rejected": rejected.get("score"),
                "chosen_action": chosen.get("action"),
                "rejected_action": rejected.get("action"),
                "dist_m": state.get("target_info", {}).get("dist"),
                "chosen_reasons": chosen.get("reasons", []),
                "rejected_reasons": rejected.get("reasons", []),
                "chosen_reasons_kr": chosen.get("reasons_kr", []),
                "rejected_reasons_kr": rejected.get("reasons_kr", []),
                "candidates_scored": candidates_scored,
            }
        }
        save_preference_log(log_entry)
    except Exception as e:
        return jsonify({"ok": False, "error": f"로그 저장 실패: {e}"})

    # 행동 반영 (기존 매핑 로직과 동일)
    action = str(chosen.get("action", "ORBIT")).upper().strip()
    params = chosen.get("params", {}) if isinstance(chosen.get("params", {}), dict) else {}

    # Map higher-level actions to executable mode in the current web-only controller
    if action == "INTERCEPT":
        mode = "CHASE"
    elif action == "PATROL":
        intent = str(params.get("intent", "patrol")).lower().strip()
        mode = "CHASE" if intent == "intercept" else "ORBIT"
    else:
        mode = action

    with SYSTEM_STATE_LOCK:
        SYSTEM_STATE["DPO_LAST_CHOSEN_ID"] = cid
        SYSTEM_STATE["DPO_LAST_CHOSEN_ACTION"] = action
        SYSTEM_STATE["DPO_LAST_CHOSEN_SCORE"] = chosen.get("score")
        SYSTEM_STATE["DPO_LAST_ACTION"] = action
        SYSTEM_STATE["DPO_LAST_PARAMS"] = params
        SYSTEM_STATE["CURRENT_MODE"] = mode

    return jsonify({"ok": True, "applied_mode": mode, "chosen_action": action, "candidate_id": cid})


@app.route("/manual_command", methods=["POST"])
def manual_command():
    """UI 버튼용 직접 제어 명령"""
    try:
        payload = request.get_json(force=True, silent=True) or {}
        action = str(payload.get("action", "")).upper()
    except Exception:
        return jsonify({"ok": False, "error": "Invalid payload"})

    # ACTION_MAP에 정의된 모드인지 확인
    # INTERCEPT/PATROL은 편의상 허용하되 실제 구동 모드(CHASE/ORBIT)로 매핑
    valid_actions = ["CHASE", "RETREAT", "ORBIT", "HOLD", "PATROL", "INTERCEPT"]
    if action not in valid_actions:
        return jsonify({"ok": False, "error": f"Unknown action: {action}"})

    mode = action
    if action == "INTERCEPT":
        mode = "CHASE"
    elif action == "PATROL":
        mode = "ORBIT"  # 기본 정찰은 선회

    with SYSTEM_STATE_LOCK:
        SYSTEM_STATE["HUMAN_OVERRIDE"] = True
        SYSTEM_STATE["CURRENT_MODE"] = mode
        # DPO 로깅용 더미 상태 업데이트 (선택 사항)
        SYSTEM_STATE["DPO_LAST_ACTION"] = action
        SYSTEM_STATE["DPO_LAST_PARAMS"] = {"source": "manual_button"}

    return jsonify({"ok": True, "mode": mode, "action": action})


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'drone.png', mimetype='image/vnd.microsoft.icon')


@app.route("/map_view")
def map_view():
    return render_template("sim_map.html")


@app.route("/status")
def status():
    with SYSTEM_STATE_LOCK:
        return jsonify({
            "mode": SYSTEM_STATE["CURRENT_MODE"],
            "permission": SYSTEM_STATE["ENGAGEMENT_PERMISSION"],
            "override": SYSTEM_STATE["HUMAN_OVERRIDE"],
            "lat": SYSTEM_STATE["LAST_LAT"],
            "lng": SYSTEM_STATE["LAST_LNG"],
            "region_name": SYSTEM_STATE.get("LAST_REGION_NAME", "-"),
            "region_type": SYSTEM_STATE.get("LAST_REGION_TYPE", "-"),
        })


@app.route("/get_drone_command")
def get_drone_command():
    commands = []
    try:
        while not DRONE_COMMAND_QUEUE.empty():
            commands.append(DRONE_COMMAND_QUEUE.get_nowait())
    except queue.Empty:
        pass
    return jsonify({"commands": commands})


@app.route("/telemetry", methods=["POST"])
def telemetry():
    try:
        lat = float(request.form.get("lat"))
        lng = float(request.form.get("lng"))
        dist = float(request.form.get("dist")) if request.form.get("dist") else None
        bearing = request.form.get("bearing")
        status_val = request.form.get("status") or "INIT"
        dist1 = float(request.form.get("dist1")) if request.form.get("dist1") else None
        dist2 = float(request.form.get("dist2")) if request.form.get("dist2") else None

        with SYSTEM_STATE_LOCK:
            SYSTEM_STATE["LAST_LAT"] = lat
            SYSTEM_STATE["LAST_LNG"] = lng
            SYSTEM_STATE["LAST_DIST"] = dist
            SYSTEM_STATE["LAST_DIST1"] = dist1
            SYSTEM_STATE["LAST_DIST2"] = dist2
            SYSTEM_STATE["LAST_BEARING"] = bearing
            SYSTEM_STATE["LAST_STATUS"] = status_val
            SYSTEM_STATE["LAST_TELEMETRY_TS"] = time.time()

            if not SYSTEM_STATE["HUMAN_OVERRIDE"] and dist is not None:
                SYSTEM_STATE["ENGAGEMENT_PERMISSION"] = (dist <= 2000.0)
    except Exception:
        pass
    return "OK"


@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.form.get("message", "") or ""

    # 1. 인사/잡담 (LLM 우회)
    greeting_reply = handle_greeting(user_msg)
    if greeting_reply:
        return jsonify({"response": greeting_reply})

    # 2. 센서 보고 (LLM 우회)
    if is_distance_command(user_msg.lower()) or any(k in user_msg.lower() for k in ["상황", "보고"]):
        return jsonify({"response": build_sensor_reply()})

    # 3. LLM 브리핑
    try:
        response = format_briefing_llm(user_msg)
        return jsonify({"response": response[:3000]})
    except Exception as e:
        return jsonify({"response": f"오류: {e}"})


# ============================================================
# 지도 생성
# ============================================================
def generate_index_html():
    html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SIMA Command</title>
<style>
:root{
  --bg:#000;
  --panel:#0d1117;
  --panel2:#161b22;
  --border:#30363d;
  --accent:#238636;
  --muted:#8b949e;
  --user:#1f6feb;
  --ai:#21262d;
  --warn:#f0b429;
  --danger:#ff6b6b;
}
body { margin:0; display:flex; height:90vh; background:var(--bg); color:#fff; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
#map-frame { width:50%; height:100%; border:none; border-right:2px solid #444; }
#commander-panel { width:50%; display:flex; flex-direction:column; background:var(--panel); }
#status-bar { padding:14px 16px; background:var(--panel2); border-bottom:3px solid var(--accent); display:flex; gap:14px; align-items:center; justify-content:space-between; }
#status-left{display:flex; gap:14px; align-items:center; flex-wrap:wrap;}
.badge{display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border:1px solid var(--border); border-radius:999px; font-size:12px; color:#fff; background:rgba(255,255,255,0.02);}
.badge .dot{width:8px; height:8px; border-radius:50%; background:var(--muted);}
.badge.good .dot{background:var(--accent);}
.badge.warn .dot{background:var(--warn);}
.badge.danger .dot{background:var(--danger);}
button { background:var(--accent); color:white; border:none; padding:10px 14px; border-radius:8px; cursor:pointer; font-weight:600; }
button.secondary{ background:#30363d; font-weight:500; }
button.small{ padding:7px 10px; font-size:12px; border-radius:8px; }
button.ghost{ background:transparent; border:1px solid var(--border); color:#fff; }

.btnrow { display:flex; gap:8px; padding:10px 16px; border-top:1px solid var(--border); flex-wrap:wrap; align-items:center;}
#main-split{display:flex; flex-direction:column; min-height:0; flex:1;}
#dpo-panel{ border-top:1px solid var(--border); padding:12px 16px; background:rgba(255,255,255,0.01); }
#dpo-head{display:flex; gap:10px; align-items:center; justify-content:space-between; margin-bottom:10px;}
#dpo-title{font-weight:700; letter-spacing:0.2px;}
#dpo-sub{color:var(--muted); font-size:12px;}
#dpo-cards{ display:grid; grid-template-columns:1fr; gap:10px; max-height:230px; overflow:auto; padding-right:4px; }
.card{ border:1px solid var(--border); border-radius:12px; background:rgba(255,255,255,0.02); padding:10px 12px; }
.card-top{display:flex; align-items:flex-start; justify-content:space-between; gap:10px;}
.card h4{margin:0; font-size:14px;}
.kv{display:flex; gap:8px; flex-wrap:wrap; align-items:center; color:var(--muted); font-size:12px; margin-top:4px;}
.pill{display:inline-flex; align-items:center; padding:2px 8px; border:1px solid var(--border); border-radius:999px; font-size:12px; color:#fff;}
.pill.good{border-color:rgba(35,134,54,0.8);}
.pill.bad{border-color:rgba(255,107,107,0.8);}
.reasons{margin:8px 0 0 0; padding-left:18px; color:#e6edf3; font-size:12px; line-height:1.4;}
.reasons li{margin:2px 0;}
.card-actions{display:flex; gap:8px; margin-top:10px; justify-content:flex-end;}
.hr{height:1px; background:var(--border); margin:10px 0;}
#chat-container { flex:1; overflow-y:auto; padding:14px 16px; display:flex; flex-direction:column; gap:10px; min-height:0;}
.msg { padding:10px 12px; border-radius:10px; max-width:95%; white-space:pre-wrap; line-height:1.35;}
.user { background:var(--user); align-self:flex-end; }
.ai { background:var(--ai); border:1px solid var(--border); align-self:flex-start; }
#input-area { padding:14px 16px; display:flex; gap:10px; border-top:1px solid var(--border); }
input { flex:1; padding:12px; background:#0d1117; color:white; border:1px solid var(--border); border-radius:10px; }
</style>
</head>
<body>
<iframe id="map-frame" src="/map_view"></iframe>

<div id="commander-panel">
  <div id="status-bar">
    <div id="status-left">
      <span class="badge" id="b-mode"><span class="dot"></span><span id="sys-mode">MODE: INIT</span></span>
      <span class="badge" id="b-pos"><span class="dot"></span><span id="sys-pos">POS: -</span></span>
      <span class="badge" id="b-dpo"><span class="dot"></span><span id="sys-dpo">DPO: -</span></span>
    </div>
    <div style="display:flex; gap:8px; align-items:center;">
      <button class="ghost small" onclick="toggleDPO()">DPO 모드 토글</button>
      <button class="secondary small" onclick="quick('상황 보고해')">상황 보고</button>
    </div>
  </div>

  <div id="main-split">
    <div id="dpo-panel">
      <div id="dpo-head">
        <div>
          <div id="dpo-title">후보 점수표</div>
          <div id="dpo-sub">최근 DPO 스텝의 후보별 점수 및 이유를 표시합니다.</div>
        </div>
        <button class="secondary small" onclick="refreshDPO(true)">새로고침</button>
      </div>
      <div id="dpo-cards">
        <div class="card"><div style="color:var(--muted); font-size:12px;">아직 후보 점수 데이터가 없습니다. (탱크가 2km 이내로 들어오면 생성됨)</div></div>
      </div>
    </div>

    <div id="chat-container">
      <div class="msg ai">SIMA 시스템 온라인</div>
    </div>

    <div id="input-area">
      <input type="text" id="txt" placeholder="명령 입력..."/>
      <button onclick="send()">전송</button>
    </div>
  </div>
</div>

<script>
let __lastDpoTs = 0;

function setBadge(el, kind){
  el.classList.remove('good','warn','danger');
  if(kind) el.classList.add(kind);
  const dot = el.querySelector('.dot');
  if(dot){
    dot.style.background = (kind==='good') ? 'var(--accent)' :
                          (kind==='warn') ? 'var(--warn)' :
                          (kind==='danger') ? 'var(--danger)' : 'var(--muted)';
  }
}

async function toggleDPO(){
  const res = await fetch('/toggle_dpo_mode', {method:'POST'});
  const data = await res.json();
  if(data && data.dpo_mode){
    refreshDPO(true);
  }
}

function quick(t){document.getElementById('txt').value=t;send();}

async function send(){
  const txt=document.getElementById('txt').value.trim();
  if(!txt)return;
  const chat=document.getElementById('chat-container');
  const u=document.createElement('div');u.className='msg user';u.textContent=txt;chat.appendChild(u);
  document.getElementById('txt').value='';
  const fd=new FormData();fd.append('message',txt);
  const res=await fetch('/chat',{method:'POST',body:fd});
  const data=await res.json();
  const a=document.createElement('div');a.className='msg ai';a.textContent=data.response;chat.appendChild(a);
  chat.scrollTop=chat.scrollHeight;
}

document.getElementById('txt').addEventListener('keypress',e=>{if(e.key==='Enter')send();});

function fmtTs(ts){
  if(!ts) return '-';
  const d = new Date(ts*1000);
  const hh = String(d.getHours()).padStart(2,'0');
  const mm = String(d.getMinutes()).padStart(2,'0');
  const ss = String(d.getSeconds()).padStart(2,'0');
  return `${hh}:${mm}:${ss}`;
}

function escapeHtml(s){
  return (s||'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
}

async function selectCandidate(cid){
  const res = await fetch('/dpo_select', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({candidate_id: cid})
  });
  const data = await res.json();
  if(data && data.ok){
    refreshDPO(true);
  }else{
    alert(data && data.error ? data.error : '선택 실패');
  }
}

function renderCards(payload){
  const cardsEl = document.getElementById('dpo-cards');
  cardsEl.innerHTML = '';

  const last = payload && payload.last ? payload.last : null;
  const cands = last && last.candidates ? last.candidates : [];
  if(!cands.length){
    const empty = document.createElement('div');
    empty.className = 'card';
    empty.innerHTML = `<div style="color:var(--muted); font-size:12px;">아직 후보 점수 데이터가 없습니다.</div>`;
    cardsEl.appendChild(empty);
    return;
  }

  const dpoMode = payload.dpo_mode || '-';
  const override = payload.override ? true : false;

  for(const c of cands){
    const card = document.createElement('div');
    card.className='card';

    const score = (typeof c.score === 'number') ? c.score.toFixed(2) : String(c.score ?? '-');
    const parseOk = !!c.parse_ok;
    const chosen = (last.chosen_id === c.candidate_id);

    const pillKind = parseOk ? 'good' : 'bad';
    const pillText = parseOk ? 'PARSE OK' : 'PARSE FAIL';

    const headHtml = `
      <div class="card-top">
        <div>
          <h4>#${c.candidate_id}  ${escapeHtml(c.action || '-')}</h4>
          <div class="kv">
            <span class="pill ${pillKind}">${pillText}</span>
            <span class="pill">${chosen ? 'SELECTED' : ' '}</span>
            <span style="color:var(--muted)">score: <b style="color:#fff">${score}</b></span>
          </div>
        </div>
        <div style="text-align:right; color:var(--muted); font-size:12px;">
          ${escapeHtml(c.warnings && c.warnings.length ? 'warn: ' + c.warnings.join(', ') : '')}
        </div>
      </div>
    `;

    const reasons = (c.reasons_kr && c.reasons_kr.length) ? c.reasons_kr : (c.reasons || []);
    let reasonsHtml = '';
    if(reasons && reasons.length){
      const lis = reasons.map(x=>`<li>${escapeHtml(String(x))}</li>`).join('');
      reasonsHtml = `<ul class="reasons">${lis}</ul>`;
    }else{
      reasonsHtml = `<div style="margin-top:8px; color:var(--muted); font-size:12px;">이유 없음</div>`;
    }

    let actionsHtml = '';
    // MANUAL + override일 때만 선택 버튼을 보여줌
    if(dpoMode === 'MANUAL' && override){
      actionsHtml = `
        <div class="card-actions">
          <button class="ghost small" onclick="selectCandidate(${c.candidate_id})">이 후보 선택</button>
        </div>
      `;
    }

    card.innerHTML = headHtml + reasonsHtml + actionsHtml;
    cardsEl.appendChild(card);
  }
}

async function refreshStatus(){
  const d = await fetch('/status').then(r=>r.json());
  document.getElementById('sys-mode').innerText = "MODE: " + d.mode;
  document.getElementById('sys-pos').innerText = "POS: " + d.lat.toFixed(4) + "," + d.lng.toFixed(4);
  setBadge(document.getElementById('b-mode'), d.mode === 'HOLD' ? 'warn' : 'good');
  setBadge(document.getElementById('b-pos'), 'good');
}

async function refreshDPO(force){
  const payload = await fetch('/dpo_last').then(r=>r.json());
  document.getElementById('sys-dpo').innerText = "DPO: " + (payload.dpo_mode || '-')
    + " / last: " + fmtTs(payload.last_ts || 0);

  setBadge(document.getElementById('b-dpo'), (payload.dpo_mode === 'MANUAL') ? 'warn' : 'good');

  const ts = payload.last_ts || 0;
  if(force || ts > __lastDpoTs){
    __lastDpoTs = ts;
    renderCards(payload);
  }
}

setInterval(()=>{ refreshStatus(); }, 800);
setInterval(()=>{ refreshDPO(false); }, 800);
</script>
</body>
</html>"""
    os.makedirs("templates", exist_ok=True)
    with open(os.path.join("templates", "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)


def generate_simulation_map():
    ensure_dirs()
    # 안양/인덕원 지역 (DSK_2026 래스터 범위)
    start_lat, start_lng = 37.40, 126.97
    end_lat, end_lng = 37.43, 127.00
    center_lat = (start_lat + end_lat) / 2
    center_lng = (start_lng + end_lng) / 2

    m = folium.Map(location=[center_lat, center_lng], zoom_start=11, control_scale=True)

    # HUD
    hud_html = """
    <div id="hud-panel" style="position:absolute;top:10px;left:50px;z-index:9999;background:#000;color:#fff;padding:16px;border:4px solid #fff;min-width:300px;font-family:monospace;font-size:20px;">
        <div style="border-bottom:2px solid #fff;margin-bottom:10px;">TACTICAL SENSOR</div>
        <div>STATUS: <span id="status-text" style="color:#0f0;">INIT</span></div>
        <div>DISTANCE: <span id="dist-text">0</span> m</div>
        <div>BEARING: <span id="bearing-text">N</span></div>
    </div>
    """
    m.get_root().html.add_child(Element(hud_html))

    # 마커
    drone_icon_path = os.path.join("static", "drone.png")
    tank_icon_path = os.path.join("static", "tank.png")
    drone_icon = CustomIcon(drone_icon_path, icon_size=(48, 48), icon_anchor=(24, 24)) if os.path.exists(drone_icon_path) else None
    tank_icon = CustomIcon(tank_icon_path, icon_size=(48, 48), icon_anchor=(24, 24)) if os.path.exists(tank_icon_path) else None
    drone = folium.Marker([start_lat + 0.02, start_lng + 0.02], icon=drone_icon).add_to(m)
    tank = folium.Marker([start_lat, start_lng], icon=tank_icon).add_to(m)
    tank2 = folium.Marker([start_lat + 0.015, start_lng - 0.01], icon=tank_icon).add_to(m)
    tank_dot = folium.CircleMarker([start_lat, start_lng], radius=8, color="red", fill=True).add_to(m)
    tank2_dot = folium.CircleMarker([start_lat + 0.015, start_lng - 0.01], radius=8, color="orange", fill=True).add_to(m)
    circle = folium.Circle([start_lat + 0.02, start_lng + 0.02], radius=2000, color="blue", fill=True, fill_opacity=0.10).add_to(m)

    js_script = f"""
    <script>
    function getBearing(lat1,lng1,lat2,lng2){{
        var y=Math.sin((lng2-lng1)*Math.PI/180)*Math.cos(lat2*Math.PI/180);
        var x=Math.cos(lat1*Math.PI/180)*Math.sin(lat2*Math.PI/180)-Math.sin(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.cos((lng2-lng1)*Math.PI/180);
        return(Math.atan2(y,x)*180/Math.PI+360)%360;
    }}
    function getCardinal(a){{return['N','NE','E','SE','S','SW','W','NW'][Math.round(a/45)%8];}}
    async function postTelemetry(lat,lng,dist,bearing,status,dist1,dist2){{
        try{{
            const body=new URLSearchParams({{lat:String(lat),lng:String(lng),dist:String(dist),bearing:String(bearing),status:String(status),dist1:String(dist1),dist2:String(dist2)}});
            await fetch('/telemetry',{{method:'POST',body}});
        }}catch(e){{}}
    }}
    window.droneTarget=null;window.droneSpeed=0.00003;window.autoTrack=true;

    // Orbit(원형 선회) 파라미터
    window.orbit = {{
      enabled: true,
      radius_m: 500,   // 목표물과 유지할 반경(예: 300~800m)
      vt_mps: 25,      // 접선 속도 (m/s) 예: 10~25
      clockwise: true, // 시계 방향
      max_mps: 30,      // 속도 상한(안정화)
      // 포화 복원 파라미터
      k: 20,        // 방사 복원 최대 강도 스케일(대략 m/s급)
      korbit: 6.0,  // 포화 민감도(클수록 빨리 포화)
    }};

    // 위경도 -> 로컬 미터 변환(근사, 작은 구역에서 충분)
    function metersPerDegLat() {{ return 111320.0; }}
    function metersPerDegLng(lat) {{ return 111320.0 * Math.cos(lat * Math.PI / 180); }}

    // (lat,lng) 차이를 meters (x=동, y=북)로
    function dLatLngToMeters(dLat, dLng, refLat) {{
      const mx = dLng * metersPerDegLng(refLat);
      const my = dLat * metersPerDegLat();
      return {{ x: mx, y: my }};
    }}

    // meters (x,y)를 (dLat,dLng)로
    function metersToDLatLng(x, y, refLat) {{
      const dLat = y / metersPerDegLat();
      const dLng = x / metersPerDegLng(refLat);
      return {{ dLat, dLng }};
    }}

    window.__lastAnimTs = null;
    window.__lastHudTs = 0;

    function animateDrone(ts) {{
        if (!window.__lastAnimTs) window.__lastAnimTs = ts;
        const dt = Math.min((ts - window.__lastAnimTs) / 1000.0, 0.05); // Limit 50ms
        window.__lastAnimTs = ts;

        if (window.droneObj && window.circleObj) {{
            const dPos = window.droneObj.getLatLng();
            let newLat = dPos.lat;
            let newLng = dPos.lng;
            let moved = false;

            // 1) MANUAL/QUEUE Mode priority
            if (window.droneTarget) {{
                const speed_mps = 20;
                const dLat = window.droneTarget.lat - dPos.lat;
                const dLng = window.droneTarget.lng - dPos.lng;
                const m = dLatLngToMeters(dLat, dLng, dPos.lat);
                const dist_m = Math.hypot(m.x, m.y);

                if (dist_m < 2.0) {{
                    window.droneTarget = null;
                }} else {{
                    const step = Math.min(speed_mps * dt, dist_m);
                    const ux = m.x / dist_m;
                    const uy = m.y / dist_m;
                    const dd = metersToDLatLng(ux * step, uy * step, dPos.lat);
                    newLat += dd.dLat;
                    newLng += dd.dLng;
                    moved = true;
                }}
            }}
            // 2) ORBIT Mode (Vector Field)
            else if (window.orbit && window.orbit.enabled && window.tankObj && window.tank2Obj) {{
                const tPos1 = window.tankObj.getLatLng();
                const tPos2 = window.tank2Obj.getLatLng();
                const dist1 = dPos.distanceTo(tPos1);
                const dist2 = dPos.distanceTo(tPos2);
                // 거리 계산
                const tPos = (dist1 <= dist2) ? tPos1 : tPos2;

                const dLat = dPos.lat - tPos.lat; // 위도 차이
                const dLng = dPos.lng - tPos.lng; // 경도 차이
                const v = dLatLngToMeters(dLat, dLng, tPos.lat); // 위도, 경도 차이를 실제 미터로 변환
                const r = Math.max(Math.hypot(v.x, v.y), 1e-3); // 중심까지의 현재 직선거리 r

                const er = r - window.orbit.radius_m; // 현재 거리 - 목표 500m 차이, 양수면 당겨야 하고 음수면 밀어내야 함.
                const cw = window.orbit.clockwise ? 1 : -1; // 시계방향이면 1, 반시계방향이면 -1
                
                // 방향 벡터를 계산하는 수식
                const urx = v.x / r, ury = v.y / r;  // 방사 벡터
                const utx = cw * ury, uty = cw * -urx; // 접선 벡터
                
                // 최종 속도 계산
                // 방사 방향 복원 속도 성분 (포화형)
                const vr = -window.orbit.k * Math.atan(window.orbit.korbit * (er / window.orbit.radius_m));

                // 최종 속도 = 방사 복원 + 접선 선회
                let vx = vr * urx + window.orbit.vt_mps * utx;
                let vy = vr * ury + window.orbit.vt_mps * uty;


                const sp = Math.hypot(vx, vy); // 현재 속도 크기
                const vmax = window.orbit.max_mps; // 최대 속도
                if (sp > vmax) {{ vx *= (vmax / sp); vy *= (vmax / sp); }} // 최대 속도를 유지하기 위해 정규화, 비율을 줄여 최대 속도에 맞춤.

                const dx = vx * dt;
                const dy = vy * dt;
                const dd = metersToDLatLng(dx, dy, dPos.lat);
                newLat += dd.dLat;
                newLng += dd.dLng;
                moved = true;
            }}

            if (moved) {{
                window.droneObj.setLatLng([newLat, newLng]);
                window.circleObj.setLatLng([newLat, newLng]);
            }}

            if (window.updateHUD && (!window.__lastHudTs || (ts - window.__lastHudTs) > 500)) {{
                window.__lastHudTs = ts;
                window.updateHUD(false);
            }}
        }}
        requestAnimationFrame(animateDrone);
    }}
    requestAnimationFrame(animateDrone);

    window.moveDrone=function(dir){{
        if(!window.droneObj||window.droneTarget)return;
        var dPos=window.droneObj.getLatLng();var step=0.001;
        if(dir==='UP')window.droneTarget={{lat:dPos.lat+step,lng:dPos.lng}};
        else if(dir==='DOWN')window.droneTarget={{lat:dPos.lat-step,lng:dPos.lng}};
        else if(dir==='LEFT')window.droneTarget={{lat:dPos.lat,lng:dPos.lng-step}};
        else if(dir==='RIGHT')window.droneTarget={{lat:dPos.lat,lng:dPos.lng+step}};
        window.updateHUD(true);
    }};
    // Use window 'load' to ensure Folium markers are initialized
    window.addEventListener("load", function(){{
        window.droneObj={drone.get_name()};window.tankObj={tank.get_name()};window.tank2Obj={tank2.get_name()};
        window.tankDotObj={tank_dot.get_name()};window.tank2DotObj={tank2_dot.get_name()};window.circleObj={circle.get_name()};
        window.updateHUD=function(force){{
            var dPos=window.droneObj.getLatLng();
            var tPos1=window.tankObj.getLatLng();var tPos2=window.tank2Obj.getLatLng();
            var dist1=dPos.distanceTo(tPos1);var dist2=dPos.distanceTo(tPos2);
            var tPos=(dist1<=dist2)?tPos1:tPos2;var dist=Math.min(dist1,dist2);
            var bearingCard=getCardinal(getBearing(dPos.lat,dPos.lng,tPos.lat,tPos.lng));
            var status=(dist<=2000)?"DETECTED":"SEARCHING";
            document.getElementById('status-text').innerText=status;
            document.getElementById('status-text').style.color=(dist<=2000)?"#FFFF00":"#00FF00";
            document.getElementById('dist-text').innerText=dist.toFixed(1);
            document.getElementById('bearing-text').innerText=bearingCard;
            window.circleObj.setStyle({{color:(dist<=2000)?'red':'blue'}});
            const now=Date.now();if(!window.__lastTeleTs)window.__lastTeleTs=0;
            if(force||(now-window.__lastTeleTs)>500){{window.__lastTeleTs=now;postTelemetry(dPos.lat,dPos.lng,dist,bearingCard,status,dist1,dist2);}}
        }};
        window.updateHUD(true);
        var curLat={start_lat};var curLng={start_lng};var curLat2={start_lat}+0.015;var curLng2={start_lng}-0.01;
        setInterval(function(){{
            curLat+=0.00002;curLng+=0.00001;curLat2+=0.00001;curLng2+=0.00001;
            window.tankObj.setLatLng([curLat,curLng]);window.tankDotObj.setLatLng([curLat,curLng]);
            window.tank2Obj.setLatLng([curLat2,curLng2]);window.tank2DotObj.setLatLng([curLat2,curLng2]);
            window.updateHUD(false);
        }},100);
        setInterval(function(){{
            fetch('/get_drone_command')
                .then(r => r.json())
                .then(d => {{
                    if (d.commands && d.commands.length) {{
                        window.moveDrone(d.commands[0]);
                    }}
                }});
        }}, 200);
    }});
    </script>
    """
    m.get_root().html.add_child(Element(js_script))
    m.save("templates/sim_map.html")

# ============================================================
# Spinal Logic (즉각 응답 자율 모드 + SFT 추론)
# ============================================================
def spinal_logic_loop():
    print("⚡ Spinal Cord Active")
    last_inference_time = 0
    inference_interval = 2.0  # 2초마다 추론 (GPU 부하 조절)

    while True:
        with SYSTEM_STATE_LOCK:
            perm = SYSTEM_STATE["ENGAGEMENT_PERMISSION"]
            override = SYSTEM_STATE["HUMAN_OVERRIDE"]
            mode = SYSTEM_STATE["CURRENT_MODE"]
            dist = SYSTEM_STATE.get("LAST_DIST")
            bearing = SYSTEM_STATE.get("LAST_BEARING")
            lat = SYSTEM_STATE.get("LAST_LAT", 37.40)
            lng = SYSTEM_STATE.get("LAST_LNG", 126.97)

        if mode == "HOLD":
            time.sleep(0.1)
            continue

        # DPO Pipeline: State -> Candidates -> Judge (AUTO는 실행/로그 저장, MANUAL은 점수표만 갱신)
        if perm and dist is not None:
            now = time.time()
            if now - last_inference_time > inference_interval:
                print(f"\n[Spinal] DPO Cycle Start (Dist: {dist}m, override={override})")

                # L1: State Builder
                state = build_dpo_state(lat, lng, dist, bearing if bearing else 0)

                # L4: Candidate Generator
                candidates = generate_candidates(state, n=3)
                print(f"[Spinal] Generated {len(candidates)} candidates")

                if candidates:
                    # L5: Judge (score + reasons)
                    chosen, rejected, log_entry = judge_candidates(candidates, state)

                    # --- UI/debug state update (MANUAL/AUTO 공통) ---
                    with SYSTEM_STATE_LOCK:
                        SYSTEM_STATE["DPO_LAST_TS"] = now
                        SYSTEM_STATE["DPO_LAST_DIST_M"] = float(dist) if dist is not None else None
                        SYSTEM_STATE["DPO_LAST_CANDIDATES"] = (log_entry or {}).get("meta", {}).get("candidates_scored", [])
                        SYSTEM_STATE["DPO_LAST_FULL"] = candidates
                        SYSTEM_STATE["DPO_LAST_STATE"] = state
                        SYSTEM_STATE["DPO_LAST_PROMPT"] = (log_entry or {}).get("prompt")

                        # AUTO일 땐 즉시 best 후보를 selected로 표시
                        if not override:
                            SYSTEM_STATE["DPO_LAST_CHOSEN_ID"] = chosen.get("candidate_id")
                            SYSTEM_STATE["DPO_LAST_CHOSEN_ACTION"] = chosen.get("action")
                            SYSTEM_STATE["DPO_LAST_CHOSEN_SCORE"] = chosen.get("score")

                    # --- AUTO: preference log 저장 + 행동 반영 ---
                    if not override and log_entry:
                        save_preference_log(log_entry)
                        print(f"[Spinal] Judge Selected: {chosen['action']} (Score: {chosen['score']}) vs Rejected: {rejected['action']}")

                        if chosen.get("action") in ["ORBIT", "CHASE", "RETREAT", "PATROL", "INTERCEPT"]:
                            action = chosen.get("action")
                            params = chosen.get("params", {}) if isinstance(chosen.get("params", {}), dict) else {}

                            # Store DPO params for debugging/telemetry
                            with SYSTEM_STATE_LOCK:
                                SYSTEM_STATE["DPO_LAST_ACTION"] = action
                                SYSTEM_STATE["DPO_LAST_PARAMS"] = params

                            # Map higher-level actions to an executable mode in the current web-only controller
                            if action == "INTERCEPT":
                                mode = "CHASE"
                            elif action == "PATROL":
                                intent = str(params.get("intent", "patrol")).lower().strip()
                                mode = "CHASE" if intent == "intercept" else "ORBIT"
                            else:
                                mode = action

                            with SYSTEM_STATE_LOCK:
                                SYSTEM_STATE["CURRENT_MODE"] = mode

                last_inference_time = now
        # 이동 명령 생성 (기존 로직 유지)
        if mode in ("CHASE", "RETREAT") and bearing:
            keys = ACTION_MAP.get(mode, {}).get(bearing, [])
            for k in keys:
                DRONE_COMMAND_QUEUE.put(k)

        time.sleep(0.1 if DRONE_COMMAND_QUEUE.qsize() <= 5 else 0.2)


# ============================================================
# Main
# ============================================================
def run_flask():
    app.run(port=5000, use_reloader=False)


def main():
    ensure_dirs()
    ensure_assets()
    init_analyzer()  # 지형 분석기 초기화 (래스터 캐싱)
    init_sft_model() # SFT 모델 로드 (Gemma 3 4B)
    generate_index_html()
    generate_simulation_map()

    print("LLM: Ollama + Gemma 3 12b")
    print("http://127.0.0.1:5000")


    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=spinal_logic_loop, daemon=True).start()

    print("Starting Terminal Chat... (Type 'exit' to quit)")
    try:
        while True:
            # 터미널 입력 대기
            user_input = input("\n[USER] 질문: ")
            
            if user_input.lower() in ["exit", "quit", "종료"]:
                print("Exit.")
                break
            
            if not user_input.strip():
                continue
                
            print("...", end="", flush=True)  # 로딩 중
            
            # 여기서 format_briefing_llm 호출 (지형/센서 분석 + RAG)
            try:
                response = format_briefing_llm(user_input)
                print(f"\r[SIMA] 답변: {response}\n")
            except Exception as e:
                print(f"\r[Error] {e}\n")
                
    except KeyboardInterrupt:
        print("\nExit.")


if __name__ == "__main__":
    main()

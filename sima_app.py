# sima_app.py - 寃쎈웾??踰꾩쟾
"""
SIMA ?쒕줎 ?꾩닠 ?쒕??덉씠???쒖뒪??(Slim Version)
- geo_db.py: 吏??遺꾩꽍 (rasterio)
- sima_model.py: 濡쒖뺄 LLM (Ollama)
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

from geo_db import get_terrain_analysis, init_analyzer  # 吏??遺꾩꽍 (Hybrid)
from sima_model import sima_chat, sima_chat_json  # 濡쒖뺄 LLM
from sima_sft import init_sft_model # 紐⑤뜽 濡쒕뜑
from dpo_core import build_dpo_state, generate_candidates, judge_candidates, save_preference_log  # DPO Core (KR reasons + UI meta)

# Flask 濡쒓렇 ?듭젣
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


# ============================================================
# ?곹깭(怨듭쑀 硫붾え由?
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
# 紐⑤뱶蹂??대룞 留?
# ============================================================
ACTION_MAP = {
    "CHASE":   {"N": ["UP"], "S": ["DOWN"], "E": ["RIGHT"], "W": ["LEFT"],
                "NE": ["UP", "RIGHT"], "NW": ["UP", "LEFT"], "SE": ["DOWN", "RIGHT"], "SW": ["DOWN", "LEFT"]},
    "RETREAT": {"N": ["DOWN"], "S": ["UP"], "E": ["LEFT"], "W": ["RIGHT"],
                "NE": ["DOWN", "LEFT"], "NW": ["DOWN", "RIGHT"], "SE": ["UP", "LEFT"], "SW": ["UP", "RIGHT"]},
    "ORBIT":   {},  # ORBIT 紐⑤뱶??JS ?먯쑉 鍮꾪뻾 (?쒕쾭 紐낅졊 ?놁쓬)
}


# ============================================================
# ?좊떅 ?뺣낫
# ============================================================
UNITS = {
    "TANK": "T-80湲??꾩감",
    "TANK2": "T-72湲??꾩감",
    "DRONE": "?뺤같???쒕줎",
}


# ============================================================
# Flask ??
# ============================================================
app = Flask(__name__)


def ensure_dirs():
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)


def ensure_assets():
    """?쒕줎/?깊겕 ?꾩씠肄??앹꽦"""
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
# SHORT Fast-path (LLM ?고쉶)
# ============================================================
def is_distance_command(msg: str) -> bool:
    m = (msg or "").lower()
    complex_kw = ["媛??, "?됯퇏", "?대뵒", "異붿쿇", "nearest", "average"]
    if any(k in m for k in complex_kw):
        return False
    keys = ["嫄곕━", "distance", "?깊겕", "t-80", "t-72", "bearing", "諛⑹쐞", "?곹깭"]
    return any(k in m for k in keys)


def build_sensor_reply() -> str:
    with SYSTEM_STATE_LOCK:
        status = SYSTEM_STATE.get("LAST_STATUS")
        dist1 = SYSTEM_STATE.get("LAST_DIST1")
        dist2 = SYSTEM_STATE.get("LAST_DIST2")
        bearing = SYSTEM_STATE.get("LAST_BEARING")
        ts = float(SYSTEM_STATE.get("LAST_TELEMETRY_TS", 0.0))

    age = time.time() - ts if ts > 0 else 999.0
    ## kr : ?쒕줎??紐⑺몴 諛섍꼍 踰쀬뼱?ъ쓣 ?? ?쇰쭏??媛뺥븯寃??ㅼ떆 ?먮옒 沅ㅻ룄濡?蹂듦??쒗궗吏 寃곗젙?섎뒗 媛?
    status_kr = {"DETECTED": "?먯???, "SEARCHING": "?먯깋以?, "INIT": "珥덇린??}.get(status, status)
    bearing_kr = {"N": "遺?, "S": "??, "E": "??, "W": "??, "NE": "遺곷룞", "NW": "遺곸꽌", "SE": "?⑤룞", "SW": "?⑥꽌"}.get(bearing, bearing)

    parts = [f"?쇱꽌 媛깆떊: {age:.1f}珥???]
    if status: parts.append(f"?곹깭: {status_kr}")
    if dist1 is not None: parts.append(f"T-80: {dist1:.1f}m")
    if dist2 is not None: parts.append(f"T-72: {dist2:.1f}m")
    if bearing: parts.append(f"諛⑹쐞: {bearing_kr}")

    return " / ".join(parts) if len(parts) > 1 else "?붾젅硫뷀듃由??湲?以?.."


# ============================================================
# ?몄궗/?〓떞 泥섎━ (LLM ?고쉶)
# ============================================================
def handle_greeting(msg: str) -> Optional[str]:
    """?몄궗/?〓떞 媛먯? ??媛꾨떒 ?묐떟 諛섑솚, ?꾨땲硫?None"""
    m = (msg or "").strip().lower()
    
    # ?몄궗 ?⑦꽩
    greetings = ["?덈뀞", "?섏씠", "?щ줈", "hello", "hi", "hey", "諛섍???, "諛섍컩", "?덈뀞?섏꽭??]
    if any(g in m for g in greetings) and len(m) < 20:
        return "?덈뀞?섏꽭?? SIMA ?꾩닠 吏???쒖뒪?쒖엯?덈떎. 吏??遺꾩꽍?대굹 ?곹솴 蹂닿퀬媛 ?꾩슂?섏떆硫?留먯???二쇱꽭??"
    
    # 媛먯궗 ?⑦꽩
    thanks = ["怨좊쭏??, "媛먯궗", "?≫걧", "thanks", "thank you", "?긱뀉"]
    if any(t in m for t in thanks) and len(m) < 20:
        return "泥쒕쭔?먯슂! 異붽? 吏?먯씠 ?꾩슂?섏떆硫??몄젣??留먯???二쇱꽭??"
    
    return None
# ============================================================
# LLM 釉뚮━??(JSON ?뺣떟吏 湲곕컲)
# ============================================================
def format_briefing_llm(user_msg: str) -> str:
    """吏???뺣떟吏 + ?쇱꽌 ?뺣낫 ??LLM 釉뚮━??""
    import json
    from datetime import datetime

    with SYSTEM_STATE_LOCK:
        lat = SYSTEM_STATE.get("LAST_LAT", 37.40)
        lng = SYSTEM_STATE.get("LAST_LNG", 126.97)
        dist1 = SYSTEM_STATE.get("LAST_DIST1")
        dist2 = SYSTEM_STATE.get("LAST_DIST2")
        bearing = SYSTEM_STATE.get("LAST_BEARING")

    # 吏??遺꾩꽍
    try:
        terrain_context = get_terrain_analysis(lat, lng, radius_m=500)
    except Exception as e:
        terrain_context = {"error": str(e)}

    # ?쇱꽌 ?뺣낫 異붽?
    terrain_context["sensor"] = {
        "tank1_dist_m": dist1,
        "tank2_dist_m": dist2,
        "bearing": bearing
    }

    # ?뺣떟吏 JSON ?뚯씪 ???
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"terrain_{timestamp}.json")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(terrain_context, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"?뺣떟吏 ????ㅽ뙣: {e}")

    system_instruction = (
        "臾댁“嫄??쒓뎅?대줈 ?듬? ?? "
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

        # 蹂댄넻 MANUAL?대㈃ ?щ엺 媛쒖엯(override) 耳쒓퀬,
        # AUTO硫?override ?꾨뒗 ?앹쑝濡??곌껐?⑸땲??
        SYSTEM_STATE["HUMAN_OVERRIDE"] = (new == "MANUAL")

        return jsonify({
            "ok": True,
            "dpo_mode": new,
            "override": SYSTEM_STATE["HUMAN_OVERRIDE"],
            "current_mode": SYSTEM_STATE.get("CURRENT_MODE", "PATROL"),
        })


@app.route("/dpo_last")
def dpo_last():
    """UI?? 理쒓렐 ?꾨낫 ?먯닔?쒕? 諛섑솚"""
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
    """MANUAL 紐⑤뱶?먯꽌 ?ъ슜?먭? ?꾨낫瑜??좏깮??preference log濡????+ ?됰룞 諛섏쁺"""
    try:
        payload = request.get_json(force=True, silent=True) or {}
        cid = int(payload.get("candidate_id"))
    except Exception:
        return jsonify({"ok": False, "error": "candidate_id媛 ?щ컮瑜댁? ?딆뒿?덈떎."})

    with SYSTEM_STATE_LOCK:
        if SYSTEM_STATE.get("DPO_MODE") != "MANUAL" or not SYSTEM_STATE.get("HUMAN_OVERRIDE"):
            return jsonify({"ok": False, "error": "MANUAL 紐⑤뱶?먯꽌留??좏깮?????덉뒿?덈떎. (DPO 紐⑤뱶 ?좉? ?꾩슂)"})

        full = SYSTEM_STATE.get("DPO_LAST_FULL") or []
        state = SYSTEM_STATE.get("DPO_LAST_STATE")
        prompt = SYSTEM_STATE.get("DPO_LAST_PROMPT")

    if not full or not state:
        return jsonify({"ok": False, "error": "?좏깮???꾨낫 ?곗씠?곌? ?놁뒿?덈떎. (?깊겕媛 2km ?대궡濡??ㅼ뼱????앹꽦??"})

    chosen = None
    for c in full:
        if int(c.get("candidate_id", -1)) == cid:
            chosen = c
            break
    if chosen is None:
        return jsonify({"ok": False, "error": f"candidate_id={cid} ?꾨낫瑜?李얠쓣 ???놁뒿?덈떎."})

    # rejected: ?섎㉧吏 以?理쒖? ?먯닔 (?놁쑝硫????遺덇?)
    others = [c for c in full if int(c.get("candidate_id", -1)) != cid]
    if not others:
        return jsonify({"ok": False, "error": "?꾨낫媛 1媛쒕퓧?대씪 preference pair瑜?留뚮뱾 ???놁뒿?덈떎."})
    rejected = sorted(others, key=lambda x: float(x.get("score", -1e9)))[0]

    # preference log entry 援ъ꽦
    try:
        import json as _json
        chosen_json = _json.dumps(chosen.get("normalized", {}), ensure_ascii=False)
        rejected_json = _json.dumps(rejected.get("normalized", {}), ensure_ascii=False)

        # UI-friendly scored table (?대? judge?먯꽌 怨꾩궛??
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
        return jsonify({"ok": False, "error": f"濡쒓렇 ????ㅽ뙣: {e}"})

    # ?됰룞 諛섏쁺 (湲곗〈 留ㅽ븨 濡쒖쭅怨??숈씪)
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
    """UI 踰꾪듉??吏곸젒 ?쒖뼱 紐낅졊"""
    try:
        payload = request.get_json(force=True, silent=True) or {}
        action = str(payload.get("action", "")).upper()
    except Exception:
        return jsonify({"ok": False, "error": "Invalid payload"})

    # ACTION_MAP???뺤쓽??紐⑤뱶?몄? ?뺤씤
    # INTERCEPT/PATROL? ?몄쓽???덉슜?섎릺 ?ㅼ젣 援щ룞 紐⑤뱶(CHASE/ORBIT)濡?留ㅽ븨
    valid_actions = ["CHASE", "RETREAT", "ORBIT", "HOLD", "PATROL", "INTERCEPT"]
    if action not in valid_actions:
        return jsonify({"ok": False, "error": f"Unknown action: {action}"})

    mode = action
    if action == "INTERCEPT":
        mode = "CHASE"
    elif action == "PATROL":
        mode = "ORBIT"  # 湲곕낯 ?뺤같? ?좏쉶

    with SYSTEM_STATE_LOCK:
        SYSTEM_STATE["HUMAN_OVERRIDE"] = True
        SYSTEM_STATE["CURRENT_MODE"] = mode
        # DPO 濡쒓퉭???붾? ?곹깭 ?낅뜲?댄듃 (?좏깮 ?ы빆)
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

    # 1. ?몄궗/?〓떞 (LLM ?고쉶)
    greeting_reply = handle_greeting(user_msg)
    if greeting_reply:
        return jsonify({"response": greeting_reply})

    # 2. ?쇱꽌 蹂닿퀬 (LLM ?고쉶)
    if is_distance_command(user_msg.lower()) or any(k in user_msg.lower() for k in ["?곹솴", "蹂닿퀬"]):
        return jsonify({"response": build_sensor_reply()})

    # 3. LLM 釉뚮━??
    try:
        response = format_briefing_llm(user_msg)
        return jsonify({"response": response[:3000]})
    except Exception as e:
        return jsonify({"response": f"?ㅻ쪟: {e}"})


# ============================================================
# 吏???앹꽦
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
      <button class="ghost small" onclick="toggleDPO()">DPO 紐⑤뱶 ?좉?</button>
      <button class="secondary small" onclick="quick('?곹솴 蹂닿퀬??)">?곹솴 蹂닿퀬</button>
    </div>
  </div>

  <div id="main-split">
    <div id="dpo-panel">
      <div id="dpo-head">
        <div>
          <div id="dpo-title">?꾨낫 ?먯닔??/div>
          <div id="dpo-sub">理쒓렐 DPO ?ㅽ뀦???꾨낫蹂??먯닔 諛??댁쑀瑜??쒖떆?⑸땲??</div>
        </div>
        <button class="secondary small" onclick="refreshDPO(true)">?덈줈怨좎묠</button>
      </div>
      <div id="dpo-cards">
        <div class="card"><div style="color:var(--muted); font-size:12px;">?꾩쭅 ?꾨낫 ?먯닔 ?곗씠?곌? ?놁뒿?덈떎. (?깊겕媛 2km ?대궡濡??ㅼ뼱?ㅻ㈃ ?앹꽦??</div></div>
      </div>
    </div>

    <div id="chat-container">
      <div class="msg ai">SIMA ?쒖뒪???⑤씪??/div>
    </div>

    <div id="input-area">
      <input type="text" id="txt" placeholder="紐낅졊 ?낅젰..."/>
      <button onclick="send()">?꾩넚</button>
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
    alert(data && data.error ? data.error : '?좏깮 ?ㅽ뙣');
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
    empty.innerHTML = `<div style="color:var(--muted); font-size:12px;">?꾩쭅 ?꾨낫 ?먯닔 ?곗씠?곌? ?놁뒿?덈떎.</div>`;
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
      reasonsHtml = `<div style="margin-top:8px; color:var(--muted); font-size:12px;">?댁쑀 ?놁쓬</div>`;
    }

    let actionsHtml = '';
    // MANUAL + override???뚮쭔 ?좏깮 踰꾪듉??蹂댁뿬以?
    if(dpoMode === 'MANUAL' && override){
      actionsHtml = `
        <div class="card-actions">
          <button class="ghost small" onclick="selectCandidate(${c.candidate_id})">???꾨낫 ?좏깮</button>
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
    # ?덉뼇/?몃뜒??吏??(DSK_2026 ?섏뒪??踰붿쐞)
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

    # 留덉빱
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

    // Orbit(?먰삎 ?좏쉶) ?뚮씪誘명꽣
    window.orbit = {{
      enabled: true,
      radius_m: 500,   // 紐⑺몴臾쇨낵 ?좎???諛섍꼍(?? 300~800m)
      vt_mps: 25,      // ?묒꽑 ?띾룄 (m/s) ?? 10~25
      clockwise: true, // ?쒓퀎 諛⑺뼢
      max_mps: 30,      // ?띾룄 ?곹븳(?덉젙??
      // ?ы솕 蹂듭썝 ?뚮씪誘명꽣
      k: 20,        // 諛⑹궗 蹂듭썝 理쒕? 媛뺣룄 ?ㅼ??????m/s湲?
      korbit: 6.0,  // ?ы솕 誘쇨컧???댁닔濡?鍮⑤━ ?ы솕)
    }};

    // ?꾧꼍??-> 濡쒖뺄 誘명꽣 蹂??洹쇱궗, ?묒? 援ъ뿭?먯꽌 異⑸텇)
    function metersPerDegLat() {{ return 111320.0; }}
    function metersPerDegLng(lat) {{ return 111320.0 * Math.cos(lat * Math.PI / 180); }}

    // (lat,lng) 李⑥씠瑜?meters (x=?? y=遺?濡?
    function dLatLngToMeters(dLat, dLng, refLat) {{
      const mx = dLng * metersPerDegLng(refLat);
      const my = dLat * metersPerDegLat();
      return {{ x: mx, y: my }};
    }}

    // meters (x,y)瑜?(dLat,dLng)濡?
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
                // 嫄곕━ 怨꾩궛
                const tPos = (dist1 <= dist2) ? tPos1 : tPos2;

                const dLat = dPos.lat - tPos.lat; // ?꾨룄 李⑥씠
                const dLng = dPos.lng - tPos.lng; // 寃쎈룄 李⑥씠
                const v = dLatLngToMeters(dLat, dLng, tPos.lat); // ?꾨룄, 寃쎈룄 李⑥씠瑜??ㅼ젣 誘명꽣濡?蹂??
                const r = Math.max(Math.hypot(v.x, v.y), 1e-3); // 以묒떖源뚯????꾩옱 吏곸꽑嫄곕━ r

                const er = r - window.orbit.radius_m; // ?꾩옱 嫄곕━ - 紐⑺몴 500m 李⑥씠, ?묒닔硫??밴꺼???섍퀬 ?뚯닔硫?諛?대궡????
                const cw = window.orbit.clockwise ? 1 : -1; // ?쒓퀎諛⑺뼢?대㈃ 1, 諛섏떆怨꾨갑?μ씠硫?-1
                
                // 諛⑺뼢 踰≫꽣瑜?怨꾩궛?섎뒗 ?섏떇
                const urx = v.x / r, ury = v.y / r;  // 諛⑹궗 踰≫꽣
                const utx = cw * ury, uty = cw * -urx; // ?묒꽑 踰≫꽣
                
                // 理쒖쥌 ?띾룄 怨꾩궛
                // 諛⑹궗 諛⑺뼢 蹂듭썝 ?띾룄 ?깅텇 (?ы솕??
                const vr = -window.orbit.k * Math.atan(window.orbit.korbit * (er / window.orbit.radius_m));

                // 理쒖쥌 ?띾룄 = 諛⑹궗 蹂듭썝 + ?묒꽑 ?좏쉶
                let vx = vr * urx + window.orbit.vt_mps * utx;
                let vy = vr * ury + window.orbit.vt_mps * uty;


                const sp = Math.hypot(vx, vy); // ?꾩옱 ?띾룄 ?ш린
                const vmax = window.orbit.max_mps; // 理쒕? ?띾룄
                if (sp > vmax) {{ vx *= (vmax / sp); vy *= (vmax / sp); }} // 理쒕? ?띾룄瑜??좎??섍린 ?꾪빐 ?뺢퇋?? 鍮꾩쑉??以꾩뿬 理쒕? ?띾룄??留욎땄.

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
# Spinal Logic (利됯컖 ?묐떟 ?먯쑉 紐⑤뱶 + SFT 異붾줎)
# ============================================================
def spinal_logic_loop():
    print("??Spinal Cord Active")
    last_inference_time = 0
    inference_interval = 2.0  # 2珥덈쭏??異붾줎 (GPU 遺??議곗젅)

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

        # DPO Pipeline: State -> Candidates -> Judge (AUTO???ㅽ뻾/濡쒓렇 ??? MANUAL? ?먯닔?쒕쭔 媛깆떊)
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

                    # --- UI/debug state update (MANUAL/AUTO 怨듯넻) ---
                    with SYSTEM_STATE_LOCK:
                        SYSTEM_STATE["DPO_LAST_TS"] = now
                        SYSTEM_STATE["DPO_LAST_DIST_M"] = float(dist) if dist is not None else None
                        SYSTEM_STATE["DPO_LAST_CANDIDATES"] = (log_entry or {}).get("meta", {}).get("candidates_scored", [])
                        SYSTEM_STATE["DPO_LAST_FULL"] = candidates
                        SYSTEM_STATE["DPO_LAST_STATE"] = state
                        SYSTEM_STATE["DPO_LAST_PROMPT"] = (log_entry or {}).get("prompt")

                        # AUTO????利됱떆 best ?꾨낫瑜?selected濡??쒖떆
                        if not override:
                            SYSTEM_STATE["DPO_LAST_CHOSEN_ID"] = chosen.get("candidate_id")
                            SYSTEM_STATE["DPO_LAST_CHOSEN_ACTION"] = chosen.get("action")
                            SYSTEM_STATE["DPO_LAST_CHOSEN_SCORE"] = chosen.get("score")

                    # --- AUTO: preference log ???+ ?됰룞 諛섏쁺 ---
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
        # ?대룞 紐낅졊 ?앹꽦 (湲곗〈 濡쒖쭅 ?좎?)
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
    init_analyzer()  # 吏??遺꾩꽍湲?珥덇린??(?섏뒪??罹먯떛)
    init_sft_model() # SFT 紐⑤뜽 濡쒕뱶 (Gemma 3 4B)
    generate_index_html()
    generate_simulation_map()

    print("LLM: Ollama + Gemma 3 12b")
    print("http://127.0.0.1:5000")


    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=spinal_logic_loop, daemon=True).start()

    print("Starting Terminal Chat... (Type 'exit' to quit)")
    try:
        while True:
            # ?곕????낅젰 ?湲?
            user_input = input("\n[USER] 吏덈Ц: ")
            
            if user_input.lower() in ["exit", "quit", "醫낅즺"]:
                print("Exit.")
                break
            
            if not user_input.strip():
                continue
                
            print("...", end="", flush=True)  # 濡쒕뵫 以?
            
            # ?ш린??format_briefing_llm ?몄텧 (吏???쇱꽌 遺꾩꽍 + RAG)
            try:
                response = format_briefing_llm(user_input)
                print(f"\r[SIMA] ?듬?: {response}\n")
            except Exception as e:
                print(f"\r[Error] {e}\n")
                
    except KeyboardInterrupt:
        print("\nExit.")


if __name__ == "__main__":
    main()

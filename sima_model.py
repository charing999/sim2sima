"""
SIMA 모델 래퍼 (Ollama + Gemma 3 12B)
파인튜닝 없이 프롬프트 엔지니어링 + 파라미터 튜닝으로 운영
"""
import requests
import json
import re
from typing import Optional

# ============================================================
# Ollama 서버 설정
# ============================================================
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "gemma3:12b"  # Ollama에서 사용할 모델명

# 최적화된 파라미터 (전술 브리핑용)
DEFAULT_TEMPERATURE = 0.1    # 일관된 응답
DEFAULT_TOP_P = 0.9          # 적절한 다양성
DEFAULT_TOP_K = 60           # 정밀한 토큰 선택


def _check_server() -> bool:
    """Ollama 서버 상태 확인"""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return resp.status_code == 200
    except:
        return False


def _check_model() -> bool:
    """모델 설치 여부 확인"""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            for m in models:
                if MODEL_NAME in m.get("name", ""):
                    return True
        return False
    except:
        return False


def sima_chat(prompt: str, max_tokens: int = 512, temperature: float = DEFAULT_TEMPERATURE) -> str:
    """
    Ollama API로 Gemma 3 12B 응답 생성
    """
    if not _check_server():
        return "[오류] Ollama 서버가 실행 중이 아닙니다. 'ollama serve'를 먼저 실행하세요."
    
    # Gemma 3 chat 템플릿은 Ollama가 자동 적용
    # 직접 템플릿 래핑 제거 (Ollama가 처리)
    clean_prompt = prompt
    if "<start_of_turn>" in prompt:
        # 기존 템플릿 제거 (Ollama가 다시 적용하므로)
        clean_prompt = prompt.replace("<start_of_turn>user\n", "").replace("<start_of_turn>model\n", "")
        clean_prompt = clean_prompt.replace("<end_of_turn>\n", "\n").replace("<end_of_turn>", "")
    
    payload = {
        "model": MODEL_NAME,
        "prompt": clean_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": DEFAULT_TOP_P,
            "top_k": DEFAULT_TOP_K,
            "num_predict": max_tokens,
            "stop": ["브리핑 종료."],  # 종료 토큰
        }
    }
    
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=60,  # Ollama는 더 오래 걸릴 수 있음
        )
        resp.raise_for_status()
        result = resp.json()
        response_text = result.get("response", "").strip()
        
        # "브리핑 종료." 가 잘리면 추가
        if response_text and not response_text.endswith("브리핑 종료."):
            response_text += "\n\n브리핑 종료."
        
        return response_text
    except Exception as e:
        return f"[오류] Ollama 호출 실패: {e}"


def sima_chat_json(prompt: str, max_tokens: int = 128) -> dict:
    """
    JSON 응답 생성 (라우터용)
    """
    json_prompt = prompt + "\n\n반드시 JSON 형식으로만 응답하세요: {\"intent\": \"...\", \"params\": {...}}"
    
    response = sima_chat(json_prompt, max_tokens, temperature=0.1)
    
    # JSON 추출 시도
    try:
        # ```json 블록 처리
        if "```" in response:
            response = response.replace("```json", "```")
            parts = response.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("{") and p.endswith("}"):
                    response = p
                    break
        
        # JSON 패턴 찾기
        if not (response.startswith("{") and response.endswith("}")):
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if m:
                response = m.group(0)
        
        return json.loads(response)
    except:
        return {"reply": response, "override": False, "permission": False}


# ============================================================
# 호환성 함수
# ============================================================
def load_model():
    """Ollama 방식에서는 모델 로드가 필요 없음"""
    print(f"[SIMA] Ollama 모드 - 모델: {MODEL_NAME}")
    return None, None


def preload_model():
    """서버 및 모델 상태 확인"""
    if _check_server():
        print("[SIMA] ✅ Ollama 서버 연결됨!")
        if _check_model():
            print(f"[SIMA] ✅ 모델 '{MODEL_NAME}' 확인됨!")
        else:
            print(f"[SIMA] ⚠️ 모델 '{MODEL_NAME}'이 설치되어 있지 않습니다!")
            print(f"[SIMA] 다음 명령어로 모델을 설치하세요:")
            print(f"  ollama pull {MODEL_NAME}")
    else:
        print("[SIMA] ⚠️ Ollama 서버가 실행 중이 아닙니다!")
        print("[SIMA] 다음 명령어로 서버를 실행하세요:")
        print("  ollama serve")


# ============================================================
# 테스트
# ============================================================
if __name__ == "__main__":
    print(f"\n=== Ollama + {MODEL_NAME} 테스트 ===")
    
    if _check_server():
        print("✅ 서버 연결됨!")
        
        if _check_model():
            print(f"✅ 모델 '{MODEL_NAME}' 확인!")
            result = sima_chat("전방 500m에 적 전차가 있습니다. 전술 브리핑을 해주세요.", max_tokens=200)
            print(f"\n응답:\n{result}")
        else:
            print(f"❌ 모델 '{MODEL_NAME}' 없음!")
            print(f"\n모델 설치: ollama pull {MODEL_NAME}")
    else:
        print("❌ 서버 연결 실패!")
        print("\n서버 실행: ollama serve")


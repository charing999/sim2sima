import torch
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Global instances
SFT_MODEL = None
SFT_TOKENIZER = None

def init_sft_model():
    """
    Load the Fine-Tuned Gemma 3 model with specific settings to avoid NaN issues.
    Strategy: bf16 + eager attention + merge_and_unload
    """
    global SFT_MODEL, SFT_TOKENIZER
    
    base_id = "google/gemma-3-4b-it"
    # Use absolute path relative to this script location for robustness
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_path = os.path.join(script_dir, "DPO_drone", "gemma3-drone-web-sft")
    
    print(f"[SFT] Loading Tokenizer: {base_id}")
    SFT_TOKENIZER = AutoTokenizer.from_pretrained(base_id)
    SFT_TOKENIZER.pad_token = SFT_TOKENIZER.eos_token
    SFT_TOKENIZER.padding_side = "right"
    
    print(f"[SFT] Loading Base Model: {base_id} (bf16)")
    # Force eager attention and float16/bfloat16 to match successful test
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map="auto",
        dtype=torch.bfloat16,  # Use bfloat16 as verified in tests
        attn_implementation="eager",
    )
    
    print(f"[SFT] Loading Adapter and Merging: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)
    
    # Critical Fix: Merge adapter to avoid runtime NaNs
    SFT_MODEL = model.merge_and_unload()
    SFT_MODEL.eval()
    
    # Configure generation
    SFT_MODEL.generation_config.do_sample = False  # Deterministic
    SFT_MODEL.generation_config.pad_token_id = SFT_TOKENIZER.pad_token_id
    SFT_MODEL.generation_config.eos_token_id = SFT_TOKENIZER.eos_token_id
    
    print("[SFT] Model Initialized Successfully (Merged)")

def build_prompt_json(state: dict, profile: str = "AUTO") -> str:
    """
    Constructs a JSON-only prompt to strictly guide the LLM's output format and tactical profile.
    """
    prompt_obj = {
        "prompt_version": "sima_policy_json_v2",
        "instruction": {
            "task": "Decide the next action for the drone in a tactical map simulation.",
            "output_rule": {
                "format": "JSON_ONLY",
                "must_be_single_object": True,
                "no_markdown": True,
                "no_explanation": True,
                "schema": {
                    "type": "object",
                    "required_keys": ["drone_1"],
                    "properties": {
                        "drone_1": {
                            "type": "object",
                            "required_keys": ["action", "params"],
                            "properties": {
                                "action": {"enum": ["ORBIT", "CHASE", "RETREAT", "PATROL"]},
                                "params": {"type": "object"},
                            },
                        }
                    },
                },
            },
        },
        "required_params_by_action": {
            "ORBIT": ["radius_m", "angular_speed_dps", "turn_rate_limit_dps"],
            "CHASE": ["desired_distance_m", "desired_speed_mps", "turn_rate_limit_dps"],
            "RETREAT": ["retreat_distance_m", "desired_speed_mps", "turn_rate_limit_dps"],
            "PATROL": ["intent", "desired_speed_mps", "turn_rate_limit_dps"],
        },
        "conditional_required_params": [
            {
                "when": {"action": "PATROL", "params.intent": "intercept"},
                "must_include": ["target_lead_time_s"],
            }
        ],
        "param_ranges": {
            "desired_speed_mps": [5, 25],
            "turn_rate_limit_dps": [20, 90],
            "desired_distance_m": [300, 2000],
            "retreat_distance_m": [150, 800],
            "radius_m": [150, 600],
            "angular_speed_dps": [10, 60],
            "target_lead_time_s": [1, 6],
        },
        "policy_rules": [
            {"if": {"dist_to_target_m": {"lt": 300}}, "prefer": ["RETREAT"]},
            {"if": {"dist_to_target_m": {"gt": 2000}}, "prefer": ["CHASE", "PATROL"]},
            {"if": {"dist_to_target_m": {"between": [500, 2000]}}, "prefer": ["CHASE", "ORBIT", "PATROL"]},
        ],
        "candidate_profile_hint": {
            "enabled": True,
            "profile": profile,
            "profiles": {
                "A_CHASE": {"prefer_actions": ["CHASE"], "patrol_intent": None},
                "B_INTERCEPT": {"prefer_actions": ["PATROL"], "patrol_intent": "intercept"},
                "C_RECON_RETREAT": {"prefer_actions": ["PATROL", "RETREAT"], "patrol_intent": "recon"},
                "AUTO": {"prefer_actions": [], "patrol_intent": None},
            },
        },
        "state": state,
        "self_checklist": [
            "Output is valid JSON object only",
            "Has key drone_1",
            "Contains action and params",
            "All required params present for the chosen action",
            "All numeric params within recommended ranges",
        ],
    }

    return json.dumps(prompt_obj, ensure_ascii=False)

# Backward compatibility alias
def build_prompt(drone_lat, drone_lng, dist_to_target):
    # Construct a minimal state for compatibility
    state = {
        "drone_id": "drone_1",
        "location": {"lat": drone_lat, "lng": drone_lng},
        "target_info": {"dist": dist_to_target if dist_to_target is not None else 0, "bearing": 0},
        "timestamp": 0
    }
    return build_prompt_json(state, profile="AUTO")

def predict_action(drone_lat=37.4, drone_lng=126.9, dist_to_target=1500):
    """
    Generate action for the drone.
    Returns: action string (e.g., "CHASE", "ORBIT")
    """
    global SFT_MODEL, SFT_TOKENIZER
    
    if SFT_MODEL is None:
        return "ORBIT" # Fallback
        
    prompt = build_prompt(drone_lat, drone_lng, dist_to_target)
    
    inputs = SFT_TOKENIZER(prompt, return_tensors="pt").to(SFT_MODEL.device)
    
    try:
        with torch.no_grad():
            out = SFT_MODEL.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
        
        # Decode output
        generated_text = SFT_TOKENIZER.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse JSON
        # Output format expected: {"drone_1": {"action": "CHASE"}, ...}
        # Try to find JSON block
        json_str = generated_text
        if "```json" in generated_text:
            json_str = generated_text.split("```json")[1].split("```")[0]
        elif "```" in generated_text:
            json_str = generated_text.split("```")[1]
            
        data = json.loads(json_str.strip())
        
        # Extract action for drone_1
        my_action = data.get("drone_1", {}).get("action", "ORBIT")
        return my_action.upper()
        
    except Exception as e:
        print(f"[SFT] Inference Error: {e}")
        return "ORBIT" # Safe fallback

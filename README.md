# 🚁 Sim2SIMA: Autonomous Drone Tactical Simulation (DPO & RLHF)

Sim2SIMA is an advanced simulation framework for training and evaluating autonomous drone tactics using **LLM-based decision making**. It features a **Dual-Brain Architecture** (SFT + DPO) and supports **Human-in-the-Loop (RLHF)** via a manual control interface.

## 🌟 Key Features

### 🧠 Dual-Brain Architecture
- **Reflex Core (`sima_sft.py`)**: Powered by **Gemma 4B SFT**, handling split-second tactical maneuvers (Orbit, Chase, Retreat) via strict JSON output.
- **Reasoning Core (`sima_model.py`)**: Powered by **Gemma 12B (Ollama)**, providing natural language situational awareness and tactical briefings.

### 🎯 Direct Preference Optimization (DPO) Pipeline
- **Auto Data Collection**: Automatically logs drone decisions (Prompt + Chosen/Rejected Action) into `dpo_preference_data_v2.jsonl`.
- **Real-time Scorecard**: Visualizes candidate actions and tactical reasoning (e.g., "Distance > 2km → Chase Preferred") in the Web UI.
- **Offline Training**: `train_dpo.py` fine-tunes the SFT model using the collected preference data.

### 🎮 Human-in-the-Loop (RLHF)
- **Manual Control Center**: Direct override buttons (`CHASE`, `RETREAT`, `INTERCEPT`, `PATROL`) in the UI.
- **Implicit Feedback**: When a human overrides the AI, the system pauses DPO generation, allowing for cleaner ground-truth data collection (future feature).

## 📂 File Structure

- `sima_app.py`: Main Flask application (Simulation Server, Web UI, Map).
- `dpo_core.py`: DPO logic, State Builder, Candidate Generation, and Scoring/Judging.
- `sima_sft.py`: Loads the SFT (Fine-tuned) model and generates actions.
- `sima_model.py`: Connects to Ollama for natural language interaction.
- `train_dpo.py`: Script to train the DPO adapter using `trl`.
- `geo_db.py`: Terrain and geospatial analysis utilities.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch (CUDA supported)
- [Ollama](https://ollama.com/) (for Gemma 12B)
- Hugging Face Token (for Gemma access)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/charing999/sim2sima.git
   cd sim2sima
   ```

2. **Install Dependencies**
   ```bash
   pip install torch transformers peft trl flask folium requests
   ```

3. **Setup Models**
   - Ensure the SFT adapter is located at `DPO_drone/gemma3-drone-web-sft`.
   - Pull the reasoning model: `ollama pull gemma3:12b`.

### ✨ Usage

1. **Run the Simulation Server**
   ```bash
   python sima_app.py
   ```
   - Open `http://localhost:5000` in your browser.
   - Use the **Manual Control Buttons** to test specific tactics.
   - Switch "DPO Mode Toggle" to **AUTO** to let the AI drive and collect data.

2. **Train DPO Model**
   Once you have collected enough data in `dpo_preference_data_v2.jsonl`:
   ```bash
   python train_dpo.py
   ```
   This will save a new DPO-tuned adapter in `gemma3-drone-dpo`.

## 🛡️ License
Private / Research Use Only.

# Sim2SIMA

A drone tactics simulator that uses LLMs for decision-making. It has a DPO
and RLHF pipeline bolted on, so preference data collected from manual human
overrides can be used to incrementally tune the SFT model.

## Deployment context

The tactical decision code here ended up in a ROS-based drone for
Drone Show Korea 2026. I handled the AI side — decision loop, DPO pipeline,
and terrain analysis. A separate engineer did the ROS integration and flight
stack. I handed off a Python package; they wired it in.

## Why this exists

I first tried using a single Gemma 12B for everything — situational awareness
and immediate action selection. Inference was too slow for a real-time loop,
so I split the model in two:

- A small SFT model (Gemma 4B) emits actions as JSON
- A larger model (Gemma 12B via Ollama) handles natural-language situational
  briefings in the background

It's not as fancy as "Dual-Brain Architecture" makes it sound, but it works.

## DPO data collection

In auto mode, every decision logs the candidate actions (Orbit, Chase, Retreat,
etc.), the chosen one, and the rejected ones into
`dpo_preference_data_v2.jsonl`. How chosen/rejected gets decided is hardcoded
as heuristics in `dpo_core.py` — e.g. "distance > 2km → prefer Chase". This
part is still rule-based and has obvious limits.

Manual overrides should give cleaner ground truth, so when a human takes
control, automatic DPO generation pauses. The pipeline that turns those
override logs into training data isn't wired up yet though.

## Manual control (for RLHF)

The web UI has `CHASE`, `RETREAT`, `INTERCEPT`, `PATROL` buttons that override
the AI's decision. This becomes implicit feedback for the model.

## Terrain data (DSK_2026)

`geo_db.py` reads raster files covering the DSK 2026 venue near Anyang /
Indeokwon. Four layers: DEM, slope, aspect, and LULC (Ministry of Environment
land classification). Files aren't in the repo. Point to yours:

    DEM_PATH=/path/to/base_dem_3857.tif
    SLOPE_PATH=/path/to/base_slope_3857.tif
    ASPECT_PATH=/path/to/base_aspect_3857.tif
    LULC_PATH=/path/to/base_l3_code_3857.tif

If they're missing, the terrain module skips gracefully and the rest still runs.

## File layout

- `sima_app.py` — Flask server, simulation loop, map UI
- `dpo_core.py` — state builder, candidate generation, scoring
- `sima_sft.py` — loads the SFT adapter and runs action inference
- `sima_model.py` — calls the 12B model via Ollama
- `train_dpo.py` — uses `trl` to train a DPO adapter from the collected jsonl
- `geo_db.py` — terrain / geospatial utilities

## Running it

Requirements:

- Python 3.10+
- PyTorch (CUDA recommended; SFT inference on CPU is painful)
- Ollama (to host the 12B model)
- A Hugging Face token (for Gemma access)

Setup:git clone https://github.com/charing999/sim2sima.git
cd sim2sima
pip install torch transformers peft trl flask folium requests

The SFT adapter should be at `DPO_drone/gemma3-drone-web-sft`. Pull the
reasoning model with `ollama pull gemma3:12b`.

Run the server:python sima_app.py

Open `http://localhost:5000`. Leave it in AUTO mode to let the AI drive and
collect data, or use the manual buttons to intervene.

Once you've collected enough samples, train a new adapter:python train_dpo.py

The DPO-tuned adapter gets saved to `gemma3-drone-dpo`.

## Known limitations

- The chosen/rejected judgment is rule-based, so any bias in the rules
  propagates into the DPO model.
- Manual-override logs aren't yet converted into DPO training samples
  automatically.
- The simulation environment is simple. The gap to a real drone control
  stack is large.

## License

Research / personal use only. Contact me for commercial use.

# Training a player model on your own playtime

PyMC_Bot can learn **how you play** and then play that way itself. The loop is:

```
  Minecraft + ./mod                  pymc_bot train                 pymc_bot run
  ─────────────────                  ───────────────                ─────────────
  /pymc record start      ──▶   dataset.jsonl   ──▶  models/<run>/  ──▶  brain = trained
  play normally                                        latest.npz        (panel or CLI)
  /pymc export                                         model.json
                                                       checkpoints/
```

Everything runs locally: no cloud, no GPU, no Ollama required for the trained brain.

## 1. Record playtime (Fabric mod)

```text
/pymc record start
# ... play for 10-60 minutes: walk, mine, fight, build, look around ...
/pymc record stop
/pymc export
```

That writes `<gameDir>/pymc-playtime/dataset.jsonl`. See [`mod/README.md`](../mod/README.md)
for the exact sample format, the sampling rate (`/pymc sample <ticks>`) and how the mod was
built. Copy that file next to the bot (or point `--dataset` at it).

No Minecraft handy? You can still exercise the whole pipeline with a synthetic session:

```bash
python -m pymc_bot train --simulate 10 --simulate-out pymc-playtime/dataset.jsonl
```

This fabricates a believable player (exploring / mining / fighting profiles) with exactly
the mod's schema — handy for smoke tests and demos, not a substitute for your own play.

## 2. Train (self-checkpointing)

```bash
python -m pymc_bot train --dataset pymc-playtime/dataset.jsonl --steps 2000
```

* **Engine** — `--engine mlp` (default) is a two-hidden-layer numpy network that trains in
  seconds on CPU and needs nothing beyond NumPy; `--engine transformer` is a small
  attention model over the observation tokens and needs `pip install -r requirements-train.txt`.
* **Checkpoints** — every `--checkpoint-every` steps (default 200) and once more when
  training ends, the trainer writes:

  ```
  models/playtime-dataset/
  ├── latest.npz            weights + optimiser state (or latest.pt for the transformer)
  ├── checkpoints/step-0000200.npz
  ├── model.json            the card the panel lists (metrics, dataset digest, vocab)
  ├── trainer_state.json    step, epoch, RNG, best val loss  -> resume support
  ├── metrics.jsonl, metrics.csv
  ```

* **Resume** — `python -m pymc_bot train --resume --run-name playtime-dataset` continues
  from `trainer_state.json`: same weights, same optimiser moments, same shuffling order.
* **Long runs** — `--max-seconds 3600` trains for an hour and stops cleanly (or stop from
  the panel); a checkpoint always lands first.
* **Inspect** — `python -m pymc_bot train --inspect --dataset ...` prints the dataset
  statistics (samples, episodes, action histogram) without training.
* **List** — `python -m pymc_bot models` shows every checkpoint with its metrics.

Useful flags: `--steps`, `--epochs`, `--batch-size`, `--lr`, `--hidden 128,64`,
`--checkpoint-every`, `--no-blocks` (smaller model for bots whose bridge cannot see blocks),
`--quiet`.

### What the model learns

Each sample becomes a feature vector (`pymc_bot/features.py`, 215 floats) — normalised
position, velocity, view as sin/cos, posture flags, hotbar slot, health/food, dimension,
the six closest entities and the 3×3×3 block neighbourhood — and two targets:

* **which action** the player took (12 classes, from the mod's `activity` field), and
* **how the player moved**: `dyaw`, `dpitch`, `forward`, `strafe` until the next sample.

Playtime is dominated by walking, so the loss uses tempered inverse-frequency class
weights (`class_weight_power`, capped) — otherwise the model just holds `forward` forever.
Metrics therefore report both plain accuracy and **balanced accuracy** (mean per-class
recall), which is the number to watch when you care about rare actions.

```bash
python -m pymc_bot train --dataset pymc-playtime/dataset.jsonl --steps 4000 --engine transformer
```

### Training from the panel

`PyMC_Bot → Player model` card: dataset path, steps, engine, checkpoint interval, then
**Train on playtime** — progress bar, live loss/val, and the checkpoint table with
**Activate** and **Export to Ollama**. **Resume** continues the selected run, and
**Demo: simulate 5 min** trains on synthetic playtime when you just want to see it work.

The trained brain goes through the same permission gates as every other brain: **looking around and
idling are always allowed, but moving, mining/using and attacking follow the switches in the panel**.
A predicted action that a setting forbids has its probability set to zero before the model picks (and
if everything it wants is disabled, it waits), so a model trained on your playtime can never do
something you turned off.

## 3. Run it

```bash
python -m pymc_bot run --think trained --run-name playtime-dataset --backend simulated
```

or in the panel: `Brain → trained — the model trained on playtime`, then **Start AI**.

The policy predicts a burst of play (default 0.4 s) and applies it with the same low-level
controls the mod recorded:

| predicted action | what the bot does |
| --- | --- |
| `forward` / `move` | holds `forward` (and `sprint` when the predicted step is fast) |
| `back` / `left` / `right` | holds that key |
| `jump` / `sneak` | taps jump / crouch |
| `look` | turns by the predicted `dyaw`/`dpitch` |
| `attack` / `use` | swings the arm (right/left click) |
| `hold` | hotbar switch (the bridge does not expose it yet) |
| `none` | stands still, exactly like the player did |

Turn deltas are clamped per burst and the bot speaks **radians** while the mod records
**degrees** — `pymc_bot/local_model.py` does that conversion, which is why a model trained
on your playtime moves in the direction you did instead of spiralling.

Other brains still work: `--think heuristic`, `--think ollama`, or `auto` (Ollama when
enabled, else the heuristic). The anti-AFK keeper always keeps every bot alive, and a
trained brain counts as activity, so the two never fight.

### Ollama, optionally

`python -m pymc_bot train ... --ollama-model pymc-playtime` (or the panel's
**Export to Ollama**) writes a Modelfile next to the checkpoint and creates an Ollama model
whose system prompt describes the learned habits, so you can pick it in the config page
like any other model. The native policy above is the faithful one — it reproduces your
inputs; the Ollama persona is a chat-friendly wrapper around the same statistics.

## 4. Tests

```bash
python -m pytest -q                      # includes the training/policy suite
python -m pytest tests/test_train.py tests/test_local_model.py -q
pytest --cov=pymc_bot
```

`tests/test_features.py` pins the encoder layout (so checkpoints stay loadable),
`tests/test_train.py` covers checkpoints/resume/metrics/service, `tests/test_local_model.py`
drives a simulated bot with a real trained model, and `tests/test_training_api.py` covers the
HTTP API end to end. Transformer tests are skipped when PyTorch is not installed.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `no playtime samples found` | Record with the mod first, or pass `--simulate 5`. |
| Model only ever walks | Train on more/varied playtime, lower `--lr`, or raise `temperature` in the config (`training.temperature`); the class weights already help. |
| Model spins in circles | You are running an old checkpoint from before the degree/radian conversion — retrain from the same dataset. |
| `the transformer engine needs PyTorch` | `pip install -r requirements-train.txt`, or use `--engine mlp`. |
| Bot never uses the model | `agent.mode` must be `trained` and `training.active_run` must point at a checkpoint (`python -m pymc_bot models`). |
| Val accuracy looks low but play looks right | Compare `val_balanced_accuracy` too: rare actions are weighted up on purpose. |

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

Run `/pymc advanced on` before recording (and `/pymc train` afterwards) if you want the recording to
also carry entities and items — see [advanced training](#advanced-training-entities-and-items).

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

### Advanced training: entities and items

Basic recordings describe the world in coarse buckets ("hostile", "passive", "resource block"). Advanced
training instead **learns the entity and item words your playtime actually contains** and gives each one its
own inputs:

| Signal | What the model gets per word |
| --- | --- |
| entity | is it here, how close is it (1 = touching, 0 = out of range), where is it relative to your view (sin/cos bearing) |
| item | is it carried, how many (log-ish, capped), is it in your hand right now |
| globals | weapon/tool/food/armour counts, what is lying on the ground and how close, closest threat, closest player, entity and player density |

So a model can learn "a creeper four blocks away while I hold a sword" separately from "a cow 20 blocks
away while I hold a pickaxe" — and it cannot learn the accidental rule "something is nearby, therefore
swing", which is what makes a basic model punch every player that walks past it.

**Record it (mod).** Turn the recording on before you play:

```
/pymc advanced on          # entities, items, held item, armour and drops are written from now on
/pymc entities 24          # how many entities per sample (default 24, 0 = none)
/pymc drops 8              # how many dropped stacks per sample (default 8)
/pymc advanced status      # what is being recorded
/pymc record start
... play normally ...
/pymc record stop
/pymc export               # dataset.jsonl now carries entities/items/held_item/armor/ground_items
```

**Train it.** Either tick *Advanced mode (entities + items)* in the panel or pass the flag:

```bash
python -m pymc_bot train --dataset "<gameDir>/pymc-playtime/dataset.jsonl" --advanced --steps 5000
python -m pymc_bot train --dataset ... --advanced --entity-slots 32 --item-slots 48   # bigger vocabularies
python -m pymc_bot train --dataset ... --inspect   # shows the vocabulary and its coverage before you train
```

The trainer ranks every entity and item by how often it appears, keeps the seeds it knows matter (mobs,
players, any sword/pickaxe/food/…) even when they are rare, and stores **the learned words in the
checkpoint and in `model.json`**. That is what makes the model runnable: the bot encodes the live world
into exactly the same slots, so no mapping can drift. `--engine transformer` works with advanced mode too.

Check the numbers before you trust it:

```bash
python -m pymc_bot train --dataset ... --inspect | head -40   # entity_coverage / item_coverage
```

`entity_coverage` is how much of what you saw the vocabulary can describe; if it is low, raise
`--entity-slots`/`--item-slots` or record more playtime. The panel shows the same numbers, plus the words
the newest model learned, under **Player model → Advanced training**.

Notes:

* A basic recording has no entity/item data, so advanced mode has nothing extra to learn from — the
  trainer still runs, it just learns the same features as before (`--inspect` says so).
* Advanced recordings are bigger (roughly 2-4x per sample with 24 entity slots) and training is a little
  slower per step; the extra inputs are usually worth it as soon as other players or mobs are around.
* The live bridge reports the players it can see and the inventory; mobs and ground drops come from the
  mod's recording. Columns the bridge cannot fill degrade to "nothing there" rather than inventing data.

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

`tests/test_features.py` pins the encoder layouts (basic v1 and advanced v2, so checkpoints stay loadable),
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
| Advanced model behaves like a basic one | `--inspect` shows an empty vocabulary: the recording was made before `/pymc advanced on`, so there is no entity/item data to learn from. |
| An advanced checkpoint refuses to load | The vocabulary in `model.json` and in the checkpoint must match the build; retrain (a basic model cannot be upgraded in place). |

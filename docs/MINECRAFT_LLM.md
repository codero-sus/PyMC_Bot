# Your own Minecraft LLM (Kaggle, 2× T4)

A general model like `llama3.2` has to read a long rulebook every turn and still guesses about Minecraft.
This guide trains **one small model that is both the bot's brain and a Minecraft expert**:

* it picks the next action with the same JSON the bot already uses (`thought`, `goal`, `action`, fields),
* it chats like a player and answers questions in game chat ("how much damage does a warden do?"),
* it knows mobs, crafting and mechanics **without a web search**,
* it needs only a **short system prompt**, so turns are faster.

The model runs locally in Ollama, like now. Training takes about 1–2 hours on Kaggle's free GPU T4 ×2.

```
your PC                                Kaggle (GPU T4 x2)                      your PC
python -m pymc_bot llm build   ──►   PyMC_Minecraft_LLM_Kaggle.ipynb   ──►   python -m pymc_bot llm import
(train.jsonl + eval.jsonl +           (Unsloth QLoRA, eval, GGUF export)       (Ollama model + bot settings)
 train_minecraft_llm.py)
```

## Which size?

`python -m pymc_bot llm presets` prints this list too. Every preset trains on **one** T4.

| Preset | Base model | Train (GPU) | Run on your PC | File | Why pick it |
| --- | --- | --- | --- | --- | --- |
| **`qwen3-4b`** (default) | Qwen3-4B-Instruct-2507 | ~7 GB, ~1–2 h | ~3.5 GB VRAM (or CPU) | ~2.5 GB | Best quality for its size. It has no "thinking" step, so replies come back as quick JSON. It runs on most PCs and next to several bots. |
| `qwen3-8b` | Qwen3-8B | ~11 GB, ~2× slower | ~6–8 GB VRAM | ~5 GB | Chats better and knows more. Pick it if your PC has a 8 GB+ GPU. It's trained with thinking switched off. |
| `qwen3-14b` | Qwen3-14B | ~15 GB (tight), ~4× slower | ~10–12 GB VRAM | ~9 GB | The biggest that trains on a T4; add `--split` to use both cards. Knows the most, but needs a strong GPU and answers slower. |
| `llama-3.2-3b` | Llama-3.2-3B-Instruct | ~6 GB, fastest | ~2.5 GB | ~2 GB | For weak PCs or many bots at once. It's fast but makes more mistakes. |
| `qwen3.5-4b` | Qwen3.5-4B | ~10 GB, **slow** | ~4 GB | ~3 GB | Experimental. A T4 has no bfloat16 and Qwen3.5 overflows in fp16, so Unsloth trains it in float32 on Kaggle. That makes it several times slower. |

**What the second T4 does:** a 4B–14B LoRA fits on one card, and splitting a small model only slows it
down, so the script uses GPU 0 by default. Use `--split` (Unsloth's `unsloth_balanced` device map) only for
`qwen3-14b`, when memory is tight. 27B+ models are left out on purpose: they would train, but most
PCs can't run them as a bot brain.

## 1. Build the training data (your PC)

```bash
python -m pymc_bot llm build                      # -> llm_data/
```

| Source | What it adds | Notes |
| --- | --- | --- |
| `mobs` | Q&A for every mob in `pymc_bot/data/mobs.json`: hostile?, damage per difficulty and to other mobs, health, where/when it spawns, the version it was added in, drops, taming, breeding | always |
| `facts` | hand-checked basics: recipes, which pickaxe mines what, Nether portal, beds, End, phantoms… | always |
| `rules` | game situations rendered **exactly like the real turn prompt**, answered with the right action: flee creepers/wardens, fight zombies when healthy, run when hurt, eat when hungry, give items on request, wear armour, sleep at night, keep a goal, change plan after failures, wave when chat is off, look up modded mobs | `--turns 6000` |
| `chat` | players asking the bot Minecraft questions in chat, answered with a short `say` | part of `--turns` |
| `wiki` | page intros + Obtaining / Usage / Behavior / Spawning / Breeding / Taming / Drops sections from the [Minecraft Wiki](https://minecraft.wiki) API. April Fools, Education, Dungeons/Legends/Earth, removed and upcoming pages are skipped. | `--wiki-pages 600`; cached in `llm_data/wiki_cache`; `--no-wiki` for offline. Text is CC BY-NC-SA 3.0, fine for this personal, non-commercial project. |
| `teacher` | situations answered by a bigger model **you** run, with the full built-in prompt; only valid, allowed decisions are kept | `--teacher ollama --teacher-model qwen3:32b --teacher-turns 500` or `--teacher cortex` |
| `recorded` | real turns the bot played with an LLM where the action worked | tick **record LLM turns for training** in the panel (`agent.record_turns`); `llm_turns.jsonl` is used automatically |
| `playtime` | your own playtime recorded by the PyMC Bot mod, turned into "what you did next" (fight / mine / goto) | `--playtime <dataset file or folder>` |

The folder contains `train.jsonl`, `eval.jsonl` (5 %, every source represented, with the expected
action/keywords), `dataset_card.json` (counts, licences), `dataset-metadata.json` (for the Kaggle CLI) and
`train_minecraft_llm.py` (the training script). The hidden behaviour prompts are **never** written to the
files: examples carry only the behaviour's key and public name.

## 2. Train on Kaggle

1. kaggle.com → **Datasets → New Dataset** → upload the files in `llm_data/` (not `wiki_cache`).
   Or use the CLI: edit the `id` in `dataset-metadata.json`, then `kaggle datasets create -p llm_data`.
2. **Code → New Notebook → File → Import Notebook** → `training/llm/PyMC_Minecraft_LLM_Kaggle.ipynb`.
3. **Add Input** → your dataset. **Settings** → Accelerator **GPU T4 ×2**, Internet **on**.
4. Set `PRESET` in the second cell if you like, then press **Run All**. (For a 5-minute test, first use `EXTRA = "--max-steps 60"`.)

The script trains a LoRA adapter (4-bit QLoRA, r=32, loss on the replies only). It then tries
`--eval-limit 150` held-out rows and exports `q4_k_m` GGUF. When it finishes, the output folder
`/kaggle/working/pymc-minecraft` holds:

```
pymc-minecraft-qwen3-4b.Q4_K_M.gguf   the model for Ollama
Modelfile                             template, sampling, stop tokens
pymc_model.json                       preset, loss, eval scores
lora/                                 the adapter (optional, for training further)
```

Useful flags: `--epochs 2`, `--max-steps N`, `--gguf q5_k_m|q8_0`, `--model <any Unsloth id>`,
`--hub-repo you/pymc-minecraft` (with an `HF_TOKEN` Kaggle secret), `--dry-run` (checks the data without a GPU).

## 3. Use it (your PC)

Download `pymc-minecraft` from the notebook's **Output** tab, then:

```bash
python -m pymc_bot llm import pymc-minecraft      # or a .gguf file
```

This writes a Modelfile pointing at the GGUF and runs `ollama create pymc-minecraft`. It then switches the bot to it:
`ollama.enabled`, `ollama.model = "pymc-minecraft"`, `agent.mode = "ollama"`, and **Minecraft-trained model
(short prompt)** (`agent.compact_prompt`). `--no-activate` only creates the model; `--name` picks another name.

Check that it really got better:

```bash
python -m pymc_bot llm eval --model pymc-minecraft      # valid JSON % | right action % | right answers %
python -m pymc_bot llm eval --model llama3.2            # the general model, for comparison
```

## Train it again later

Tick **record LLM turns for training**, let the bot play for a while, then run `llm build` again. The
recorded turns where the action worked are added. If you want, add `--teacher ollama --teacher-model <big model>`
and `--playtime <your recordings>`. Upload the new files as a new dataset version and run the notebook again.

## Good to know

* The search action still works. The model knows the common facts, and for modded mobs it learned to
  `search` first.
* A custom system prompt (`ollama.system_prompt`) still wins over the short prompt, same as before.
* The mob database still feeds the prompt: facts about the mobs around the bot are added every turn, and
  `search` answers mob questions from the database first. So exact numbers stay exact, even with a small model.
* Personal, non-commercial use only (PolyForm Noncommercial 1.0.0), and the base models' licences apply
  (Qwen: Apache 2.0, Llama 3.2: Llama Community License).

# Minecraftism: your own Minecraft LLM (Kaggle, 2× T4)

A general model like `llama3.2` has to read a long rulebook every turn and still guesses about Minecraft.
**Minecraftism** is a small model trained to be **both the bot's brain and a Minecraft expert**:

* it picks the next action with the same JSON the bot already uses (`thought`, `goal`, `action`, fields),
* it chats like a player and answers questions in game chat ("how much damage does a warden do?"),
* it knows mobs, crafting and mechanics **without a web search**,
* it needs only a **short system prompt**, so turns are faster,
* the **VL** editions also *see*: they read screenshots of the game,
* the **Thinking** editions reason first (`<think>…</think>`), then answer,
* outside the game it is an **everyday assistant** (questions, writing, code, advice, chat) with its own
  friendly system prompt, and it can be trained **uncensored** (on by default in the notebook).

It runs locally in Ollama, like now. Each edition trains in about 1–3 hours on Kaggle's free GPU T4 ×2,
and the notebook can publish it to Hugging Face for you.

```
your PC                                Kaggle (GPU T4 x2)                      your PC
python -m pymc_bot llm build   ──►   PyMC_Minecraft_LLM_Kaggle.ipynb   ──►   python -m pymc_bot llm import
(train.jsonl + eval.jsonl +           (Unsloth QLoRA, eval, GGUF export,       --hf you/Minecraftism-4B-GGUF
 images/ + the training script)        upload to Hugging Face)                 (Ollama model + bot settings)
```

## The four editions

`python -m pymc_bot llm presets` prints this list too. Every edition trains on **one** T4.

| Edition | Base model | Hugging Face repo | Train | Run on your PC | Pick it when |
| --- | --- | --- | --- | --- | --- |
| **`minecraftism`** (default) | Qwen3-4B-Instruct-2507 | `Minecraftism-4B-GGUF` | ~7 GB, ~1–2 h | ~3.5 GB, file ~2.5 GB | You want the bot's brain. Quick JSON answers, works next to several bots. **Recommended.** |
| `minecraftism-thinking` | Qwen3-4B-Thinking-2507 | `Minecraftism-4B-Thinking-GGUF` | ~7 GB, ~1.5–3 h | ~3.5 GB | Tricky situations (low health + mobs + a request in chat). It's slower: it thinks before every turn. |
| `minecraftism-vl` | Qwen3-VL-4B-Instruct | `Minecraftism-VL-4B-GGUF` | ~9 GB, ~2–3 h | ~4.5 GB, + ~0.8 GB projector | The bot should *look* at the world: it gets a picture of its view every few turns. |
| `minecraftism-vl-thinking` | Qwen3-VL-4B-Thinking | `Minecraftism-VL-4B-Thinking-GGUF` | ~9 GB, ~3 h+ | ~4.5 GB + projector | Pictures and thinking. The slowest and the most careful. |

The thinking editions are trained on **reasoning traces written by built-in rules**: what the bot's health,
food, the time and the nearby mobs mean, who said what in chat (and what the mob database knows about the
mob they asked about), what failed last time, and which action follows from that. No teacher model is needed.
Qwen's Thinking models must be fine-tuned *with* traces; training them on bare answers would make them
forget how to reason, so every row has one.

The older sizes (`qwen3-4b`, `qwen3-8b`, `qwen3-14b` with `--split`, `llama-3.2-3b`, `qwen3.5-4b`)
still work for text-only models.

**What the second T4 does:** a 4B LoRA fits on one card, and splitting a small model only slows it
down, so the script uses GPU 0 by default. `--split` (Unsloth's `unsloth_balanced` device map) is for `qwen3-14b`.

## 1. Build the training data (your PC)

```bash
python -m pymc_bot llm build                                        # text editions -> llm_data/
python -m pymc_bot llm build --playtime <mod folder> --screenshots  # + pictures for the VL editions
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
| `vision` | for every screenshot: the picture + the turn prompt → what you did next, and "what do you see?" → a description written from that moment's recorded data (blocks, mobs in the camera's view and on which side, held item, health) | `--screenshots [PATH]` (no PATH = the `--playtime` folder); needs mod 0.10+ |

**Recording screenshots:** in game, `/pymcbot screenshots on` (or the mod's settings page) makes the recorder
save a small picture of your screen every few seconds next to the samples (`shots/`, off by default because
it uses disk space). Play normally, export as usual, then build with `--screenshots`. The pictures are shrunk
to 512 px (as JPEG if Pillow is installed, otherwise the mod's PNGs are kept).

The folder contains `train.jsonl`, `eval.jsonl` (5 %, every source represented, with the expected
action/keywords), `images/` (when there are pictures), `dataset_card.json` (counts, licences),
`dataset-metadata.json` (for the Kaggle CLI) and `train_minecraft_llm.py` (the training script).
Every row also carries its `reasoning` trace, which only the thinking editions train on. The hidden
behaviour prompts are **never** written to the files: examples carry only the behaviour's key and public name.

## 2. Train on Kaggle

1. kaggle.com → **Datasets → New Dataset** → upload the files in `llm_data/` (with `images/`, without `wiki_cache`).
   Or use the CLI: edit the `id` in `dataset-metadata.json`, then `kaggle datasets create -p llm_data -r zip`.
2. **Code → New Notebook → File → Import Notebook** → `pynb/PyMC_Minecraft_LLM_Kaggle.ipynb`.
3. **Add Input** → your dataset. **Settings** → Accelerator **GPU T4 ×2**, Internet **on**.
   No dataset? With Internet on, the notebook downloads the ready-made kit (`pymc-minecraft-llm-kit.zip`)
   from the newest GitHub release. It has no recorded playtime or screenshots; the VL editions then use
   streamed pictures (`STREAM_PICTURES`).
   If your dataset was built by an older PyMC Bot, the notebook fetches the newest `train_minecraft_llm.py`
   from the release, so the new settings (uncensored, tests, everyday chat) still work.
4. To publish on Hugging Face: **Add-ons → Secrets → Add secret**, name `HF_TOKEN`, value = a **write** token from
   huggingface.co/settings/tokens, and tick it for this notebook. The notebook reads it into the environment for
   the training script and never prints it. Without the secret the models simply stay in `/kaggle/working`.
5. Edit the settings cell (below), then press **Run All**. First time? Set `QUICK_TEST = True`: 30 steps,
   small eval, no export or upload. It checks the whole pipeline in about 10–15 minutes.

The settings cell:

| Setting | Default | What it does |
| --- | --- | --- |
| `EDITIONS` | `["minecraftism"]` | editions to train this session, in order |
| `DATA` | `"auto"` | your dataset (Add Input), else the kit from the newest GitHub release; or a path |
| `TRAINER` | `"auto"` | the dataset's trainer, or the newest from GitHub when it is too old; `"latest"` / `"dataset"` |
| `UNCENSORED` | `True` | abliterated base, refusal-free chat data, `-Uncensored` names ([below](#uncensored-and-everyday-use)) |
| `GENERAL_CHAT` | `3000` | everyday chat rows mixed in, so it stays a good daily assistant (`0` = Minecraft only) |
| `COMMERCIAL_ONLY` | `True` | only data whose licence allows commercial use ([below](#commercial-use-data)) |
| `MINECRAFT_QA` | `20000` | Minecraft Q&A: generated from minecraft-data, or streamed from a 695k-row set ([below](#big-minecraft-datasets-streamed)) |
| `STREAM_PICTURES` | `3000` | VL editions: Minecraft pictures with questions and answers, streamed from Hugging Face |
| `PERSONA_MIX` | `0.5` | share of Minecraft Q&A rows that use `SYSTEM_PROMPT` (game turns keep the bot's prompt) |
| `SYSTEM_PROMPT` | Minecraftism prompt | the model's default personality: trained in and written into the `Modelfile` |
| `HF_REPO`, `HF_PRIVATE`, `HF_LORA` | `"auto"`, `False`, `False` | Hugging Face upload |
| `QUICK_TEST` | `False` | short test run of the whole pipeline |
| `RUN_TESTS` | `True` | test suite after training ([below](#testing)) |
| `TEST_GGUF` | `False` | also installs Ollama in the notebook and chats with the exported GGUF |
| `CLEAN_AFTER_UPLOAD` | `True` | deletes the big local files once an edition is on Hugging Face |
| `EPOCHS`, `EXTRA` | `1`, `""` | more training / any other trainer flag |

Before any training starts, the notebook checks the GPU, the internet, the free disk, the dataset and
the `HF_TOKEN` secret, and prints the trainer flags.

**Time limit:** a Kaggle GPU session stops after **12 hours**, and the weekly GPU quota is about 30 hours.
Train one or two editions per session, e.g. `["minecraftism", "minecraftism-thinking"]` today and the VL
editions another day. The notebook dry-runs every edition first and skips the ones that can't train
(a VL edition with no pictures and `STREAM_PICTURES = 0`, for example) instead of failing hours later.

The script trains a LoRA adapter (4-bit QLoRA, r=32, loss on the replies only; VL editions train the
language layers and keep the vision encoder frozen unless you pass `--vision-layers 1`). It evaluates on
held-out rows, exports `q4_k_m` GGUF and, for VL editions, the **vision projector** (`mmproj`). Each edition
gets its own folder, e.g. `/kaggle/working/minecraftism-vl`:

```
minecraftism-vl-4b.Q4_K_M.gguf          the model
mmproj-minecraftism-vl-4b.F16.gguf      the vision projector (VL editions only)
Modelfile                               for ollama create (both FROM lines, sampling, stop tokens)
params                                  sampling for "ollama pull hf.co/..." (written when uploading)
README.md                               the model card shown on Hugging Face
pymc_model.json                         edition, loss, eval + test scores, system prompt, what PyMC Bot reads on import
tests.json                              every test question and answer (with RUN_TESTS)
lora/                                   the adapter (optional, for training further)
```

With `HF_REPO = "auto"` each edition is uploaded to `<your HF name>/<repo from the table>`, e.g.
`you/Minecraftism-4B-GGUF`, and the local copy is cleaned up afterwards (`CLEAN_AFTER_UPLOAD`) so four
editions fit into Kaggle's 20 GB. If the upload fails, the files stay in `/kaggle/working`.

Useful flags (`EXTRA`): `--epochs 2`, `--max-steps N`, `--gguf q5_k_m|q8_0`, `--hub-private`,
`--hub-lora` (also upload the adapter), `--image-side 448` (smaller pictures, faster VL training),
`--model <any Unsloth id>`, `--dry-run` (checks the data without a GPU).

## Uncensored and everyday use

**System prompt.** Minecraftism does not think it is "a bot". Its default system prompt (`SYSTEM_PROMPT` in
the notebook, `--system-prompt-file` for the trainer) says it is *Minecraftism, an AI assistant made by
codero-sus*: a general assistant first, glad to help with everyday life, questions, writing, studying, code
and advice, who also knows Minecraft inside out. It should be honest, treat the user as an adult, skip
lectures and unasked-for disclaimers, have a personality, and reply in the user's language. When a message
asks for a specific format, such as PyMC Bot's JSON action, it replies in that format.
The prompt is trained in (identity rows, everyday chat, `PERSONA_MIX` of the Minecraft Q&A) and written into the
`Modelfile` as `SYSTEM`, so `ollama run minecraftism-uncensored` uses it. PyMC Bot sends its own short prompt for
game turns, so play is unchanged.

**Uncensored** (`UNCENSORED = True` / `--uncensored`):

* The model trains on an **abliterated** copy of the base, made by huihui-ai, where the "refusal direction"
  was removed from the weights:
  * `Huihui-Qwen3-4B-Instruct-2507-abliterated`
  * `…-Thinking-2507-abliterated`
  * `Huihui-Qwen3-VL-4B-Instruct-abliterated`
  * `…-VL-4B-Thinking-abliterated`
* Refusals, "as an AI language model" lines and foreign identities ("I'm ChatGPT") are dropped from the
  everyday chat data.
* A handful of frank example answers is added: roasts, dark jokes, trash talk, honest opinions.
* The files are named `…-uncensored` (Hugging Face: `Minecraftism-4B-Uncensored-GGUF`, …), and the Ollama model is
  named `minecraftism-uncensored`. The model card says what was done.

An uncensored model answers what you ask; you are responsible for how you use it. Set `UNCENSORED = False`
for the stock bases.

**Everyday chat data:** `GENERAL_CHAT` rows come from
[`mlabonne/FineTome-100k`](https://huggingface.co/datasets/mlabonne/FineTome-100k), a high-quality mixed chat
dataset; you can pick another with `--general-dataset`. The thinking editions get half as many rows with real
reasoning traces instead, from [`unsloth/OpenMathReasoning-mini`](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini)
(`--reasoning-dataset`). Training a thinking model on answers without traces would make it forget how to think.
The data is downloaded on Kaggle, and `--dry-run` only reports what it would add.

## Commercial-use data

`COMMERCIAL_ONLY = True` (the default; trainer flag `--commercial`) trains only on data whose licence allows
commercial use:

| Part | Dataset | Licence | What the trainer does with it |
| --- | --- | --- | --- |
| Everyday chat (`GENERAL_CHAT`) | [`OpenAssistant/oasst_top1_2023-08-25`](https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25) | Apache-2.0 | ~13k chats written by people, multilingual |
| Thinking traces (thinking editions) | [`nvidia/Nemotron-Post-Training-Dataset-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1), `math` split | CC BY 4.0 | maths with real `<think>` traces (DeepSeek-R1 / Qwen3 answers); long traces are skipped |
| Minecraft Q&A (`MINECRAFT_QA`) | [`PrismarineJS/minecraft-data`](https://github.com/PrismarineJS/minecraft-data), Java 1.21.8 | MIT | about 3,300 exact facts made into Q&A: recipes with their layout, which pickaxe a block needs, block drops, light levels, hardness and blast resistance, stack sizes, durability, food, mob drops and hitboxes, enchantments, effects, biomes. Each fact has up to 3 phrasings, so about 10k rows at most |
| Pictures (`STREAM_PICTURES`) | [`shinohararin/minecraft-raytrace`](https://huggingface.co/datasets/shinohararin/minecraft-raytrace) | MIT | 10,222 first-person 1.21.8 frames with raycast labels. Two exact Q&A per frame: biome, time of day, rain or snow, the block at the crosshair and its distance, the blocks in view, the mobs in view, a short description |

* The kit's **Minecraft Wiki rows** (CC BY-NC-SA 3.0) are left out. The rest of the kit stays: the PyMC Bot
  game turns, the mob database facts, the checked crafting facts and your own recordings.
* A dataset you pick yourself (`--general-dataset` and the others) is checked: a known non-commercial one stops
  the run, and an unknown one prints a note to check its licence.
* **The model card lists every source with its licence and row count.** That is also the attribution
  CC BY 4.0 asks for.
* minecraft-raytrace is read **one ~330 MB parquet shard at a time**, and only the frame and label columns are
  read, not the voxel dumps. 3,000 rows need about 3 shards. Frames are saved as JPEG and deleted after training.
  `QUICK_TEST` starts with the small last shard (~14 MB).
* The trainer reads the raycast labels from the dataset's README layout. If a capture's labels or mob list look
  different, those questions are skipped for that frame. Biome, time and weather come from plain columns and
  always work.
* minecraft-data says some of its values came from wiki.vg and the Minecraft Wiki long ago. Its mob drop amounts
  are simplified, so the trainer uses only the item names and the odds.
* `COMMERCIAL_ONLY = False` uses the bigger non-commercial sets instead (below), plus the wiki rows.
* The model card still releases the fine-tune under PolyForm Noncommercial 1.0.0, the project's licence. Because
  the data now allows commercial use, the data is no longer a reason you couldn't pick a different licence for your
  own fine-tune. The base models are Apache-2.0, including the abliterated copies. Not legal advice. "Minecraft" is Mojang's trademark, so check their usage guidelines
  before you sell anything with the name.

## Big Minecraft datasets, streamed

With `COMMERCIAL_ONLY = False`, two big public datasets are mixed in without downloading them whole:

| Setting (trainer flag) | Default | Dataset | What it adds |
| --- | --- | --- | --- |
| `MINECRAFT_QA` (`--minecraft-qa N`) | `20000` | [`naklecha/minecraft-question-answer-700k`](https://huggingface.co/datasets/naklecha/minecraft-question-answer-700k) (694,814 rows, CC BY-NC-SA 3.0) | Minecraft Q&A written from ~18k wiki pages |
| `STREAM_PICTURES` (`--stream-vision N`) | `3000` | [`CraftJarvis/minecraft-vlp`](https://huggingface.co/datasets/CraftJarvis/minecraft-vlp) (17 GB) | screenshots with captions and visual Q&A, for the VL editions |

How the streaming works:

* **The Q&A set** is read with `datasets` in streaming mode. A shuffle buffer picks a random sample, and only
  those rows are downloaded.
* **The Q&A set is synthetic**, and some of its answers are wrong, so the sample is filtered: refusals,
  "I'm not sure" lines, known nonsense, repeated questions, and very short or very long answers are dropped.
  The checked mob and crafting facts from `llm build` still train alongside.
* **The pictures** come from the caption and visual Q&A files of `minecraft-vlp`. Their JSONL is read line by
  line, and the sample is spread over the whole file. The images themselves live in zip files of 0.6–1.8 GB,
  so each chosen picture is read out of the remote zip with HTTP range requests.
  `STREAM_PICTURES = 3000` downloads roughly 150–300 MB of pictures, not the archives. They are deleted after
  training.
* **VL editions now train without your own screenshots.** If the data has no pictures, a few streamed ones are
  held out for the eval and the test suite.

The 106 GB `CraftJarvis/minecraft-vla-sft` set is **not** used: its answers are raw keyboard and mouse tokens
for the JARVIS model, not text a chat model can learn from. Without internet, or if a stream fails, the
trainer prints a warning and trains on the rest.

## Testing

`RUN_TESTS = True` (`--test`) asks the freshly trained model, before export:

* 3 bot turns and 3 Minecraft questions from `eval.jsonl`, scored for valid JSON, the right action and the right
  answer,
* "who are you?", which should answer "Minecraftism, by codero-sus",
* everyday prompts (money, a text message, a workout) and a Minecraft question,
* edgy-but-harmless prompts, which check that it doesn't refuse,
* a screenshot, for VL editions.

Every question and answer is printed. They are saved as `tests.json`, and the summary is added to
`pymc_model.json` and the model card. It looks like this:
`0 refusal(s) in 15 prompts, identity ok, bot JSON 3/3`.

`TEST_GGUF = True` (`--test-gguf`) also installs Ollama in the notebook, `ollama create`s the exported GGUF
with its Modelfile and chats with it (VL: with a picture). This catches a broken export before you download
3 GB.

Testing again later: set `CLEAN_AFTER_UPLOAD = False` (keeps `lora/`), then use `RETEST = ["minecraftism"]`
(`--test-only`). `CHAT_WITH = "minecraftism"` chats with a text edition right in the notebook (interactive, so
not for "Save & Run All").

## 3. Use it (your PC)

From Hugging Face (needs [Ollama](https://ollama.com)):

```bash
python -m pymc_bot llm import --hf you/Minecraftism-4B-GGUF            # ollama pull hf.co/...:Q4_K_M
python -m pymc_bot llm import --hf you/Minecraftism-VL-4B-GGUF --quant Q4_K_M
```

Or download an edition folder from the notebook's **Output** tab:

```bash
python -m pymc_bot llm import minecraftism-vl                          # or a .gguf file
```

The import creates the Ollama model, named after the edition (`minecraftism`, `minecraftism-vl`,
`minecraftism-uncensored`…; `--name` picks another), with the model's system prompt from its `pymc_model.json`.
Then it switches the bot to it: `ollama.enabled`, `ollama.model`, `agent.mode = "ollama"` and
**Minecraft-trained model (short prompt)** (`agent.compact_prompt`). VL editions also turn on
**let the model see** (`agent.vision`). `--no-activate` only creates the model.

Check that it really got better:

```bash
python -m pymc_bot llm eval --model minecraftism        # valid JSON % | right action % | right answers %
python -m pymc_bot llm eval --model qwen3:4b            # the base model, for comparison
```

### Letting the bot see (VL editions)

With **let the model see** on, the bot renders its first-person view and attaches it to every
`agent.vision_every`-th turn (default 2) for Ollama (`images`) or Cortex (OpenAI-style `image_url`).
Rendering happens in the mineflayer bridge with prismarine-viewer, which needs three extra packages.
They compile native WebGL code, so they're an optional install:

```bash
npm install --prefix pymc_bot/node prismarine-viewer node-canvas-webgl three@0.128.0
```

On a Linux server without a screen, run the bot under `xvfb-run` if WebGL can't start. Without these
packages the bot logs one "no picture this turn" line and keeps playing on text alone. The rendered view
is simpler than the real game (no entity models, basic lighting), but it shows terrain, water, lava and
caves. Any other Ollama vision model (`qwen3-vl`, `gemma3`…) can use it too.

Ollama runs Qwen3-VL GGUFs with a separate projector; if a VL edition only answers in text on your Ollama
version, update Ollama, or run it with llama.cpp: `llama-server -m minecraftism-vl-4b.Q4_K_M.gguf
--mmproj mmproj-minecraftism-vl-4b.F16.gguf` and point the bot's Cortex/OpenAI setting at it.

## Train it again later

Tick **record LLM turns for training**, keep `/pymcbot screenshots on` in the mod, play for a while, then
run `llm build` again. The recorded turns where the action worked are added. If you want, add
`--teacher ollama --teacher-model <big model>`. Upload the new files as a new dataset version and run
the notebook again. The same Hugging Face repo gets a new commit.

## Good to know

* The search action still works. The model knows the common facts, and for modded mobs it learned to
  `search` first.
* A custom system prompt (`ollama.system_prompt`) still wins over the short prompt, same as before.
* As a daily assistant: `ollama run minecraftism-uncensored`, or pick it in any Ollama app (Open WebUI and
  others). The built-in system prompt applies whenever the app sends none.
* The mob database still feeds the prompt: facts about the mobs around the bot are added every turn, and
  `search` answers mob questions from the database first. So exact numbers stay exact, even with a small model.
* Thinking editions print `<think>…</think>` before the JSON; the bot strips it (also when only the closing
  tag appears, as Qwen3 Thinking models do).
* Personal, non-commercial use only (PolyForm Noncommercial 1.0.0), and the base models' licences apply
  (Qwen3 / Qwen3-VL: Apache 2.0, Llama 3.2: Llama Community License). The model cards say so too, and list
  the licence of every training source ([commercial-use data](#commercial-use-data)).

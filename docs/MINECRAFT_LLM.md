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
python -m pymc_bot llm build   ──►   train_minecraft_llm.py   ──►   python -m pymc_bot llm import
(train.jsonl + eval.jsonl +           (Unsloth QLoRA, eval, safetensors ->      --hf you/Minecraftism-4B-GGUF
 images/ + the training script)        GGUF quants, upload to HF)              (Ollama model + bot settings)
```

Every edition is published in **two formats**, each in its own Hugging Face repo:

| Repo | Format | Use it with |
| --- | --- | --- |
| `you/Minecraftism-4B` | **safetensors**: the LoRA merged into the base, 16-bit (~8 GB, VL ~9 GB), `tokenizer.json` + configs | transformers, vLLM, further fine-tuning, your own quantisations (AWQ, EXL2) |
| `you/Minecraftism-4B-GGUF` | GGUF: **34 requested formats**, including experimental types (`Q4_K_M` the default; + `mmproj` projector for VL); some may be skipped, the importance matrix, `Modelfile`, `params`, `tokenizer.json` + configs | Ollama, llama.cpp, LM Studio, **PyMC Bot** (`llm import --hf`) |

The safetensors repo is the GGUF repo's name without `-GGUF` (uncensored: `Minecraftism-4B-Uncensored`). Both
model cards link to each other; the safetensors card has transformers and vLLM code. `llm import --hf` on a
safetensors repo uses its GGUF twin automatically, since Ollama runs GGUF.

**The order:** the safetensors are saved, compacted and uploaded first. Then llama.cpp converts *those exact
weights* into one F16 GGUF (in `/kaggle/tmp`, off Kaggle's 20 GB output disk) and `llama-quantize` makes every quant from it.
Each quant is uploaded as soon as it is ready and deleted, so the disk never holds more than the F16 file and one
quant. The GGUF card lists every file with its size (`ollama run hf.co/you/Minecraftism-4B-GGUF:Q8_0` picks one;
`llm import --hf ... --quant Q8_0` does it for the bot). llama.cpp comes from its newest GitHub release (Linux
build + source at the same tag; `llama-quantize` is compiled when the prebuilt one cannot run), or from
`$LLAMA_CPP_DIR`. If that route fails, Unsloth's own GGUF exporter makes only the standard formats and reports the rest as skipped.

`GGUF_QUANTS = "all"` (the default) attempts 34 distinct dense-model formats. Tool support, time, disk space and calibration may limit the files actually produced:

| Group | Quants | 4B files | Use them when |
| --- | --- | --- | --- |
| full precision | `BF16`, `F16` | ~8 GB | making your own quants; `BF16` is converted straight from the safetensors |
| near-lossless | `Q8_0`, `Q6_K` | 4.3 / 3.3 GB | plenty of RAM, best answers |
| recommended | `Q5_K_M`, `Q5_K_S`, **`Q4_K_M`** (default), `Q4_K_S`, `IQ4_XS`, `IQ4_NL` | 2.3-2.9 GB | most PCs |
| low RAM | `Q3_K_L`, `Q3_K_M`, `Q3_K_S`, `IQ3_M`, `IQ3_S`, `IQ3_XS`, `IQ3_XXS` | 1.7-2.2 GB | old laptops, phones |
| tiny | `Q2_K`, `Q2_K_S`, `IQ2_M`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS`, `IQ1_M`, `IQ1_S` | 1.1-1.7 GB | experiments: clearly weaker |
| legacy | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1` | 2.4-3.1 GB | old apps; `Q4_0` is fast on ARM CPUs |
| experimental | `Q1_0`, `Q2_0`, `TQ1_0`, `TQ2_0`, `F32` | varies | low-bit and ternary formats can be poor quality; F32 is expanded from F16 and does not restore precision |

(VL editions add the `mmproj` vision projector, ~0.8 GB.)

**Importance matrix (imatrix).** Before the quants below Q8_0, `llama-imatrix` runs the fresh Q8_0 over
`--imatrix-chunks` (64) x 512 tokens of the edition's own training data (~10 min on the CPU) and every quant below
Q8_0 is made with it: better quality at the same size, and the `IQ1`/`IQ2`/`IQ3_XXS`/`Q2_K_S` quants can't be made
without one (llama.cpp refuses: "the result will be garbage"). The matrix is uploaded too
(`*.imatrix.gguf`), so anyone can make more quants. `MXFP4_MOE` is excluded for this dense model; aliases and COPY
are not separate formats.

**Time.** Export and upload times vary. Unsupported formats, missing imatrix, deadline and disk constraints can leave
files missing; review the model card and logs. Uploads replace files by name in the existing repos, but old files not
replaced may remain from an earlier run. Check revisions before using them. `GGUF_QUANTS = "standard"` makes just
`Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q3_K_M`.

**The model card** (written by the trainer) has a quick start for PyMC Bot, Ollama, llama.cpp (`-hf`) and
`hf download`, a "which file should I download?" guide, the full quant table (size, RAM to run, imatrix, direct
download links), the prompt format, the recommended settings, limitations, the training data and its licences, and
the PolyForm `Required Notice:`. The GGUF repo is linked to the safetensors repo on Hugging Face's model tree
(`base_model_relation: quantized`).

**Why the safetensors are ~8 GB, and why that is the smallest a 4B model gets in 16-bit:** each of the ~4.0 billion
weights takes 2 bytes. A LoRA merge can make the files *bigger* than the base - float32 leftovers (4 bytes a weight)
and a second copy of the ~0.4B-weight embedding matrix as an untied `lm_head` - which is how a 4B model ends up the size
of a 5-6B one. The trainer **compacts** the merge (`--no-compact` turns it off): float32 becomes the model's 16-bit type,
the `lm_head` copy is dropped when `tie_word_embeddings` says the base shares it, `model.safetensors.index.json` is
rewritten, and the report shows the real parameter count (`params_b`). An older trainer also had a bug that could
name Unsloth's ~8 GB F16 intermediate `...Q4_K_M.gguf`: the GGUF for each quant is now picked by its name. Real
*distillation* would mean training a different, smaller model - for small files use the GGUF quants.

## The four editions

`python -m pymc_bot llm presets` prints this list too. A 4B edition fits on **one** T4, so the notebook keeps
both T4s busy ([speed](#speed-both-t4s-every-minute)).

| Edition | Base model | Hugging Face repo | Train | Run on your PC | Pick it when |
| --- | --- | --- | --- | --- | --- |
| **`minecraftism`** (default) | Qwen3-4B-Instruct-2507 | `Minecraftism-4B-GGUF` | ~7 GB, ~1–2 h | ~3.5 GB, file ~2.5 GB | You want the bot's brain. Quick JSON answers, works next to several bots. **Recommended.** |
| `minecraftism-turbo` | Qwen3-4B-Instruct-2507 | `Minecraftism-4B-Turbo-GGUF` | ~7 GB | ~3.5 GB | Concise direct-answer training. Actual speed and quality are unbenchmarked. |
| `minecraftism-turbo-thinking` | Qwen3-4B-Thinking-2507 | `Minecraftism-4B-Turbo-Thinking-GGUF` | ~7 GB | ~3.5 GB | Thinking with bounded Ollama output. Speed and quality are unbenchmarked. |
| `minecraftism-thinking` | Qwen3-4B-Thinking-2507 | `Minecraftism-4B-Thinking-GGUF` | ~7 GB, ~1.5–3 h | ~3.5 GB | Tricky situations (low health + mobs + a request in chat). It's slower: it thinks before every turn. |
| `minecraftism-vl` | Qwen3-VL-4B-Instruct | `Minecraftism-VL-4B-GGUF` | ~9 GB, ~2–3 h | ~4.5 GB, + ~0.8 GB projector | The bot should *look* at the world: it gets a picture of its view every few turns. |
| `minecraftism-vl-thinking` | Qwen3-VL-4B-Thinking | `Minecraftism-VL-4B-Thinking-GGUF` | ~9 GB, ~3 h+ | ~4.5 GB + projector | Pictures and thinking. The slowest and the most careful. |

Thinking editions use `<think>` traces; the regular Instruct edition remains direct-answer. Minecraft Q&A
uses eligible dataset-provided reasoning if available, otherwise a generated answer-plan template (not genuine
chain-of-thought). For other Q&A, optional teacher distillation can replace a plan with a teacher trace while keeping the
original answer; the private Minecraftism rows retain their source traces or fallback plans. Game turns use built-in rule-based plans; other reasoning datasets may supply their own traces.
Neither source traces nor teacher traces are verified facts; even a numeric consistency check cannot establish
correctness. Teacher output is optional and may be unavailable if generation fails or the session runs out of time.

The default uncensored fine-tunes use Huihui's abliterated weights. Their safetensors cards use the original
Qwen checkpoint for upstream Hub-tree grouping while explicitly identifying the actual Huihui training base.
GGUF cards link to the corresponding safetensors fine-tune when available. Custom base overrides are not
misrepresented as Qwen.

**Repair missing GGUFs without training:** set `editions_to_be_trained = "gguf"`. `GGUF_EDITIONS` defaults
to the four text editions. The notebook reads each existing safetensors repository at a pinned revision and
attempts the missing types from the curated 34-format list. It uploads completed files individually without
overwriting existing GGUFs or model cards; re-run to retry missing formats. Existing GGUFs may be from a different
revision than the newly generated files. The private dataset is read for local importance-matrix calibration;
neither the calibration text nor the resulting matrix is uploaded. The importance matrix requires a verified CUDA
GPU and offloads all layers. By default, two editions run concurrently on separate GPUs when at least 48 GB of
scratch space is free; a single edition or `GGUF_GPU_MODE = "merge"` combines available GPUs instead. CPU threads
are divided between workers, and quantization overlaps one bounded upload per edition. Quantization itself uses CPU.
Parallelism depends on space and is not a guaranteed speedup. Discord uses the
existing webhook and role secrets for important events and periodic updates. See `/kaggle/working/gguf-repair-report.json`
for outcomes. Missing model weights, unsupported formats, failed uploads or the session deadline can leave gaps.

**Replace Thinking Uncensored GGUFs:** set `editions_to_be_trained = "gguf_replace"` and
`UNCENSORED = True`. This mode regenerates all formats from the existing pinned Thinking safetensors and replaces
existing GGUFs in the Thinking Uncensored GGUF repository, one file at a time; missing formats are added. It does
not touch the other editions, safetensors or model card. An interrupted run can leave mixed checkpoint revisions;
re-running this mode repeats already replaced formats. Quantization and one bounded upload overlap for speed.

**Allot names without training:** set `editions_to_be_trained = "allot"` and run all notebook cells.
With the current `UNCENSORED` setting it reserves GGUF and safetensors names for all six editions (12 names)
using README-only placeholders. Existing repositories are not modified; account identity is checked via the
Kaggle `HF_TOKEN` secret. No training stack, dataset, model weights or GPU training are run in this mode. Names
are not reserved until this mode actually runs.

**Dataset-first training:** accepted `mr-sus/Minecraftism-dataset` rows train in the first phase; other rows
train afterward on the same weights, with a fresh optimizer. `EPOCHS=1` schedules one pass through each phase,
not two passes through the private dataset. If the first phase is interrupted or step-limited, later training and
on-policy distillation are skipped. Loading data and generating optional teacher answers for other data happen
before model training. The logs and `pymc_model.json` include phase status and scheduled counts; counts do not
prove every row was seen. A session deadline can interrupt training.

The notebook selects **Turbo** and **Turbo Thinking** by default. It favors complete, shorter everyday examples and optionally
requests concise teacher answers, without truncating Minecraft references or game JSON. A shorter answer may
reduce end-to-end latency but does not guarantee faster token generation. Turbo's Ollama profile limits context
to 2048 tokens and output to 128 tokens; change `num_ctx` and `num_predict` for longer tasks. These limits can
truncate input or output and are not applied automatically in other runtimes. Turbo Thinking keeps reasoning traces and defaults in Ollama to a 4096-token context and 512-token output budget,
which includes thinking and may cut off the answer. Four editions take more time and may
not all finish within the session limit. With automatic Hugging Face uploads, the notebook creates README-only
Turbo placeholders for newly created safetensors and GGUF repos before training; existing repositories are left
untouched. Reservation happens only when the notebook runs with appropriate write access, not when it is imported.

The older sizes (`qwen3-4b`, `qwen3-8b`, `qwen3-14b` with `--split`, `llama-3.2-3b`, `qwen3.5-4b`)
still work for text-only models.

**What the second T4 does:** it trains too, see [speed](#speed-both-t4s-every-minute). Splitting one small model's
*layers* across both cards only slows it down, so `--split` (Unsloth's `unsloth_balanced` device map) stays for
`qwen3-14b`. The notebook splits the *work* instead: one edition per T4, or both T4s on one edition's batches (DDP).

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

## 2. Train your model

The notebooks are not included in this branch. `python -m pymc_bot llm build` writes a
`train_minecraft_llm.py` trainer with the dataset; run it with `--help` for its options or
`--dry-run` to validate a dataset and plan without a GPU. Training and uploads require separately
installed dependencies and suitable hardware. The earlier notebook-specific settings below are
historical context, not instructions for a bundled notebook.

## Uncensored and everyday use

**System prompt.** Minecraftism does not think it is "a bot". Its default system prompt (`SYSTEM_PROMPT` in
the notebook, `--system-prompt-file` for the trainer) says it is *Minecraftism, an AI assistant made by
codero-sus*: a general assistant first, glad to help with everyday life, questions, writing, studying, code
and advice, who also knows Minecraft inside out. It should be honest, treat the user as an adult, skip
lectures and unasked-for disclaimers, have a personality, and reply in the user's language. When a message
asks for a specific format, such as PyMC Bot's JSON action, it replies in that format.
The prompt is context for some training examples and is written into the `Modelfile` as `SYSTEM`, so Ollama can
use it by default. The loss trains assistant responses, not the prompt as an unchangeable rule. Identity examples
include both system-prompt and no-system-prompt cases; `SYSTEMLESS_FRACTION = 0.5` also removes the default prompt
from a sample of ordinary training examples, while game, tool and custom instructions are preserved. Identity rows
are excluded from teacher distillation. The test report checks identity with and without the prompt and warns on
failure. These measures encourage identity but cannot guarantee it; a fresh training run is required, and Qwen is
still the base-model lineage. PyMC Bot sends its own short prompt for game turns.

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
| Everyday chat (`GENERAL_CHAT`) | [`OpenAssistant/oasst_top1_2023-08-25`](https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25) | Apache-2.0 | ~13k chats written by people, multilingual. Thinking editions get it too: it fills every everyday row the traces below don't, and with `DISTILL` the teacher writes the thinking for up to `DISTILL_ROWS` of them |
| Tool calling (`TOOL_CALLING`) | [`nvidia/Nemotron-Post-Training-Dataset-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) (`tool_calling`) | CC BY 4.0 | 310k multi-turn tool chats with their tool definitions, by Qwen3-235B-A22B; with and without `<think>` traces |
| Thinking traces (thinking editions) | [`nvidia/Nemotron-Post-Training-Dataset-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1), `math` split | CC BY 4.0 | maths with real `<think>` traces (DeepSeek-R1 answers). Only traces that fit `--max-seq` are used; most are far longer, so expect a few dozen rows (the log says how many) |
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

## Agentic: tool calling, and less bias

A fine-tune on mostly Minecraft rows slowly forgets what the base model could do - and tool calling (what agentic
benchmarks measure) is the first to go. `TOOL_CALLING = 2000` (`--tools`) mixes in multi-turn tool chats from
[`nvidia/Nemotron-Post-Training-Dataset-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1)
(`tool_calling` split, CC BY 4.0, written by Qwen3-235B-A22B - the bases' own family):

* **Qwen3's own format, character for character**: the tool list in the system prompt, `<tool_call>` JSON in the
  answers, tool results as `<tool_response>` turns (never trained on) - a test renders the rows with Qwen's real chat
  template and compares. So it works the same in Ollama, llama.cpp, vLLM or anything that sends `tools`.
* **Thinking editions** train on the chats with a `<think>` trace in every assistant turn; the others on plain ones.
* **When to call, not only how**: ~15% more rows offer tools that don't fit an everyday question, and the answer is a
  normal reply. Half the rows have the Minecraftism system prompt, half only the tool prompt (like benchmarks).
* **Agentic score, base vs fine-tuned**: 60 held-out tool chats it never trained on, plus 12 everyday prompts with
  tools that don't fit. Right function with exactly the right arguments, and no call when none fits. The base model
  is measured on the same prompts with the LoRA switched off, and both scores go into the model card and the summary.

**Bias fixes in the game data** (the next run trains on them; the models uploaded before Oct 27 do not have them):
no bot action takes more than 15% of the game turns (`say` used to be 27%, so the model chatted when it should act);
thinking traces start with what matters (a creeper 3 blocks away, a player talking, low health) instead of always
"Health X/20, food Y/20."; when a trace is too long the context is trimmed, never the decision; and the earlier fixes
(no fixed thinking scaffold, no "simple one" shortcut, any-planks recipes).

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

## Building a dataset

The optional Python collector is `pymc_bot.llm_webdata`; it does not require a bundled notebook.
Review source licences and verify facts before using collected data.

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
**let the model see** (`agent.vision`). `--no-activate` only creates the model. A local VL import
requires the matching `mmproj` file declared by `pymc_model.json` (or an `mmproj` file alongside
the GGUF if no filename is declared). If creation fails or Ollama is missing, the previous bot model
is left unchanged. Support for a model's architecture and projector depends on the Ollama version.

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

Select an **installed vision-capable model** in the panel, then enable **let the model see**.
In the mod, use `/pymcbot bot ai vl <installed-name>` to select an installed Ollama VL model
and enable vision, or use `/pymcbot bot ai model <installed-name>` and `/pymcbot bot ai vision on`;
`/pymcbot bot status` reports whether a frame was captured, sent, unavailable, or the image request
failed. The panel exposes image cadence and width. “Sent” means the image request succeeded, not
that the runtime interpreted the image correctly. Fleet brain overrides still apply.

On a Linux server without a screen, run the bot under `xvfb-run` if WebGL can't start. Without the
packages, a supported Minecraft version, or a working renderer, status reports vision unavailable
and the bot uses text context only. The simplified renderer is not the player's actual game view and
can omit or misrender details; its result depends on the installed viewer version. Images are sent to
the configured model service; enable vision only if that is acceptable. Any other compatible Ollama
or OpenAI-style vision model (`qwen3-vl`, `gemma3`…) can use it too. The screenshot controls used
for recording training data are separate from this live bot renderer.

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

<sub>Made with ❤️ by [codero-sus](https://github.com/codero-sus) · [codero.is-a.dev](https://codero.is-a.dev) · [the-codero.web.app](https://the-codero.web.app/)</sub>

### Import files in model exports

The GGUF folder includes `Modelfile`, `params` and `IMPORTING.md` alongside the default GGUF and available
config/tokenizer files. The safetensors folder also contains its own `Modelfile` and `IMPORTING.md`, next to its
weights and tokenizer/config. Download the full files required by your runtime before importing; vision requires a
projector when exported. Ollama, llama.cpp and other applications vary in architecture and vision support.

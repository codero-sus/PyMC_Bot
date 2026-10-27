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
(train.jsonl + eval.jsonl +           (Unsloth QLoRA, eval, safetensors ->      --hf you/Minecraftism-4B-GGUF
 images/ + the training script)        GGUF quants, upload to HF)              (Ollama model + bot settings)
```

Every edition is published in **two formats**, each in its own Hugging Face repo:

| Repo | Format | Use it with |
| --- | --- | --- |
| `you/Minecraftism-4B` | **safetensors**: the LoRA merged into the base, 16-bit (~8 GB, VL ~9 GB), `tokenizer.json` + configs | transformers, vLLM, further fine-tuning, your own quantisations (AWQ, EXL2) |
| `you/Minecraftism-4B-GGUF` | GGUF: **every quant**, `F16` / `BF16` down to `IQ1_S` (29 files, `Q4_K_M` the default; + `mmproj` projector for VL), the importance matrix, `Modelfile`, `params`, `tokenizer.json` + configs | Ollama, llama.cpp, LM Studio, **PyMC Bot** (`llm import --hf`) |

The safetensors repo is the GGUF repo's name without `-GGUF` (uncensored: `Minecraftism-4B-Uncensored`). Both
model cards link to each other; the safetensors card has transformers and vLLM code. `llm import --hf` on a
safetensors repo uses its GGUF twin automatically, since Ollama runs GGUF.

**The order:** the safetensors are saved, compacted and uploaded first. Then llama.cpp converts *those exact
weights* into one F16 GGUF (in `/kaggle/tmp`, off Kaggle's 20 GB output disk) and `llama-quantize` makes every quant from it.
Each quant is uploaded as soon as it is ready and deleted, so the disk never holds more than the F16 file and one
quant. The GGUF card lists every file with its size (`ollama run hf.co/you/Minecraftism-4B-GGUF:Q8_0` picks one;
`llm import --hf ... --quant Q8_0` does it for the bot). llama.cpp comes from its newest GitHub release (Linux
build + source at the same tag; `llama-quantize` is compiled when the prebuilt one cannot run), or from
`$LLAMA_CPP_DIR`. If that route fails, Unsloth's own GGUF exporter makes the same quants.

`GGUF_QUANTS = "all"` (the default) makes **every llama.cpp quant** that works on a normal model, 29 files:

| Group | Quants | 4B files | Use them when |
| --- | --- | --- | --- |
| full precision | `BF16`, `F16` | ~8 GB | making your own quants; `BF16` is converted straight from the safetensors |
| near-lossless | `Q8_0`, `Q6_K` | 4.3 / 3.3 GB | plenty of RAM, best answers |
| recommended | `Q5_K_M`, `Q5_K_S`, **`Q4_K_M`** (default), `Q4_K_S`, `IQ4_XS`, `IQ4_NL` | 2.3-2.9 GB | most PCs |
| low RAM | `Q3_K_L`, `Q3_K_M`, `Q3_K_S`, `IQ3_M`, `IQ3_S`, `IQ3_XS`, `IQ3_XXS` | 1.7-2.2 GB | old laptops, phones |
| tiny | `Q2_K`, `Q2_K_S`, `IQ2_M`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS`, `IQ1_M`, `IQ1_S` | 1.1-1.7 GB | experiments: clearly weaker |
| legacy | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1` | 2.4-3.1 GB | old apps; `Q4_0` is fast on ARM CPUs |

(VL editions add the `mmproj` vision projector, ~0.8 GB.)

**Importance matrix (imatrix).** Before the quants below Q8_0, `llama-imatrix` runs the fresh Q8_0 over
`--imatrix-chunks` (64) x 512 tokens of the edition's own training data (~10 min on the CPU) and every quant below
Q8_0 is made with it: better quality at the same size, and the `IQ1`/`IQ2`/`IQ3_XXS`/`Q2_K_S` quants can't be made
without one (llama.cpp refuses: "the result will be garbage"). The matrix is uploaded too
(`*.imatrix.gguf`), so anyone can make more quants. Not made on purpose: `F32` (twice the size, no better),
`TQ1_0`/`TQ2_0` and `Q1_0`/`Q2_0` (only for models *trained* ternary or at 1-2 bits) and `MXFP4_MOE`
(mixture-of-experts only).

**Time.** All 29 take roughly an hour of CPU per edition plus ~80 GB of uploads. They are made most useful first
(`Q8_0`, `Q4_K_M`, `Q6_K`, `Q5_K_M`, `IQ4_XS`, `Q3_K_M`, ...; the legacy ones, `BF16` and `F16` last) and whatever does
not fit before `--deadline` is skipped and listed on the model card, so the session never runs out mid-upload.
With more editions than T4s this happens on the CPU while the next edition trains; only the last edition can run
short. `GGUF_QUANTS = "standard"` makes just `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q3_K_M`.

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

## 2. Train on Kaggle

1. kaggle.com → **Datasets → New Dataset** → upload the files in `llm_data/` (with `images/`, without `wiki_cache`).
   Or use the CLI: edit the `id` in `dataset-metadata.json`, then `kaggle datasets create -p llm_data -r zip`.
2. **Code → New Notebook → File → Import Notebook** → paste the link of the latest notebook
   `https://github.com/codero-sus/PyMC_Bot/raw/3b02790b04b2f557d3fc5e819332624feb6423ee/pynb/PyMC_Minecraft_LLM_Kaggle.ipynb`
   ([view it on GitHub](https://github.com/codero-sus/PyMC_Bot/blob/3b02790b04b2f557d3fc5e819332624feb6423ee/pynb/PyMC_Minecraft_LLM_Kaggle.ipynb)), or upload `pynb/PyMC_Minecraft_LLM_Kaggle.ipynb`.
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
| `editions_to_be_trained` | `{txt}` | **only these editions are trained**: any of `{txt, txt_think, vl, vl_think}` (= `minecraftism`, `-thinking`, `-vl`, `-vl-thinking`), e.g. `{txt, vl}`; `"all"` = all four. Quotes are optional. They run longest first |
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
| `SAFETENSORS` | `True` | also export the merged 16-bit safetensors edition and upload it to `<repo without -GGUF>` (`--safetensors none` = GGUF only, made by Unsloth) |
| `GGUF_QUANTS` | `"all"` | GGUFs made from the safetensors and uploaded one by one (only when uploading): `"all"` = every llama.cpp quant (29, F16 to IQ1_S, with an imatrix; what does not fit in the time is skipped), `"standard"` = `q8_0,q6_k,q5_k_m,q4_k_m,q3_k_m`, or a list; `""` = only `q4_k_m` |
| `QUICK_TEST` | `False` | short test run of the whole pipeline |
| `RUN_TESTS` | `True` | test suite after training ([below](#testing)) |
| `TEST_GGUF` | `False` | also installs Ollama in the notebook and chats with the exported GGUF |
| `CLEAN_AFTER_UPLOAD` | `True` | deletes the big local files once an edition is on Hugging Face |
| `GPU_MODE` | `"parallel"` | `"parallel"`: one edition per T4 at the same time; with one edition (or the last one left) both T4s merge on it; `"merge"`: both T4s train every edition together (DDP via HF Accelerate, fp16), editions one after another; `"single"`: one T4, one edition after another; `"auto"` = same as parallel |
| `DISTILL` | `True` | Qwen3's strong-to-weak distillation from a bigger teacher ([below](#distillation-the-qwen3-recipe)) |
| `TEACHER` | `"auto"` | the best teacher for each phase: `Qwen/Qwen3-14B-AWQ` writes the answers (on vLLM) and `unsloth/Qwen3-8B` grades; with `UNCENSORED`, `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` does both. A name (any Qwen3 with the same tokenizer, e.g. `"unsloth/Qwen3-14B"`) is used for both phases |
| `DISTILL_ROWS`, `DISTILL_MINUTES` | `1500`, `40` | answers the teacher writes (phase 1) and the most minutes it may spend on them |
| `ONPOLICY_STEPS` | `60` | on-policy steps (phase 2, text editions), 4 student replies per T4 each; `0` = phase 1 only |
| `VLLM_TEACHER` | `True` | the teacher writes its answers with vLLM on both T4s (installed in its own venv in the background); transformers if vLLM can't run |
| `VLLM_PREFLIGHT` | `True` | starts the teacher once on vLLM while the T4s are still idle (during the data check): a broken vLLM shows its cause in ~5 min of idle time instead of costing ~40 min of training time, and the teacher is downloaded by then |
| `SLOW_TEACHER` | `False` | when vLLM can't run, the teacher writes no new phase-1 answers with transformers (~10x slower); cached answers are still used and phase 2 still runs. `True` = write with transformers for up to `DISTILL_MINUTES` |
| `TEACHER_CACHE` | `True` | keeps the teacher's answers in a private Hugging Face dataset (`<you>/Minecraftism-teacher-answers`) so later sessions reuse them |
| `FAST` | `True` | batches of ~12k tokens per GPU pass (same effective batch), 2 data-loader workers per GPU, 60 eval rows instead of 150, Hugging Face transfers at full speed |
| `MAX_HOURS` | `11.75` | everything (training, tests, export, upload) is done 15 min before THE 12 HOURs, Kaggle's cut-off: training stops early enough that the export and upload still fit |
| `AUTO_EXIT` | `True` | last cell: ends an interactive session once every edition is on Hugging Face |
| `DISCORD` | `True` | Discord webhook messages: server online (with `NOTEBOOK_URL`), status, a THE 12 HOURs countdown before Kaggle's 12 h limit kills the server: 5 h, 3 h, 2 h, 1 h, 30 / 15 / 10 min, then every minute of the last 5 and a final report 30 s before, each with everything: every edition's state, its Hugging Face links, what is still only on the server (lost with it), GPUs, RAM, disk, low disk/RAM, server closing (with the full report once more), edition starts, uploads with their URLs, results, failures, errors |
| `DISCORD_SECRET` | `"discord_webhook_inform_me1"` | name of the Kaggle secret that holds the webhook URL (never put the URL in the notebook) |
| `DISCORD_EVERY_MIN` | `30` | a server status every N minutes (GPUs, RAM, disk, step / loss / ETA, time left); 0 = events and warnings only |
| `DISCORD_MENTION` | `""` | who is pinged: setup ready, now training / distilling / quantizing / uploading, every model upload to Hugging Face, each model ready, THE 12 HOURs at 5 h, 3 h, 2 h, 1 h, 30 / 15 / 10 / 5 / 1 min and the 30 s final report, the close, low disk, failures and errors (server status and the 4 / 3 / 2 min updates come silently); e.g. `"<@your user ID>"` or `"@here"` |
| `PING_ROLE_SECRET` | `"ping_role"` | name of the Kaggle secret that holds a Discord role ID (or `<@&ID>`); that role is pinged on the important messages: startup (setup ready), each stage (training, distilling, quantizing, uploading), every Hugging Face upload, each model ready, errors and failures, low disk, THE 12 HOURs at 5 h, 3 h, 2 h and 1 h, the close and the notebook's end. Server status and the last-hour countdown stay silent. No secret = no role ping; never printed. Turn on *Allow anyone to @mention this role* in Discord |
| `NOTEBOOK_URL` | `""` | this notebook's Kaggle address, shown in the messages |
| `EPOCHS`, `EXTRA` | `1`, `""` | more training / any other trainer flag |
| `STOP_WHEN_LEARNED` | `True` | ends an edition once its training loss has stopped falling for 10% of the run (only after 3/4 of the first epoch): templated data is learned long before the last epoch, so hours of weekly GPU time go to the next edition instead |

Before any training starts, the notebook checks the GPU, the internet, the free disk, the dataset and
the `HF_TOKEN` secret, and prints the trainer flags.

### Distillation: the Qwen3 recipe

Alibaba made the small Qwen3 models (0.6B–14B) strong without training each one from scratch: they *distilled* them
from the big ones (the Qwen3 technical report calls it **strong-to-weak distillation**). It has two phases, and
`DISTILL = True` runs both. Each phase gets the best teacher that runs on Kaggle's 2× T4. Both teachers are Qwen3 with
the same tokenizer (151,936 tokens) and the same way of thinking as the 4B, and both are Apache-2.0, so a
`COMMERCIAL_ONLY` run stays commercial.

#### Which teacher, and why

| phase | needs | teacher (`TEACHER = "auto"`) |
| --- | --- | --- |
| 1. writes the answers | only its text; runs alone on both T4s in vLLM | **Qwen3-14B** (`Qwen/Qwen3-14B-AWQ`, Qwen's own 4-bit build, ~10 GB). It falls back to Qwen3-8B without vLLM, or if the 14B doesn't start |
| 2. grades the student's replies | the same tokenizer; fits on one T4 *next to* the student | **Qwen3-8B** (4-bit, ~5.7 GB). A 4-bit 14B (~10.5 GB) plus the student plus the 152k-token probabilities don't fit in 15 GB |
| both, with `UNCENSORED` | a teacher that doesn't refuse | **huihui-ai's abliterated Qwen3-8B**. No 4-bit *uncensored* 14B runs on a T4 (the community AWQ builds need Ampere kernels) |

Why not something bigger or newer:
* **Qwen3-30B-A3B / 32B:** their 4-bit builds need Marlin kernels, which a T4 (Turing) doesn't have, and a 32B
  would be several times slower.
* **Qwen3.5 / 3.6:** they use a different tokenizer (248k tokens), so they can't grade a Qwen3 student, and their
  linear-attention layers are untested on a T4.
* **Qwen3.5 or Llama students** (`--preset qwen3.5-4b`, `llama-3.2-3b`): the trainer checks that the tokenizers
  match and skips phase 2 when they don't. Phase 1 still uses the teacher's answers.

1. **Off-policy (response distillation), `DISTILL_ROWS = 1500`.** Before training, the teacher writes the answers:
   * **everyday chat:** the human-written answers are replaced by the teacher's. Thinking editions also get the
     teacher's own `<think>` trace (Qwen3-14B and 8B are hybrid: `enable_thinking` on for them, off for the others);
   * **Minecraft Q&A:** each one is rewritten in a natural voice, *around the exact answer*. The answer from
     minecraft-data goes into the teacher's prompt as a reference, and a rewrite is used only if every number in
     that reference is in it;
   * **not touched:** game turns keep their rule-checked JSON, and the Nemotron thinking rows keep their traces
     (they already came from a far bigger model);
   * **dropped:** cut-off answers, empty answers and (when `UNCENSORED`) refusals. Those rows keep their original
     answer.

   With `GPU_MODE = "merge"` both T4s write answers, half each. They are saved in the stream cache, so the next
   edition of the session reuses them (VL editions use the text answers too). The teacher gets at most
   `DISTILL_MINUTES`, and never more than 40% of the time left before `MAX_HOURS`.
2. **On-policy (logit distillation), `ONPOLICY_STEPS = 60`, text editions.** After fine-tuning, the 4B **answers
   prompts itself** (sampled, as it would really answer) and the teacher scores **every token of those answers**. The
   student then minimises the per-token **reverse KL(student ‖ teacher)** on its own replies, so it learns the
   teacher's full probabilities instead of copying one sampled answer, on exactly the kind of text it produces.
   Qwen reports that this beats reinforcement learning at a tenth of the GPU hours. Some details:
   * each T4 takes different prompts and the LoRA gradients are averaged;
   * the KL is computed in fp32 in chunks;
   * for a non-thinking student the teacher reads the reply after Qwen3's empty `<think></think>` marker, so it never
     pulls the student towards thinking;
   * on running out of memory the batch is halved, and any failure keeps the fine-tuned model as it is.

The model card gets a **Distillation** section (teacher, rows rewritten, KL at the start and end) and lists the teacher
in the data table. `DISTILL = False` trains as before.

**Making distillation fast:**

* **The teacher writes with vLLM** (`VLLM_TEACHER = True`). vLLM 0.18.1, the version validated on Kaggle's 2× T4, is
  installed with `uv` into its own venv (`/kaggle/tmp/vllm`) in the background while the data is checked, so
  Unsloth's torch stays untouched. The trainer then runs the teacher in a separate process:
  * fp16, split over **both T4s** (tensor parallelism), with paged attention, continuous batching and prefix caching;
  * typically several times faster than transformers' `generate`;
  * with one visible T4 (`GPU_MODE = "parallel"`), the model is quantized to 4-bit while loading;
  * eager mode (no CUDA graphs), with as much GPU memory as is free, and **without the DDP launcher's variables**:
    inheriting `TORCHELASTIC_USE_AGENT_STORE` from torchrun / accelerate made vLLM's workers hang ~20 min per try;
  * an engine that isn't ready after 15 min is stopped with all its processes. A quick failure gets one retry with
    less memory; then transformers takes over (half the prompts per T4), unless under 5 min of teacher time is left;
  * a failure prints its cause and keeps the full log as `/kaggle/working/vllm-teacher.log`; the next edition of
    the same session goes straight to transformers.
* **Answers are written once.** They're kept for the session (two editions side by side take turns, and the second
  reuses the first one's answers). With `TEACHER_CACHE` and an `HF_TOKEN`, they're also saved to a private
  Hugging Face dataset, so the next session downloads them instead of writing them again.
* **Batched on-policy replies** (`--onpolicy-refresh 4` with `FAST`): the student writes the replies for 4 steps in one
  batched call. They are at most 3 small LoRA steps old, which is practically on-policy.
* The eval and the test suite ask **16 prompts per batch** instead of one at a time.

Rough cost (estimates, not measured): with vLLM, a few minutes of both T4s plus ~3 min to start the engine for
1500 answers (transformers: ~25–40 min); then ~10–15 min per text edition for the on-policy phase.

### Where files go

Kaggle gives two disks. The notebook keeps what is needed again on the saved one and puts the throwaway files on
the big scratch one:

| `/kaggle/working`: 20 GB, kept (Output tab, file persistence) | `/kaggle/tmp`: ~60 GB, thrown away |
| --- | --- |
| each edition's folder (model until uploaded, `report.json`, `export_job.json`) | downloaded base and teacher models (`hf-cache/`) |
| the logs (`<edition>.log`, `vllm-install.log`, `vllm-teacher.log` if the teacher failed) | the vLLM venv (`vllm/`) and its job folders |
| the data kit (`kit/`) and `train_minecraft_llm.py` | the F16 GGUF and the quants before upload (`gguf/`) |
| **`cache/`**: chat, Q&A, pictures and teacher answers, fetched once and used by every edition | checkpoints, GPU hand-over files, locks |
| **`tools/`**: llama.cpp, downloaded once for every edition | temp files (`TMPDIR`), pip/uv caches, Unsloth's compiled cache, the kit zip, the system prompt |

So the 20 GB output disk only fills with real results, and a crash never leaves gigabytes of junk in the Output.
With the notebook's file persistence on, `cache/` and `tools/` carry over to the next session. The trainer finds
the scratch disk by itself on Kaggle (`/kaggle/tmp`; `PYMC_SCRATCH=<folder>` picks another one). Elsewhere it
keeps using the output folder and the system temp folder, as before. `--tools-dir <folder>` sets where llama.cpp goes.

### Speed: both T4s, every minute

A Kaggle GPU session stops after **12 hours** and the weekly GPU quota is about 30 hours, so the notebook never
leaves a T4 idle:

* **One edition per T4** (`GPU_MODE = "parallel"`, the default) is described two bullets below. With a single
  edition, or when only the last edition is left and the other T4 is free, both T4s merge on it instead, so a T4
  never idles.
* **Both T4s merged on every edition** (`GPU_MODE = "merge"`). Every edition is launched with
  **HF Accelerate** (`accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16`; plain `torchrun` if
  Accelerate is missing) and trained by both T4s as described in the DDP bullet below. The editions run one after
  another, and with more than one edition the GGUFs are made on the CPU while both T4s train the next one. Both T4s
  also share the distillation teacher's work.
* **One edition per T4** (`GPU_MODE = "parallel"`, the default). Each edition runs in its own process with
  `CUDA_VISIBLE_DEVICES` set to its own T4, so two editions finish in the time of one. The editions run **longest
  first** (`vl_think`, `vl`, `txt_think`, `txt`), so both T4s finish at about the same time. When a T4 is free, the next
  edition starts on it. The merges and the GGUF steps each take turns (two locks), so two of them never fill the
  20 GB disk or the RAM at the same time; training on the other T4 continues meanwhile.
* **DDP: both T4s train one edition.** Each T4 holds the 4-bit model and
  trains half of every batch; the gradients are averaged, so the result matches one T4 with the same effective batch
  (gradient accumulation is halved). Expect roughly 1.6–1.9× the speed of one T4. Only the first process reads the
  data, streams the extra rows and exports. If the DDP run fails before training ends, the notebook trains the same
  edition again on one T4, so a session isn't lost.
* **`FAST = True`**:
  * **Batches sized by tokens** (`--token-budget 12288`, text editions). Unsloth trains *padding-free*, so a batch
    costs only its real tokens; the trainer measures the rows (95% of them are shorter than N tokens) and puts as
    many rows into one forward pass as fit ~12k tokens, e.g. all 8 rows of an optimizer step in **one** pass instead
    of 4 passes of 2. The effective batch (rows per optimizer step) stays the same, so the model learns the same.
    If a T4 runs out of memory before the first step, the batch is halved and training starts again. If it happens
    later, the notebook trains the edition again with the preset's batch (`--token-budget 0`).
  * **Streamed once, used by every edition** (`--stream-cache`). The everyday chat, Minecraft Q&A, minecraft-data
    facts and pictures are downloaded by the first edition that needs them. The next ones (e.g. `vl` after `txt`, or
    `vl_think` after `vl`) reuse them instead of streaming the same rows again. If two editions start together, one
    waits for the other's download.
  * 2 data-loader workers per GPU, 60 eval rows instead of 150, and `HF_XET_HIGH_PERFORMANCE=1` (Hugging Face's
    downloads and uploads use all CPU cores and the full bandwidth).
* **Faster export ("distillation")**, always on:
  * the safetensors upload runs **in the background while llama.cpp makes the F16 GGUF and the quants**, and each
    quant uploads while the next one is being made (same upload order as before: safetensors, then each quant);
  * with `CLEAN_AFTER_UPLOAD`, the ~8 GB safetensors folder is deleted as soon as it is on Hugging Face and in the
    F16 GGUF, not at the very end;
  * **with more editions than T4s, the GGUF step leaves the GPU** (`--defer-gguf`). Once the safetensors are saved,
    the training process ends and a CPU-only `--export-only` process makes the quants and uploads them, while that
    T4 already trains the next edition. That's ~20–40 minutes of T4 time per edition that used to wait for llama.cpp
    and the uploads. This happens only when llama.cpp is ready; otherwise the edition exports in place, with
    Unsloth's exporter as the fallback.
* The dry runs of all editions run side by side.
* **Every core and the idle minutes work too**: tokenizers and data loaders use all CPU cores, and while vLLM installs and
  the data is checked, the first editions' base models and the grading teacher download in the background and the
  vLLM teacher is test-started on the idle T4s (`VLLM_PREFLIGHT`).
* **`MAX_HOURS = 11.75`**: the trainer gets a deadline 15 min before THE 12 HOURs. When the time left drops below the export reserve (60 min, or 90
  with two editions in parallel), training stops at the current step, the model is saved, exported and uploaded as
  usual, and `pymc_model.json` records `stopped_early` (step N of M). An edition that wouldn't even have time to start
  is skipped (exit code 3) and the results cell says so. Train it in the next session.

Two editions such as `{txt, txt_think}` take one session
(~2–3 h), and all four fit into one 12 h session with uploads in between (`CLEAN_AFTER_UPLOAD` keeps the disk free).

**Ending the session.** Kaggle has no command that stops a session from inside it. For long runs use
**Save Version → Save & Run All (Commit)**: it runs in the background, saves `/kaggle/working` as the notebook's
Output and stops when the last cell is done, so no quota is wasted. In an interactive session, the last cell
(`AUTO_EXIT = True`) ends the kernel and every process of the session once **every** edition is on Hugging Face
(GGUF and safetensors), so nothing keeps the GPU busy. If the editor still shows the session as running, press
**Run → Stop session** (otherwise Kaggle stops it after its idle timeout). If anything is only local, it doesn't shut down and says why, because an interactive session's files
disappear when the session ends. The notebook dry-runs every edition first and skips the ones that can't train
(a VL edition with no pictures and `STREAM_PICTURES = 0`, for example) instead of failing hours later.

The script trains a LoRA adapter (4-bit QLoRA, r=32, loss on the replies only; VL editions train the
language layers and keep the vision encoder frozen unless you pass `--vision-layers 1`). It evaluates on
held-out rows, saves the compact safetensors and makes the GGUFs from them (`q4_k_m` stays in the folder; when uploading
every other quant too, each uploaded and deleted as soon as it is ready) and, for VL editions, the **vision projector** (`mmproj`). Each edition
gets its own folder, e.g. `/kaggle/working/minecraftism-vl`:

```
minecraftism-vl-4b.Q4_K_M.gguf          the model
mmproj-minecraftism-vl-4b.F16.gguf      the vision projector (VL editions only)
Modelfile                               for ollama create (both FROM lines, sampling, stop tokens)
params                                  sampling for "ollama pull hf.co/..." (written when uploading)
README.md                               the model card shown on Hugging Face
pymc_model.json                         edition, loss, eval + test scores, system prompt, what PyMC Bot reads on import
tests.json                              every test question and answer (with RUN_TESTS)
tokenizer.json, config.json, ...        tokenizer, configs and chat template (also uploaded to the GGUF repo)
lora/                                   the adapter (optional, for training further)
safetensors/                            the safetensors edition: model-*.safetensors, config, tokenizer /
                                        processor, its own README.md (with SAFETENSORS, the default)
```

With `HF_REPO = "auto"` each edition is uploaded twice: first `safetensors/` to `<your HF name>/<repo without -GGUF>`,
e.g. `you/Minecraftism-4B`, then the GGUF quants (one by one) and the folder to `<your HF name>/<repo from the table>`,
e.g. `you/Minecraftism-4B-GGUF`.
The local copy is cleaned up afterwards (`CLEAN_AFTER_UPLOAD`) so four editions fit into Kaggle's 20 GB; the
safetensors folder is only deleted once its own upload worked. If an upload fails, the files stay in
`/kaggle/working`. Without an upload an edition takes ~11 GB (GGUF + safetensors): download it from the Output tab
before training the next one, or set `SAFETENSORS = False`.

Useful flags (`EXTRA`): `--epochs 2`, `--max-steps N`, `--gguf q5_k_m|q8_0` (the main GGUF),
`--gguf-quants q8_0,q4_k_m` (which others), `--gguf-scratch <folder>`, `--no-compact`, `--hub-private`,
`--hub-lora` (also upload the adapter), `--image-side 448` (smaller pictures, faster VL training),
`--model <any Unsloth id>`, `--dry-run` (checks the data without a GPU).
Speed flags (the notebook sets them): `--deadline <unix time>` + `--reserve-minutes 60`, `--export-lock <file>`,
`--clean-after-upload`, `--token-budget 12288`, `--loader-workers 2`, `--stream-cache <folder>`, `--tools-dir <folder>`, `--defer-gguf` (then
`--export-only <out>` makes the GGUFs on the CPU), `--group-by-length` (without `--token-budget`). Two T4s on one edition:
`accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 train_minecraft_llm.py --data ... --preset minecraftism`
(or `torchrun --standalone --nproc_per_node 2 train_minecraft_llm.py ...`).
Distillation flags: `--distill-rows 1500`, `--distill-minutes 40`, `--onpolicy-steps 60`, `--teacher auto`,
`--onpolicy-batch 4`, `--onpolicy-tokens 256`, `--onpolicy-lr 5e-5`, `--onpolicy-refresh 4`,
`--vllm-python /kaggle/tmp/vllm/bin/python` (`--teacher-engine auto|vllm|hf`), `--teacher-cache-repo auto`
(all off by default; the notebook turns them on).

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
| Everyday chat (`GENERAL_CHAT`) | [`OpenAssistant/oasst_top1_2023-08-25`](https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25) | Apache-2.0 | ~13k chats written by people, multilingual. Thinking editions get it too: it fills every everyday row the traces below don't, and with `DISTILL` the teacher writes the thinking for up to `DISTILL_ROWS` of them |
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

## A trusted dataset of your own (Kaggle, CPU only)

`pynb/Minecraftism_Dataset_Kaggle.ipynb` builds **`<you>/Minecraftism-dataset`** on Hugging Face without a GPU:
DuckDuckGo search (`ddgs`) finds pages, and an LLM runs in Kaggle's 30 GB of RAM on llama.cpp's CPU server.
`MODEL = "auto"` (the default) takes the biggest that fits the free RAM: on Kaggle **Qwen3-30B-A3B-Instruct-2507**
(Q4_K_M, ~23 GB RAM). It is a mixture of experts: all 30B weights sit in RAM but only ~3B work on each token, so on
a CPU it runs about as fast as a 4B model and judges far better. `"qwen3-4b"` (~4 GB) and `"qwen3-1.7b"` (~2.5 GB,
2-3× faster, least careful) are the smaller choices and the fallback on smaller machines.
For speed on the CPU it works on `PARALLEL = 4` topics at once (llama-server's parallel slots batch them, and
while one topic waits for a web search the others keep the cores busy), skips the second web search when the game
data already confirms an answer, and keeps the judge's reason to one sentence. **Quality first** (`MIN_TOKENS_PER_S = 0`, the default): on Kaggle
it always runs the biggest model, Qwen3-30B-A3B (probably ~8-15 tok/s on the 4 CPU threads). It still uses the whole
machine: llama-server is its own process on all 4 threads, batching the 4 topics; web search, downloads, HTML parsing
and picking passages run in `PARALLEL` separate worker **processes** (`PROCESSES = True`), so they never wait on the
main Python process. A short speed test on the real job measures llama.cpp's n-gram speculation (`SPECULATE =
"auto"`: it guesses the quotes the LLM copies and checks many tokens in one step; the text stays the same) and keeps
the faster way. Set `MIN_TOKENS_PER_S = 20` for speed over quality: it then keeps the biggest model that writes at
least 20 tok/s (probably the 4B or 1.7B). The log shows the speed test, rows/h and the LLM's tokens/s. **GPU T4 x2** (Settings → Accelerator; uses your weekly GPU
hours): `DEVICE = "auto"` runs llama.cpp's CUDA build on both GPUs - the prebuilt one, else compiled here for the T4
(~20-30 min, once per session), else back to the CPU. On the GPU `MODEL = "auto"` runs **Qwen3.5-4B** (on
benchmarks about as strong as the 30B-A3B, ~4x smaller): it fits one T4, so each T4 gets its own copy (two
llama-servers, 8 requests each, every request to the least busy one) - roughly 60-90 tok/s per request on a T4.
`"qwen3.5-9b"` is stronger at about half the speed. On the CPU the 30B-A3B stays (there it is as fast as a 4B).

**As many rows as possible.** After the ~1,700 game-data topics it reads every other article of the Fandom wiki
(`WIKI_PAGES = True`; without version/snapshot pages, other games and sub-pages except tutorials), opens each page
directly (no search) and reads it **whole**, in parts of ~2,800 characters (changelog, gallery and link sections
skipped), with 5 pairs per part. To go faster, run several copies of the notebook at the same time with
`SESSIONS = N` and `SESSION = 1 ... N`: each takes every N-th topic and writes its own `data/train-NN.jsonl`,
`meta/progress-NN.json` (it uploads only those), and reads the others' rows and topics so nothing is done twice.
`load_dataset` reads all the files. The LLM writes question/answer pairs
only from the page text, and each pair has to pass every check:

| check | rejects |
|---|---|
| licence | any page that is not Minecraft Fandom wiki (CC BY-SA 3.0) or Wikipedia (CC BY-SA 4.0); minecraft.wiki (CC BY-NC-SA) is never read, official Mojang pages only confirm |
| evidence | a quoted sentence that is not really on the page |
| numbers | a number in the answer that is not in the quotes |
| second look | a separate LLM pass that does not answer SUPPORTED from question + answer + quotes alone |
| game data | a crafting answer with a missing ingredient or a wrong count (minecraft-data recipes; catches outdated pages such as the old lead recipe) |
| independent page | an answer another page contradicts (the question is searched again) |

Rows confirmed by the game data or an independent page are `trust: high`, the rest `medium` (`KEEP = "high"`
keeps only the confirmed ones). Rejected pairs and the reason go to `meta/rejected-NN.jsonl`. It uploads every 30
minutes and `RESUME = True` continues the topic list in the next session; **Stop** keeps and uploads what it has.
The dataset is **private** by default (`PRIVATE = True`; an existing public repo is switched to private before every
upload), a read-only token is caught before the model download, and the next topic's pages are fetched while the
LLM checks the current one. `load_dataset("<you>/Minecraftism-dataset", "high")` loads only the confirmed rows. Kaggle setup: Accelerator **None**,
Internet on, secret `HF_TOKEN` (optional `discord_webhook_inform_me1` and `ping_role`).
Import link: `https://github.com/codero-sus/PyMC_Bot/raw/f0a2601e517b9901a23073543d64dd6d1fa82e2c/pynb/Minecraftism_Dataset_Kaggle.ipynb`.

Use it for training: `--minecraft-qa-dataset <you>/Minecraftism-dataset` (the trainer reads `question` /
`answer`). The collector is `pymc_bot/llm_webdata.py`, and the notebook carries it verbatim (`python
scripts/build_webdata_notebook.py` rebuilds it).

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

<sub>Made with ❤️ by [codero-sus](https://github.com/codero-sus) · [codero.is-a.dev](https://codero.is-a.dev) · [the-codero.web.app](https://the-codero.web.app/)</sub>

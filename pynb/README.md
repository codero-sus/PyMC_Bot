# pynb – Jupyter notebooks

| Notebook | Runs on | What it does |
| --- | --- | --- |
| [`PyMC_Minecraft_LLM_Kaggle.ipynb`](PyMC_Minecraft_LLM_Kaggle.ipynb) | Kaggle, GPU T4 ×2 | trains **Minecraftism**, PyMC Bot's own LLM: a Minecraft expert and an everyday assistant, uncensored by default, trained only on data that allows commercial use (`COMMERCIAL_ONLY = True`). Four editions (text / vision × fast / thinking), optional test suite, uploads the compact 16-bit **safetensors** first, then **every GGUF quant** made from them (29 files: F16 and BF16 down to IQ1_S, with an importance matrix from the training data, each uploaded as soon as it is ready; the model card has a "which file?" guide and the full table), both with `tokenizer.json` and the configs |
| [`Minecraftism_Dataset_Kaggle.ipynb`](Minecraftism_Dataset_Kaggle.ipynb) | Kaggle, **CPU only** (no GPU) | collects a **trusted Minecraft Q&A dataset**: web search + a small LLM in RAM (Qwen3-4B or 1.7B on llama.cpp) write answers only from pages that allow commercial use, every answer is checked (quotes on the page, numbers in the quotes, a second LLM look, the game's recipes, an independent page), then it is uploaded to your Hugging Face dataset `<you>/Minecraftism-dataset` |

**Quick start**

1. kaggle.com → **Code → New Notebook → File → Import Notebook** → paste the link of the latest notebook:
   `https://github.com/codero-sus/PyMC_Bot/raw/3b02790b04b2f557d3fc5e819332624feb6423ee/pynb/PyMC_Minecraft_LLM_Kaggle.ipynb`
   ([view it on GitHub](https://github.com/codero-sus/PyMC_Bot/blob/3b02790b04b2f557d3fc5e819332624feb6423ee/pynb/PyMC_Minecraft_LLM_Kaggle.ipynb)), or upload this `.ipynb`.
2. **Settings** → Accelerator **GPU T4 ×2**, Internet **on**.
3. Optional:
   * **Add Input** → your own `llm_data` from `python -m pymc_bot llm build`. Without it, the ready-made kit
     is downloaded from the newest GitHub release.
   * **Add-ons → Secrets** → `HF_TOKEN` (a Hugging Face *write* token) to publish the model.
4. Edit the settings cell. Pick the editions with `editions_to_be_trained = {txt, txt_think, vl, vl_think}`
   (only those are trained, e.g. `{txt}` or `{txt, vl}`). `GPU_MODE = "parallel"` (default) trains two editions at once, one per T4, and merges both T4s on a single
   edition (DDP launched with HF Accelerate, fp16); `"merge"` = both T4s on every edition, one after another. `DISTILL = True` runs Qwen3's strong-to-weak distillation: Qwen3-14B (4-bit AWQ) writes the answers to
   `DISTILL_ROWS` chat and Minecraft prompts, then Qwen3-8B teaches the fine-tuned 4B token by token on its own replies
   (`ONPOLICY_STEPS`, reverse KL; with `UNCENSORED` the abliterated Qwen3-8B does both). The teacher writes with vLLM on both T4s (`VLLM_TEACHER`, installed in its own venv in
   the background) and its answers are kept in a private HF dataset for the next session (`TEACHER_CACHE`). `MAX_HOURS` makes training stop early
   enough for the export and upload to finish before Kaggle's 12 h limit. `FAST = True` sizes batches by tokens,
   streams the shared data once for all editions, uploads while llama.cpp makes the GGUFs and, with more editions
   than T4s, makes the GGUFs on the CPU while the T4 already trains the next edition. The first time, use `QUICK_TEST = True`, which takes about 15-25 minutes. Then **Run All**.
5. On your PC: `python -m pymc_bot llm import --hf <your-hf-name>/Minecraftism-4B-Uncensored-GGUF`.
   The safetensors edition, `<your-hf-name>/Minecraftism-4B-Uncensored`, is for transformers, vLLM or further
   fine-tuning. Its model card has the code. Other GGUF sizes: `ollama run hf.co/<your-hf-name>/Minecraftism-4B-Uncensored-GGUF:Q8_0`.

For long runs use **Save Version → Save & Run All**: the session ends by itself when the notebook is done, and
`/kaggle/working` is kept as the Output. In an interactive session, `AUTO_EXIT = True` (the last cell) shuts the kernel
down once everything is on Hugging Face.

**Discord updates:** add your channel's webhook URL as the Kaggle secret `discord_webhook_inform_me1`
(**Add-ons → Secrets**; the name is `DISCORD_SECRET` in the settings). The notebook then posts:
* **Kaggle server online**, with the notebook's URL (set `NOTEBOOK_URL`), its GPUs, RAM and disk
* **Server status** every `DISCORD_EVERY_MIN` minutes (30 by default): GPU load and memory, RAM, free disk, each edition's
  step / loss / ETA, time left. If these messages stop coming, the server is gone
* A **THE 12 HOURs countdown** before Kaggle's 12 h limit kills the server: 5 h, 3 h, 2 h, 1 h, 30 / 15 / 10 min, then **every minute of the last 5** and a final report 30 s before, each with everything: every edition's state, its Hugging Face links, what is still only on the server (lost with it), GPUs, RAM, disk; and when the disk or RAM is almost full
* **Server closing** when the kernel shuts down before the notebook finished (stopped, restarted, timed out), or
  **Server closed** after a normal finish - both with the full report once more (so does **Session finished**)
* each edition's start, "learned" (loss stopped falling), every upload with its Hugging Face URL, the results, failures
  (with the end of the log) and notebook errors

**Ping a role:** add the Kaggle secret `ping_role` (the name is `PING_ROLE_SECRET` in the settings) with the
role's ID (Discord: Developer Mode on, Server Settings → Roles → … → Copy Role ID, and turn on *Allow anyone to @mention this
role*). That role is pinged on the important messages: startup (setup ready), each stage (training, distilling, quantizing, uploading), every Hugging Face upload, each model ready, errors and failures, low disk, THE 12 HOURs at 5 h, 3 h, 2 h and 1 h, the close and the notebook's end. Server status and the last-hour
countdown stay silent. No secret = no role ping.

`DISCORD_MENTION = "<@your user ID>"` pings you on: setup ready, now training / distilling / quantizing / uploading, every model upload to Hugging Face, each model ready, THE 12 HOURs at 5 h, 3 h, 2 h, 1 h, 30 / 15 / 10 / 5 / 1 min and the 30 s final report, the close, low disk, failures and errors (server status and the 4 / 3 / 2 min updates come silently). The webhook URL is never printed.

**Where files go:** things needed again stay in `/kaggle/working`: models until uploaded, reports, logs, the data kit,
`cache/` (streamed data and teacher answers, shared by every edition) and `tools/` (llama.cpp). Throwaway files go to
`/kaggle/tmp` (~60 GB, not saved): model downloads, vLLM, GGUF work files, checkpoints, locks and temp files.

## Trusted dataset notebook (CPU only)

1. kaggle.com → **Code → New Notebook → File → Import Notebook** → paste:
   `https://github.com/codero-sus/PyMC_Bot/raw/96dbd6dd5abed4af3360f55a358409189bd19090/pynb/Minecraftism_Dataset_Kaggle.ipynb`
   ([view it on GitHub](https://github.com/codero-sus/PyMC_Bot/blob/96dbd6dd5abed4af3360f55a358409189bd19090/pynb/Minecraftism_Dataset_Kaggle.ipynb)).
2. **Settings** → Accelerator **None** (it needs no GPU and uses none of your GPU quota), Internet **on**.
3. **Add-ons → Secrets** → `HF_TOKEN` (a Hugging Face *write* token; a read token is caught before anything starts). Optional: `discord_webhook_inform_me1` and
   `ping_role` (pinged on start, uploads, errors and the end; the hourly progress stays silent).
4. **Save Version → Save & Run All**. It works through 84 game mechanics, then mobs, items, blocks, enchantments,
   effects and biomes until `MAX_HOURS`, uploads every 30 minutes, and the next run goes on where it stopped.
   **Stop** keeps and uploads what it has. The dataset is **private** (`PRIVATE = True`): only you can see it, and a
   repo that already exists as public is switched to private before every upload.

Each row has `question` / `answer` (plus `messages`), `trust` (`high` = confirmed by the game data or an
independent page, `medium` = one source, every check passed; `load_dataset(..., "high")` loads only those), its sources with their licences and the quotes that
back it. Rejected pairs and why are in `meta/rejected.jsonl`. Only Minecraft Fandom wiki (CC BY-SA 3.0) and
Wikipedia (CC BY-SA 4.0) text is used, so the dataset is CC BY-SA 4.0 and allows commercial use. minecraft.wiki is
non-commercial and never read. Train on it with `train_minecraft_llm.py --minecraft-qa-dataset
<you>/Minecraftism-dataset`. The code is `pymc_bot/llm_webdata.py`. After editing it, run `python scripts/build_webdata_notebook.py`.

Full guide: [docs/MINECRAFT_LLM.md](../docs/MINECRAFT_LLM.md). Releases do not attach the notebook: their notes link it pinned to the release's commit.

Made with ❤️ by [codero-sus](https://github.com/codero-sus) · [codero.is-a.dev](https://codero.is-a.dev) · [the-codero.web.app](https://the-codero.web.app/) – personal, non-commercial use (PolyForm Noncommercial 1.0.0).

# pynb – Jupyter notebooks

| Notebook | Runs on | What it does |
| --- | --- | --- |
| [`PyMC_Minecraft_LLM_Kaggle.ipynb`](PyMC_Minecraft_LLM_Kaggle.ipynb) | Kaggle, GPU T4 ×2 | trains **Minecraftism**, PyMC Bot's own LLM: a Minecraft expert and an everyday assistant, uncensored by default, trained only on data that allows commercial use (`COMMERCIAL_ONLY = True`). Four editions (text / vision × fast / thinking), optional test suite, uploads the compact 16-bit **safetensors** first, then **GGUF** quants made from them (Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, each uploaded as soon as it is ready), both with `tokenizer.json` and the configs |

**Quick start**

1. kaggle.com → **Code → New Notebook → File → Import Notebook** → this `.ipynb`.
2. **Settings** → Accelerator **GPU T4 ×2**, Internet **on**.
3. Optional:
   * **Add Input** → your own `llm_data` from `python -m pymc_bot llm build`. Without it, the ready-made kit
     is downloaded from the newest GitHub release.
   * **Add-ons → Secrets** → `HF_TOKEN` (a Hugging Face *write* token) to publish the model.
4. Edit the settings cell. Pick the editions with `editions_to_be_trained = {txt, txt_think, vl, vl_think}`
   (only those are trained, e.g. `{txt}` or `{txt, vl}`). `GPU_MODE = "auto"` keeps both T4s busy: with two or more editions each T4 trains its own
   edition at the same time; with one edition both T4s train it together (DDP). `MAX_HOURS` makes training stop early
   enough for the export and upload to finish before Kaggle's 12 h limit. `FAST = True` sizes batches by tokens,
   streams the shared data once for all editions, uploads while llama.cpp makes the GGUFs and, with more editions
   than T4s, makes the GGUFs on the CPU while the T4 already trains the next edition. The first time, use `QUICK_TEST = True`, which takes about 15 minutes. Then **Run All**.
5. On your PC: `python -m pymc_bot llm import --hf <your-hf-name>/Minecraftism-4B-Uncensored-GGUF`.
   The safetensors edition, `<your-hf-name>/Minecraftism-4B-Uncensored`, is for transformers, vLLM or further
   fine-tuning. Its model card has the code. Other GGUF sizes: `ollama run hf.co/<your-hf-name>/Minecraftism-4B-Uncensored-GGUF:Q8_0`.

For long runs use **Save Version → Save & Run All**: the session ends by itself when the notebook is done, and
`/kaggle/working` is kept as the Output. In an interactive session, `AUTO_EXIT = True` (the last cell) shuts the kernel
down once everything is on Hugging Face.

Full guide: [docs/MINECRAFT_LLM.md](../docs/MINECRAFT_LLM.md). The same notebook is attached to every GitHub release.

Made with ❤️ by [codero-sus](https://github.com/codero-sus) · [codero.is-a.dev](https://codero.is-a.dev) · [the-codero.web.app](https://the-codero.web.app/) – personal, non-commercial use (PolyForm Noncommercial 1.0.0).

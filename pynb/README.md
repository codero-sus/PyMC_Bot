# pynb – Jupyter notebooks

| Notebook | Runs on | What it does |
| --- | --- | --- |
| [`PyMC_Minecraft_LLM_Kaggle.ipynb`](PyMC_Minecraft_LLM_Kaggle.ipynb) | Kaggle, GPU T4 ×2 | trains **Minecraftism**, PyMC Bot's own LLM: a Minecraft expert and an everyday assistant, uncensored by default. Four editions (text / vision × fast / thinking), optional test suite, GGUF export, upload to Hugging Face |

**Quick start**

1. kaggle.com → **Code → New Notebook → File → Import Notebook** → this `.ipynb`.
2. **Settings** → Accelerator **GPU T4 ×2**, Internet **on**.
3. Optional:
   * **Add Input** → your own `llm_data` from `python -m pymc_bot llm build`. Without it, the ready-made kit
     is downloaded from the newest GitHub release.
   * **Add-ons → Secrets** → `HF_TOKEN` (a Hugging Face *write* token) to publish the model.
4. Edit the settings cell. The first time, use `QUICK_TEST = True`, which takes about 15 minutes. Then **Run All**.
5. On your PC: `python -m pymc_bot llm import --hf <your-hf-name>/Minecraftism-4B-Uncensored-GGUF`.

Full guide: [docs/MINECRAFT_LLM.md](../docs/MINECRAFT_LLM.md). The same notebook is attached to every GitHub release.

Made with ❤️ by codero-sus – personal, non-commercial use (PolyForm Noncommercial 1.0.0).

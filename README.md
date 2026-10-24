# Fine-Tuning DistilGPT-2 on Police Field-Report Narratives

Fine-tuning assignment – Navi Dharshan B (1716785) and *partner*

A pretrained **DistilGPT-2** model is fine-tuned as a causal language model on **600 redacted Boston Police
Field Interrogation and Observation (FIO) narratives** (2024), then five hyperparameter experiments are run, each
changing one setting from the reference (learning rate 5e-4, 5 epochs, batch size 4, block size 128):

| Run | Change | Purpose |
|---|---|---|
| `lr_high` | learning rate 1e-2 | expected divergence |
| `lr_low` | learning rate 5e-6 | expected underfitting |
| `epochs_15` | 15 epochs | expected overfitting |
| `batch_16` | batch size 16 | fewer, smoother updates |
| `block_64` | block size 64 | shorter context per example |

## Files

| File | Purpose |
|---|---|
| `FIO_Narrative_LM_Finetuning.ipynb` | The full assignment notebook (runs top to bottom on a Colab T4 GPU) |
| `app.py` | Streamlit app: compare base vs fine-tuned generations, view the experiments |
| `requirements.txt` | Packages for the app |
| `experiment_results.csv`, `generations.csv`, `loss_log.csv`, `all_loss_curves.png` | Results from the notebook run |

## Data

Boston Police Department, *BPD Field Interrogation and Observation (FIO)*, `2024_annual_fio.csv`, Analyze Boston
(data.boston.gov), Open Data Commons Public Domain Dedication and License (PDDL). Only the narrative field is used;
names, officer names, addresses, neighbourhoods and phone numbers are replaced with placeholders before training.
The raw data is not stored in this repository.

## Deployment (Streamlit Community Cloud)

1. Run the notebook; download `fio_lm_model.zip` and `fio_lm_results.zip`.
2. Upload the files inside `fio_lm_model.zip` to a Hugging Face model repository, e.g.
   `<hf-username>/fio-narrative-distilgpt2`.
3. Upload the files inside `fio_lm_results.zip` to this repository.
4. At share.streamlit.io: **Create app → Deploy a public app from GitHub**, main file `app.py`, Python 3.11, and the
   secret `LM_REPO = "<hf-username>/fio-narrative-distilgpt2"`.

## Responsible use

All generated text is **synthetic**: it imitates the style of field reports, describes no real event and must not
be presented as a police record. The source narratives contain demographic descriptors that the model can reproduce.

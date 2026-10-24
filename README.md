# Fine-Tuned DistilGPT-2 for Police Field-Report Narratives

MAAI 902 project – Navi Dharshan B (1716785)

A pretrained **DistilGPT-2** is fine-tuned as a causal language model on **redacted Boston Police Field
Interrogation and Observation (FIO) narratives** (2024), so that it writes in the style and vocabulary of police
field reports. The final model is trained once with a carefully chosen configuration and **early stopping**, then
evaluated on a held-out test set against the base model. A Streamlit app compares the two models live.

## Training configuration (`FIO_LM_Final_Model.ipynb`)

| Setting | Value | Why |
|---|---|---|
| Data split | 80 / 10 / 10 % (≈3,700 / 460 / 460 narratives) | enough data to delay overfitting; separate validation (early stopping) and test (metrics) sets |
| Learning rate | 1e-4, 10 % warm-up, linear decay | an earlier study showed 5e-4 overfits within 2 epochs and 5e-6 underfits |
| Batch size / block size | 8 / 128 tokens | smoother updates; fixed-length token blocks |
| Weight decay | 0.01 | mild regularisation |
| Epochs | up to 10 with `EarlyStoppingCallback` (patience 3, threshold 0.005) | stops when validation loss stops improving |
| Checkpoint | best validation loss (`load_best_model_at_end`) | deployed weights are taken before overfitting |

## Evaluation (test set, base vs fine-tuned)

Loss and perplexity, next-token accuracy (top-1 and top-5), and – for completions of unseen test narratives given
their first 15 words – ROUGE-L precision/recall/F1, BLEU-4, distinct-1/2, repetition rate, domain-style rate and the
share of outputs that end naturally. All plots and tables are saved in the model's `results/` folder on Hugging Face.

## Files

| File | Purpose |
|---|---|
| `FIO_LM_Final_Model.ipynb` | Final fine-tune with early stopping and full evaluation (Colab T4) |
| `FIO_Narrative_LM_Finetuning.ipynb` | Earlier hyperparameter study that informed the configuration |
| `app.py`, `requirements.txt` | Streamlit app |

## Data

Boston Police Department, *BPD Field Interrogation and Observation (FIO)*, `2024_annual_fio.csv`, Analyze Boston
(data.boston.gov), Open Data Commons Public Domain Dedication and License (PDDL). Only the narrative field is used;
names, officer names, addresses, neighbourhoods and phone numbers are replaced with placeholders before training.
The raw data is not stored in this repository.

## Deployment (Streamlit Community Cloud)

The fine-tuned weights are hosted on Hugging Face (`dharshan3119/fio-narrative-distilgpt2`); the app downloads them
at start-up using the secret `LM_REPO = "dharshan3119/fio-narrative-distilgpt2"`.

**Updating the deployed model.** `FIO_LM_Final_Model.ipynb` saves the model, tokenizer, `training_info.json`, a
model card and a `results/` folder (all plots and metric tables), copies them to Google Drive
(`MyDrive/FIO_LM/final_model_<date-time>/`) and pushes them to the Hugging Face repo. Rebooting the Streamlit app then
loads the new model and its charts – no change to this repository is needed.

## Responsible use

All generated text is **synthetic**: it imitates the style of field reports, describes no real event and must not
be presented as a police record. The source narratives contain demographic descriptors that the model can reproduce.

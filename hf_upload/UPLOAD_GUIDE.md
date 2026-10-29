# Hugging Face upload guide (Windows)

The dataset is staged locally at:

`D:\IndicSyntheticSceneText\hf_upload\indic-synthetic-scene-text-1000`

The staging validation is local only. No Hugging Face repository has been created and no files have been uploaded.

## A. Install or update the Hugging Face Hub CLI

This project currently has a virtual environment. From PowerShell:

```powershell
Set-Location "D:\IndicSyntheticSceneText"
.\.venv\Scripts\python.exe -m pip install --upgrade huggingface_hub
```

If `.venv` is absent, create a virtual environment with `python -m venv .venv`, then install with the environment's Python as above. Alternatively, use a separately managed Python installation and run `python -m pip install --upgrade huggingface_hub`.

## B. Authenticate securely

```powershell
hf auth login
hf auth whoami
```

Enter the token only in the interactive authentication prompt. Never put it in a script, README, source file, command-line argument, or shared terminal history. Do not share token output.

## C. Create a private dataset repository

1. Visit <https://huggingface.co/new-dataset>.
2. Sign in and create a repository named `indic-synthetic-scene-text-1000`.
3. Keep it **private** until dataset, font, and asset redistribution rights have been reviewed.
4. Use the username shown by `hf auth whoami` when forming the repository ID. Do not assume the account name is the same as a local username.

Creating the repository is a separate action; this guide does not create it.

## D. Upload only after explicit approval

Do not run the upload command until the repository ID is confirmed and the dataset owner explicitly approves publishing/uploading. Replace the placeholder below with the actual Hugging Face username:

```powershell
$RepoId = "Aus-23/indic-synthetic-scene-text-1000"
$Upload = "D:\IndicSyntheticSceneText\hf_upload\indic-synthetic-scene-text-1000"

if (!(Test-Path "$Upload\README.md")) {
    throw "Dataset README is missing."
}

hf upload $RepoId $Upload . --repo-type dataset
```

The destination repository should remain private while licensing is unresolved. The command uploads the staging directory only; it does not upload the project directory.

## E. Verify the upload

Open `https://huggingface.co/datasets/<YOUR_USERNAME>/indic-synthetic-scene-text-1000` and verify:

- The repository is private.
- `README.md` renders as a dataset card.
- `images/`, `annotations/`, and `metadata/manifest.jsonl` are present.
- `metadata/dataset_metadata.json` and source metadata are present.
- `validation_reports/` contains the source reports and staging audit.
- Sample paths in the manifest resolve to uploaded files.

The uploaded repository is not automatically a `datasets.load_dataset()` image dataset. Follow the dataset card's loading example or build an explicit loader that resolves the JSONL image paths.

## Common errors

- **Authentication failure:** Run `hf auth login` interactively, then confirm the account with `hf auth whoami`.
- **Permission/repository not found:** Confirm the username and repository ID, ensure the account owns the repository or has write access, and check that it is a Dataset repository.
- **Path or missing README:** Confirm `$Upload` points to the staging directory and that its `README.md` exists.
- **Large-file or storage error:** Review the CLI error and current Hub limits; do not work around it by uploading the whole project or adding unrelated files.
- **Network interruption:** Retry the same upload after checking connectivity and repository state. Confirm files on the repository page before retrying.
- **Unexpected staging contents:** Stop before upload and rerun the local audit with `.\.venv\Scripts\python.exe scripts\prepare_hf_dataset.py`.

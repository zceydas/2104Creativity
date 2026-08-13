# 2104 Creativity — EEG & Behavioral Analysis

Analysis code for the 2104 Creativity study: a within-subject psilocybin
experiment in which participants performed verbal creativity tasks (Remote
Associates Test, RAT, and Alternate Uses Task, AUT) while EEG was recorded.
Each participant completed three sessions — **Baseline** (always Session 1 /
`ses-0`), and **Placebo** and **Psilocybin (10 mg/70 kg)** in a
counterbalanced order across Sessions 2–3 (`ses-1`, `ses-2`). Condition order
varies per participant and is defined in the allocation file, **not** by
session number.

The repository is organized into two areas: EEG signal processing / analysis,
and behavioral (task performance) analysis.

---

## Repository structure

```
2104Creativity/
├── EEG_analysis/
│   ├── RAT_preprocessing.m          # EEG preprocessing pipeline (BDF → cleaned .set)
│   ├── build_QC_table.m             # Rebuild the QC summary table from processed files
│   └── RAT_bandpower_analysis_1.m   # Thinking-window band-power analysis & figures
└── BehavioralDataAnalysis/
    ├── 2104Creativity.ipynb         # Behavioral analysis notebook (Python)
    ├── classify_aut_all_items.py    # AUT response classification/scoring
    ├── EBRCode_CS.m                 # (behavioral scoring — see note below)
    ├── FlexibilityCalculator.m      # AUT flexibility scoring
    ├── FullDataset.Rmd / .html      # Combined-dataset report (R Markdown)
    └── SortAUTRATresults.m          # Organize/merge AUT & RAT behavioral results
```

---

## EEG analysis (`EEG_analysis/`)

These scripts require **MATLAB** with **EEGLAB** on the path. Line-noise
handling additionally uses the **zapline-plus** EEGLAB plugin (with a cleanline
fallback if it is not installed). The spectral routines implement Welch's
method locally, so the Signal Processing Toolbox is **not** required.

### `RAT_preprocessing.m`
The main preprocessing pipeline. Batch-processes every RAT recording across all
subjects and sessions, taking raw BioSemi `.bdf` files to cleaned, artifact-
corrected `.set` files, and producing per-file quality-control (QC) outputs.

Pipeline steps, in order:
1. **Import** each `.bdf` (BioSemi) and look up channel locations.
2. **Scalp-only selection** — keep the 64 scalp channels; drop the 8 external
   (EXG) channels, which are not part of the analysis montage.
3. **Downsample** to 512 Hz.
4. **Line-noise removal** (early, rank-preserving cleanline pass at 60 Hz).
5. **Bad-channel removal** via `clean_rawdata`/ASR — channels are removed but
   **not yet interpolated**, so ICA runs on full-rank data.
6. **Speech-aware ICA.** Because responses are spoken aloud, the speaking
   periods (bracketed by the `speakcode`→`insightcode` markers) are excluded
   from ICA training so jaw/muscle activity does not corrupt the decomposition.
   ICLabel then flags and removes muscle, eye, and line-noise components.
7. **Interpolate** the removed channels back and **average-reference** — done
   *after* ICA to keep the decomposition full-rank.
8. **Conditional final line-noise cleanup.** Any file still exceeding a 60 Hz
   residual threshold is cleaned with **Zapline-plus** (adaptive, per-file);
   clean files are left untouched.

Outputs, per subject/session:
- `PreICA_*.set`, `PostICA_*.set` — cleaned EEG at pre- and post-ICA stages.
- `PSD_*.png` / `.fig` — power-spectral-density overlay across all pipeline
  stages, for visual QC.
- `PSDchan_*.png` — per-channel PSD diagnostic (written only for files with
  residual line noise, to identify bad electrodes).
- `QC_summary.csv` — one row per file with channel/sample/IC-removal counts and
  line-noise metrics (see below).

### `build_QC_table.m`
Standalone scanner that **rebuilds `QC_summary.csv` from the saved
`PostICA_*.set` files** without reprocessing. The preprocessing script writes
the QC table as it runs, but a resumed/partial batch can truncate it; this
script regenerates the complete, authoritative table by reading the metrics
stored in each `.set` (channel and sample masks, IC classifications, events)
and recomputing the residual line-noise numbers. Run it after any batch —
partial or complete — to get the full table.

Key QC columns: channels removed (and labels), % samples removed by ASR, data
rank, IC counts (total / removed / muscle / eye / line), residual 60 Hz level,
whether line-noise cleanup was applied, number of speech markers, and the
condition-relevant `speech_aware_ICA` flag.

### `RAT_bandpower_analysis_1.m`
The main analysis: spectral power in the pre-speech **"thinking" window** of the
RAT, compared across drug conditions within participant.

- **Window definition.** For each trial, the thinking window runs from
  `ideacode` (202, prompt onset) to `speakcode` (203, the keypress to respond),
  with a **speech-prep buffer** trimmed off the end (default 0.5 s) to exclude
  pre-articulatory motor preparation. Timeout trials (no `speakcode`) are
  skipped.
- **Condition mapping.** Sessions are mapped to Baseline / Placebo / Psilocybin
  using the allocation spreadsheet (`Allocations.xlsx`), applying the
  Session 1/2/3 → `ses-0/1/2` shift. Order is per-participant.
- **Power.** Band power (delta, theta, alpha, beta, gamma) is computed per
  channel per trial via Welch's method, averaged over trials.
- **Statistics.** Paired within-subject comparison (Placebo vs Psilocybin) per
  band, averaged over channels, with paired *t*, *p*, and Cohen's *dz*. The
  contrast is configurable at the top of the script (e.g. Psilocybin vs
  Baseline).
- **Excluded sessions** (heavy channel/sample rejection, from QC review) are
  listed near the top and skipped.

Outputs (in an `analysis/` subfolder):
- `bandpower_long.csv` — power per subject × session × condition × band × channel.
- `bandpower_by_subject_condition.csv` — channel-averaged summary.
- `placebo_vs_drug_paired.csv` — the paired statistics table.
- `figures/` — condition-comparison spectrum, per-band paired-line plots, and
  scalp topographies of the Psilocybin − Placebo difference (`.png` and `.fig`).

> **Note:** the analysis script expects `Allocations.xlsx` on the path (columns:
> Participant ID, Dose, Session). Update the `ALLOC_XLSX` and `OutputBase` paths
> at the top of the script to match your environment.

---

## Behavioral analysis (`BehavioralDataAnalysis/`)

Scoring and analysis of task performance (RAT solutions, AUT responses).
Mixed MATLAB / Python / R.

> The descriptions below are inferred from filenames — **please verify and
> expand** each with the specifics of what it computes and its inputs/outputs.

- **`2104Creativity.ipynb`** — main behavioral analysis notebook (Python):
  loads scored responses and produces the behavioral results/figures.
- **`classify_aut_all_items.py`** — classifies/scores AUT (Alternate Uses)
  responses across all items (e.g. originality/category tagging).
- **`FlexibilityCalculator.m`** — computes AUT *flexibility* (number of distinct
  semantic categories used).
- **`SortAUTRATresults.m`** — organizes and merges the raw AUT and RAT
  behavioral output into analysis-ready tables.
- **`EBRCode_CS.m`** — *[describe: what this scores/computes]*.
- **`FullDataset.Rmd` / `FullDataset.html`** — R Markdown report combining the
  full behavioral dataset (`.html` is the rendered output).

---

## Task event codes (EEG triggers)

Sent by the task during each RAT trial (see the task's `RATtrialstructure.m`):

| Code | Name           | Meaning                                  |
|------|----------------|------------------------------------------|
| 200  | fixationstart  | fixation cross onset                     |
| 201  | fixationend    | fixation offset                          |
| 202  | ideacode       | RAT prompt presented (thinking begins)   |
| 203  | speakcode      | participant presses to respond aloud     |
| 204  | insightcode    | insight rating begins (speech finished)  |

The **thinking window** analyzed in the EEG is **202 → 203**.

---

## Typical workflow

1. **Preprocess:** run `RAT_preprocessing.m` over the raw `.bdf` files
   → cleaned `.set` files, PSD QC plots, and `QC_summary.csv`.
2. **QC:** review `QC_summary.csv` (rebuild with `build_QC_table.m` if needed);
   decide session exclusions on channel/sample rejection and residual line noise.
3. **Analyze:** list exclusions in `RAT_bandpower_analysis_1.m`, set the
   allocation and output paths, and run → statistics tables and figures.

## Requirements
- **MATLAB** (developed on R2024b) with **EEGLAB** (2024.2), plus the
  **clean_rawdata**, **ICLabel**, **cleanline**, and **zapline-plus** plugins.
- **Python** and **R** for the behavioral scripts (see individual files).

# Cost-Optimal Seller Segment Assignment for Alternative Credit Scoring

> **Anonymised for double-blind review.** This is the anonymised replication
> repository accompanying a manuscript under review. Author, affiliation, and
> antecedent-citation details are withheld and will be restored on acceptance.
> Anonymous mirror: `https://anonymous.4open.science/r/seller-level-cost-optimal-acs-0634`

Simulation code, derived results, and manuscript sources for a study on
**seller-level segment assignment** and **cost-optimal routing thresholds**
in alternative credit scoring (ACS).

This repository is a **direct continuation** of the *Explainable Seller
Segmentation Framework* (ESSF) and reuses its publicly released dataset and
Stage-1 classifier. It extends that framework in two directions: it lifts
segment assignment from the **store** to the **seller** (who may operate
several stores), and it replaces the antecedent's arbitrary auto-assignment
threshold with a **cost-minimising optimum**.

> **Antecedent study.** This work directly continues a prior study that
> introduced the *Explainable Seller Segmentation Framework* (ESSF) for
> product-name-based segment auto-assignment in alternative credit scoring,
> and reuses its publicly released dataset and Stage-1 classifier. The
> antecedent reference is withheld here to preserve double-blind review and
> will be restored on acceptance.

---

## What this study adds

The antecedent framework left two questions open — one technical, one
financial — and this study resolves both.

1. **Store → Seller (technical).** Assignment is lifted to the seller level and
   seller uncertainty is decomposed into a **within-store** term `W_s^dist`
   and a **between-store** term `B_s`:

   ```
   SCS_s = W_s^dist × B_s
   ```

   A single-store seller has `B_s = 1` and reduces exactly to the antecedent
   store-level score. Empirically, **misassigned sellers are distinguished
   chiefly by low between-store agreement**, identifying cross-store
   disagreement as the dominant source of unreliability.

2. **Arbitrary τ → Cost-optimal τ\* (financial).** Routing is cast as a
   cost-sensitive decision. The expected cost of a threshold `τ` is

   ```
   E[C(τ)] = c_err ∫_τ^1 e(s) f(s) ds  +  c_rev ∫_0^τ f(s) ds
   ```

   whose first-order condition equates the marginal misassignment rate to the
   inverse cost ratio:

   ```
   e(τ*) = c_rev / c_err = 1 / ρ
   ```

   Monotonicity of the score-conditional misassignment rate `e(s)` — enforced
   via **isotonic regression** — guarantees a **unique** optimum. Relative to
   the antecedent's fixed 80% rule, the cost-optimal threshold reduces expected
   cost by **28–95%** as the cost ratio rises.

---

## Research questions and headline results

| RQ | Question | Result |
|----|----------|--------|
| **RQ1** | Does a seller-level SCS track assignment reliability, and does a distribution- or confidence-based within term track it more robustly? | Monotone score–reliability curve (0.59 → 1.00 across SCS ranges); distribution-based SCS is more robust across backends. |
| **RQ2** | Is uncertainty dominated by within-store mixing or between-store disagreement? | **Between-store.** Misassigned vs correct: `B_s` 0.686 vs 0.876; `H_btw` 0.637 vs 0.236. |
| **RQ3** | What is the cost-optimal threshold τ\*, and how much does it save? | FOC `e(τ*) ≈ 1/ρ` holds at interior crossings; **28–95%** cost reduction vs the fixed 80% rule. |
| **RQ4** | How does the inclusion–risk trade-off shift across markets? | Emerging vs developed markets equalise risk (~2.6–2.8%) but differ **tenfold** in inclusion (7.7% vs 78.5%). |

All findings are stable across four Stage-1 backends, indicating they are
properties of the framework rather than of any one classifier.

---

## Repository layout

```
.
├── data/                     # reused 110k Korean product-name corpus (train/val/test)
├── notebooks/                # end-to-end pipeline, run in order 00 → 06
│   ├── 00_data_prep_classifier.ipynb   # Stage-1 classifiers (TF-IDF + 3 PLMs)
│   ├── 01_seller_simulation.ipynb      # synthesise sellers under controlled regimes
│   ├── 02_seller_scs.ipynb             # seller-level SCS (distribution vs confidence)
│   ├── 03_tau_optimization.ipynb       # cost-optimal τ*, FOC, vs 80% rule
│   ├── 04_inclusion_risk_tradeoff.ipynb# inclusion–risk frontier, market scenarios
│   ├── 05_figures.ipynb                # all result figures (grayscale, 600 dpi)
│   ├── 06_robustness.ipynb             # multi-backend RQ1 robustness
│   └── 07_robustness_and_cost_extensions.ipynb  # directional cost, fair baselines, CIs
├── results/
│   ├── tables/               # all derived result tables (.csv)
│   └── figures/              # result figures (.png and .pdf)
└── artifacts/                # intermediate outputs (predictions, seller SCS, curves)
```

---

## Data

The dataset is the antecedent study's publicly released corpus: **110,000
Korean product names** from Naver Smart Store, balanced at 10,000 records
across eleven depth-1 categories, split 70/10/20 into `train.csv` /
`val.csv` / `test.csv`. Each record is a product name and its depth-1 label.
No new data are collected; the seller structure studied here is introduced by
simulation from this corpus.

---

## Requirements

The pipeline runs on CPU except for notebook `00`, which fine-tunes
transformer classifiers and benefits from a GPU.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# For notebook 00 (GPU recommended); match the CUDA build to your driver
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install transformers datasets accelerate \
            scikit-learn pandas numpy scipy \
            seaborn matplotlib jupyter nbformat nbconvert
```

Tested with Python 3.12, PyTorch 2.6 (CUDA 12.4), scikit-learn 1.8.
Notebook `00` was run on a single NVIDIA RTX 4060 Ti; each base PLM trains in
~13 minutes.

---

## Reproducing the results

Run the notebooks in order from the `notebooks/` directory. Only `00` needs a
GPU; `01`–`06` run on CPU in a few minutes.

```bash
cd notebooks
for nb in 00_data_prep_classifier 01_seller_simulation 02_seller_scs \
          03_tau_optimization 04_inclusion_risk_tradeoff 05_figures \
          06_robustness 07_robustness_and_cost_extensions; do
  jupyter nbconvert --to notebook --execute --inplace ${nb}.ipynb \
          --ExecutePreprocessor.timeout=1500
done
```

**Stage-1 backends.** Notebook `00` trains a character *n*-gram
**TF-IDF + logistic regression** model plus three Korean pre-trained language
models — `klue/roberta-base`, `klue/bert-base`, and
`KoELECTRA-base-v3` — under the antecedent study's hyperparameters
(max length 64, AdamW, warmup ratio 0.1, 5 epochs, batch 64, lr 2e-5). Test
Macro-F1: 0.872 (RoBERTa-base), 0.864 (TF-IDF+LR), 0.863 (BERT-base),
0.859 (KoELECTRA), reproducing the antecedent ordering.

**Choosing the main backend.** Notebooks `01`–`05` read
`artifacts/test_predictions.csv` as the main backend. To make
`klue/roberta-base` the main backend (as in the manuscript), copy its
predictions over the defaults before running `01`:

```bash
cd artifacts
cp test_predictions_klue_roberta_base.csv test_predictions.csv
cp test_proba_klue_roberta_base.npy       test_proba.npy
cd ../notebooks
```

Notebook `06` reads all four backends' predictions automatically for the RQ1
robustness comparison.

---

## Key outputs

**Tables** (`results/tables/`)

| File | Content |
|------|---------|
| `table_scs_monotonicity_distribution.csv` / `..._confidence.csv` | RQ1: bin-level assignment accuracy by SCS range |
| `table_scs_assignment_correlation.csv` | RQ1: correlation of each SCS variant with assignment correctness |
| `table_robustness_backends.csv` / `..._binlevel.csv` | RQ1: cross-backend robustness |
| `table_within_between_diagnosis.csv` | RQ2: within/between components for correct vs misassigned |
| `table_tau_star_by_rho.csv` | RQ3: cost-optimal τ\* by cost ratio |
| `table_tau_foc_check.csv` | RQ3: first-order condition check |
| `table_tau_asymmetric.csv` | RQ3: τ\* under FN/FP cost asymmetry |
| `table_tau_vs_80rule.csv` | RQ3: cost saving vs the antecedent 80% rule |
| `table_optimal_operating_points.csv` | RQ4: inclusion–risk operating points |
| `table_market_scenarios.csv` | RQ4: emerging vs developed market policy |
| `table_tau_directional.csv` | Directional FN/FP cost: τ\* under error-direction asymmetry |
| `table_tau_fair_baseline.csv` | Cost-optimal vs 80% rule vs symmetric-cost optimum |
| `table_rq2_within_regime.csv` | Within-regime within/between gaps (composition check) |
| `table_scs_tail_density.csv` | SCS upper-tail density (threshold-saturation regime) |
| `table_bootstrap_ci.csv` | Bootstrap 95% CIs for headline quantities |

**Figures** (`results/figures/`, grayscale, 600 dpi, PNG + PDF)

`fig1` SCS monotonicity · `fig2` within/between · `fig3` between-store by
regime · `fig4` τ\*(ρ) curve · `fig5` expected-cost curves · `fig6` asymmetric
cost · `fig7` inclusion–risk frontier · `fig8` isotonic e(s) · `fig9`
cross-backend robustness.

---

## Method notes

- **Seller simulation.** Virtual sellers are drawn from the real test pool,
  preserving each product's true label, prediction, confidence, and
  correctness. Four factors are manipulated: dominant-category regime
  (focused / cross / diversified), store count `M ∈ {1,2,3,5}`, product count
  `N ∈ {30,100,300}`, and a difficulty tilt. The grid yields 36 cells × 40
  replicates = **1,440 sellers per disagreement condition** (Method A: natural;
  Method B: noise-injected, η = 0.15).
- **Isotonic e(s).** The score-conditional misassignment rate is fit under a
  monotone non-increasing constraint, which guarantees a unique cost-optimal
  threshold and cleans up small-sample noise in the raw binned estimate.
- **High-ρ saturation.** For high cost ratios the optimum lies in the sparse
  upper tail of the SCS distribution, where `e(·)` has already reached zero; the
  FOC is then satisfied trivially rather than at an interior crossing. This is a
  property of the simulated score distribution.
- **Directional cost model** (notebook 07). Beyond the single-cost objective,
  misassignments are typed by direction — routing into a lower-risk vs a
  higher-risk segment than the true one — and the threshold is re-optimised
  under an FN/FP asymmetry, with segment risk proxied by classification
  difficulty.
- **Within-regime check** (notebook 07). The pooled between-store effect is
  recomputed within each dominant-ratio regime. The effect attenuates and
  reverses within a homogeneous regime, indicating it is a population-level
  association driven by the prevalence of multi-store, category-diverse sellers
  rather than a within-regime causal relationship — reported honestly as such.

---

## Limitations

Seller structure is **simulated**, not observed: public data linking multiple
stores to one owner are not available. Cost parameters (`ρ`, FN/FP asymmetry)
are **illustrative**, swept over ranges rather than calibrated to a specific
institution. The SCS–misassignment relationship is a simulation product; its
external validity against realised credit outcomes is left to future work.

---

## Citing

Full citation details for the antecedent study and this work are withheld to
preserve **double-blind review**, and will be added on acceptance. Reviewers
are referred to the anonymised antecedent as the *Explainable Seller
Segmentation Framework* (ESSF).

```bibtex
@article{essf_anonymous,
  author  = {Anonymous},
  title   = {Explainable Seller Segment Auto-Assignment for Alternative Credit Scoring},
  journal = {[journal withheld for double-blind review]},
  year    = {2026},
  note    = {Antecedent study; full citation restored on acceptance}
}
```

---

## License

Released for research replication and extension. See the antecedent study's
data terms for the reused corpus.

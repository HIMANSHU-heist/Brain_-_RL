# Can Brain Signals Tell Learning Models Apart?

**Decoding reward prediction error (RPE) from EEG with three reinforcement-learning models, two decoders and two datasets.**

Several reinforcement-learning (RL) models can explain human choices almost equally well, yet they predict different trial-by-trial RPE. This project asks whether EEG can help decide which model is closer to the truth, and it reports honestly what held up and what did not.

> **One-line summary:** On ds004295, a small but robust parieto-occipital interaction (predicted RPE x reward) appears for Rescorla-Wagner (M1) and Pearce-Hall (M3) but not for the dual-learning-rate model (M2). It survives seeds, leave-one-person-out checks and a global 90-test FDR correction, but it was **not replicated** on ds003458 (p = 0.17).

---

## Research question

Can competing RL models be told apart by decoding their predicted RPE from EEG, and does any such difference hold across decoder types and across independent datasets?

## Models

| Model | Idea | Free parameters |
|---|---|---|
| **M1** Rescorla-Wagner | One fixed learning rate | alpha, beta |
| **M2** Dual learning rate | Separate rates after wins and losses | alpha_gain, alpha_loss, beta |
| **M3** Pearce-Hall | Learning rate adapts to recent surprise | alpha0, kappa, eta, beta |

Trial-by-trial RPE from the three models is very similar (mean correlation: M1-M3 0.970, M1-M2 0.938, M2-M3 0.906), which is why they are hard to separate.

## Datasets

| | ds004295 (main) | ds003458 (replication attempt) |
|---|---|---|
| People used | 23 (of 26) | 23 |
| Task | Two-option reversal learning, reward and punishment blocks | Three-armed bandit, slowly drifting probabilities |
| Trials per person | 560 | ~479 |
| EEG | BioSemi, 1024 Hz, 66 raw channels | 64 channels, 500 Hz epochs |
| Regions | Frontal 8, central 8, parieto-occipital 9 | Frontal 16, central 20, parieto-occipital 17 |
| Decoding design | Leave-one-person-out, multi-channel input | Within-person 5-fold, one averaged trace per region |

Epochs are -0.2 to 0.8 s around feedback. **Note:** the two decoding designs are not identical, so ds003458 is not a fully matched replication.

## Method

1. **Behaviour first:** choices and feedback verified against the dataset authors' ground-truth files; each behaviour record matched to the right person by reward sequence vs EEG feedback markers (23 of 23 exact).
2. **Model fitting** per person and block (several random starts), with **parameter recovery** checks.
3. **Model comparison:** in-sample (AIC) and out-of-sample (fit on one block, test on the other).
4. **EEG decoding:** 1D-CNN and a compact Transformer predict RPE per region, leave-one-subject-out, 4 seeds.
5. **Confound control:** raw correlation, partial correlation after removing win/loss, and a predicted-RPE x reward **interaction test**.
6. **Robustness:** jackknife over people, global FDR over 90 planned tests, z-scored interaction with permutation test, gradient-boosting baselines, motor (reaction-time) control.

## Key results

- **Behaviour:** In-sample AIC prefers M2 (27 of 46 person-blocks), but out-of-sample no pair differs significantly (p = 0.29 to 0.83). The same lack of a clear winner appears on ds003458.
- **Decoding is mostly win vs loss:** raw correlations are small (frontal 0.080, central 0.066, parieto-occipital 0.047), and partial correlations after removing reward are about zero in every region.
- **Robust interaction (ds004295, parieto-occipital):** M1 beta = -0.429 (p = 0.0025), M3 beta = -0.488 (p = 0.0027), M2 not significant (p = 0.61). Significant in 4/4 (M1) and 3/4 (M3) CNN seeds, 23/23 jackknife runs, and survives global FDR (adjusted p ~ 0.013).
- **Decoder matters:** a Transformer trained for 30 epochs found nothing (under-trained); with early stopping the M1 effect appeared in 4/4 seeds (z-scored test).
- **M1 vs M3 cannot be separated** (paired p median 0.57), so the defensible claim is M1/M3 vs M2.
- **Replication on ds003458:** interaction not significant (M1 p = 0.1735, M3 p = 0.1745, 0/23 jackknife runs), sign negative as in ds004295.

## What is and is not claimed

**Claimed**
- A robust, modest parieto-occipital interaction in ds004295 for M1 and M3, not M2.
- In-sample and out-of-sample behavioural model comparisons can disagree (seen in both datasets).
- A poorly trained decoder can produce a false "no effect".

**Not claimed**
- That EEG decodes graded RPE (partial correlations ~ 0).
- That M1 and M3 are neurally different.
- That M2 is the best model (its in-sample edge is probably flexibility).
- That the finding replicates, or that we know which model the brain uses.

## Limitations

- Small samples (23 people per dataset), low power for subtle model differences.
- ds003458 used a different decoding design (shuffled within-person folds can leak information between neighbouring trials).
- M3's individual parameters are poorly recovered; M2/M3 recovery not yet re-checked on ds003458.
- Out-of-sample test also changes the task (reward to punishment).
- The 25-of-66 channel selection and EEG cleaning for stored region tensors are not fully documented.
- The final person-matching step is not yet included in the notebook cells.
- Diffusion-based augmentation was planned but not carried out.

## Repository structure

```
.
├── README.md
├── report/
│   └── RL_EEG_Final_Report.pdf
└── notebooks/
    ├── 1_data_understanding_behaviour_ds004295.ipynb  # data inspection, behaviour fixes, matching
    ├── 2_dataset2_ds003458.ipynb                      # fitting, recovery, EEG decoding, replication attempt
    └── 3_main_analysis_pipeline.ipynb                 # LOSO CNN/Transformer, confounds, FDR, jackknife
```

## How to reproduce

1. **Notebook 1:** download ds004295, extract events and epochs, verify behaviour against the ground-truth files.
2. **Notebook 3:** load stored region tensors and fits, run the cached leave-one-person-out decoding, then the confound, jackknife and FDR analyses. Expensive steps are cached so results can be regenerated exactly.
3. **Notebook 2:** run the ds003458 behaviour and within-person decoding replication attempt.

Main libraries: PyTorch, scikit-learn, MNE, NumPy, SciPy, pandas, statsmodels.

## Future work

- Run the exact ds004295 pipeline (multi-channel tensors, leave-one-person-out, same regression) on ds003458.
- Use blocked folds instead of shuffled folds; decode M2 on ds003458 too.
- Hierarchical (Bayesian) parameter fitting; more models; time-resolved analysis.
- Pick one primary significance test in advance (z-scored + permutation).
- A third reversal-learning dataset for an independent replication.

## Data sources

- ds004295 (Stolz, Pickering & Mueller), OpenNeuro
- ds003458 (Cavanagh lab), OpenNeuro

Please follow each dataset's license and citation requirements.

## License

Add your preferred license here (e.g. MIT).

## Author
Himanshu sunil Bendale
Second-year undergraduate student. Feedback and issues welcome.

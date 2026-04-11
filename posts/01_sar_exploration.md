# Chemical Space, Activity Patterns, and Structure–Activity Relationships in the PXR Challenge Dataset

*April 2026*

---

## Dataset Overview

The challenge comprises four experimental screens consolidated into a single table of **12,782 unique compounds** (identified by InChIKey). The datasets are hierarchically nested rather than independent:

| Dataset | Compounds | Role |
|---|---|---|
| Single-dose screen | 12,269 | Primary funnel — all tested compounds |
| Dose-response (training) | 4,138 | Quantitative pEC50 + Emax; prediction target |
| Counter screen | 2,858 | Selectivity filter; subset of dose-response |
| Test set (blinded) | 513 | No labels; held-out for prediction |

The dose-response set is entirely contained within the single-dose screen (0 DR compounds absent from single-dose). The counter screen is entirely contained within the dose-response set. The test set has **no overlap** with either the dose-response training set or the counter screen — it is a clean hold-out.

Of the 12,269 screened compounds, **8,131 appear only in the single-dose screen** and have no dose-response data. These represent the broad hit-finding funnel from which the quantitative training data was derived.

---

## Activity Distribution

### Single-dose hit rates

At 30 µM — the most widely tested concentration — **25.6% of compounds** (2,439 / 9,523) were classified as hits (median log₂FC > 1, FDR-BH < 0.05). The hit rate drops sharply with concentration: 8.1% at 10 µM and near-baseline at 1 µM (only 1 compound recorded as a hit). At the highest concentration tested (100 µM, a much smaller subset of 706 compounds), 61.2% were hits, consistent with compounds active only at high concentration. The dose-response assay then quantifies potency for the 4,138 compounds that passed initial triage.

### pEC50 distribution in the dose-response training set

The pEC50 distribution (dose-response, n = 4,138) is left-skewed, concentrated in the low-to-moderate potency range:

- **Mean pEC50: 4.32** (SD 1.12); **median: 4.65**
- 30.1% of compounds have pEC50 < 4 (weak activity; EC50 > 100 µM)
- 68.3% fall in the 4–6 range (moderate activity)
- Only **67 compounds (1.6%) have pEC50 ≥ 6** (EC50 ≤ 1 µM)
- A single compound reaches pEC50 = 7.55

The pEC50 range spans nearly 6 log units (1.61–7.55), which is substantial and beneficial for model training. Emax (log₂FC vs. baseline) ranges from 1.59 to 5.74, with 26% of compounds showing Emax ≥ 3 (strong induction) and 59% in the 2–3 range. The combination of a wide dynamic range and a realistic proportion of weak and strong activators makes this a challenging but tractable regression problem.

### Counter-screen selectivity

Of the 4,138 dose-response compounds, 2,858 were also tested in a counter screen (orthogonal assay to flag non-specific hits), and 2,646 of those have quantitative pEC50 values in both assays. The correlation between the two pEC50 values is low (r = 0.107), confirming that the counter screen captures a different biological signal rather than a technical replicate of the primary assay.

Among the 2,646 compounds with dual measurements:
- **1,721 (65%)** are dose-response selective: pEC50_DR > pEC50_counter + 1
- **1,129 (43%)** are highly selective: pEC50_DR > pEC50_counter + 2
- **875 (33%)** are non-selective: |pEC50_DR − pEC50_counter| ≤ 1
- Only **50 (2%)** are counter-selective (stronger counter signal than primary)

The low correlation and large fraction of DR-selective compounds indicate that most activity in the primary screen is target-specific rather than assay artefact.

---

## Chemical Space

### UMAP and t-SNE embeddings

ECFP6 fingerprints (radius 3, 4096-bit, count-based, chirality-aware) were computed for all 12,782 compounds. UMAP (Jaccard metric) and t-SNE (PCA pre-reduction to 50 components, then t-SNE) both project the full collection into 2D.

![UMAP ECFP6 Chemical Space](../plots/1_sar_exploration/umap_ecfp6_chemical_space.png)

The UMAP plot reveals a single, broadly dispersed cloud with no strong cluster separation. The absence of tight, well-separated clusters indicates **high chemical diversity without dominant scaffold families**. Dose-response compounds (light blue) and counter-screen compounds (dark blue) are distributed throughout the same space as the single-dose compounds (grey), confirming that the hit-selection pipeline did not introduce a strong structural bias — the quantitative subset samples the same broad chemical space as the full library.

The test set (dark green, n = 513) is similarly distributed across the embedding. There are no obvious regions of the chemical space that are entirely absent from the training data, though the test set is sparse relative to the training density.

![t-SNE ECFP6 Chemical Space](../plots/1_sar_exploration/tsne_ecfp6_chemical_space.png)

The t-SNE embedding tells the same story. The library is diffuse; no structural class dominates. A handful of small, isolated clusters at the periphery of the t-SNE plot likely represent structurally unique scaffolds that have few analogues in the collection.

### pEC50 landscape across chemical space

![UMAP Chemical Space coloured by pEC50](../plots/1_sar_exploration/umap_pec50_chemical_space.png)

The UMAP of dose-response compounds coloured by pEC50 shows that high-potency compounds (yellow, pEC50 > 6) are **scattered throughout chemical space** rather than concentrated in one region. There is no single structural cluster that can be identified as the sole source of active compounds. This has two implications: (1) the SAR is multi-modal — multiple chemotypes contribute to activity; (2) global structural similarity to known actives is a poor proxy for potency, which will challenge similarity-based models.

---

## Pairwise Similarity Distributions

### Within the dose-response set

Pairwise Tanimoto similarities were computed for all unique compound pairs in the dose-response training set using three fingerprints.

![Pairwise similarity distributions — dose-response compounds](../plots/1_sar_exploration/density_sim_fingerprints.png)

All three distributions are heavily left-skewed with modes below 0.2, confirming that the dose-response set is structurally diverse. ECFP4 and ECFP6 peak near Tanimoto = 0.1, while MACCS (which captures broader pharmacophoric features) has a wider distribution peaking near 0.45. The ECFP6 distribution is narrower and shifted left relative to ECFP4, reflecting its sensitivity to finer structural differences. The very low inter-compound similarities mean that threshold-based activity cliff detection at Tanimoto ≥ 0.4 (ECFP4) or ≥ 0.8 (MACCS) captures only a small fraction of pairs.

### Train–test similarity comparison

![ECFP4 Tanimoto similarity — train/test comparison](../plots/1_sar_exploration/density_sim_train_test.png)

The three distributions — within-train, within-test, and train-vs-test — are virtually superimposable. All three peak near Tanimoto = 0.12–0.15 and decay in the same manner. This is a key observation: **the test set is statistically indistinguishable from the training set in terms of structural diversity and coverage**. Models should not be expected to extrapolate into structurally novel space; the challenge is interpolation within the same broad chemical landscape, not extrapolation beyond it. The close alignment of train-vs-test with within-train means that nearest-neighbour similarity-based baselines will have access to reasonable analogues for most test compounds.

---

## Matched Molecular Pair Network

Matched molecular pairs (MMPs) were enumerated using mmpdb across all 12,782 compounds, yielding 68,476 raw pairs. After filtering to retain only pairs where the variable substituent is smaller than the common core (core_transform_ratio < 1.0), **10,272 valid MMP pairs** remain. Restricting to the dose-response training subset yields **1,503 MMP pairs connecting 783 unique compounds** (19% of the dose-response set).

![MMP network](../plots/1_sar_exploration/mmp_network.png)

The MMP network shows a fragmented topology: one large hub cluster (lower left), several medium clusters, and a large number of isolated pairs. The large, diffuse grid-like region (upper portion) represents compounds with single-substituent changes at common positions — many light systematic analogues sharing a common core. The denser cluster at lower left is a more deeply explored chemotype with multiple overlapping transformations. Test-set compounds (green) appear in several clusters alongside training compounds, which is consistent with the near-identical similarity distributions seen above.

---

## Activity Cliffs

### MMP-based cliffs

Applying a |ΔpEC50| ≥ 2 threshold to the 1,503 MMP pairs within the dose-response set identifies **60 activity cliff pairs** involving **66 unique compounds** — a small but significant fraction of the MMP-connected compounds (7.7%). The maximum observed ΔpEC50 across an MMP is 3.77 log units, and the mean is 2.54.

The scarcity of MMP cliffs is consistent with the low overall MMP propensity of a diverse library: when structural neighbours are rare, steep potency gradients have fewer opportunities to be observed. However, those 60 pairs that do constitute cliffs carry the most information-dense SAR signal in the dataset — they directly demonstrate which single structural changes produce the largest potency changes.

### Fingerprint-based cliffs (ECFP4, Tanimoto ≥ 0.4)

Using a broader similarity definition (any pair with ECFP4 Tanimoto ≥ 0.4, not limited to MMP-type transformations), the cliff count increases substantially. Of all pairs above the 0.4 similarity threshold, a notable fraction exceed the |ΔpEC50| ≥ 2 criterion, with some cliff pairs reaching maximum pEC50 values above 6. These cases — similar structures with one highly potent and one weak — represent the most challenging prediction targets, as small structural errors will translate to large pEC50 errors.

---

## Scaffold Analysis

Scaffold decomposition (ring-system → linked ring systems → full Bemis-Murcko scaffold) was performed for all unique molecules. The three scaffold levels capture structural diversity at increasing complexity:

- **Ring systems**: individual fused/bridged polycyclic cores
- **Linked ring systems**: pairs of ring systems joined by a linker
- **Full scaffold**: all ring systems and linkers for molecules with ≥ 3 ring systems

The scaffold coverage scatter (dose-response training vs. test set, log-log scale) shows that the most frequent scaffolds in the training set are well-represented in the test set — confirming the statistical alignment seen in the similarity distributions. Scaffolds that appear in hundreds of training compounds also appear in tens of test compounds. However, some test-set scaffolds are present in only 1–2 training compounds (low training coverage), flagging potential prediction difficulty for those test molecules.

---

## Implications for Modelling

Several structural properties of this dataset carry direct consequences for model design and evaluation:

1. **High diversity, low MMP propensity.** Similarity-based methods will struggle: most test compounds have no close structural analogues in the training set at ECFP4 Tanimoto ≥ 0.4. Models that rely on interpolation in fingerprint space must deal with sparse neighbourhoods throughout.

2. **Multi-modal SAR.** Potent compounds are dispersed across chemical space with no dominant structural family. A model with strong local accuracy in one region of structure space cannot be assumed to generalise to others.

3. **Activity cliffs are rare but high-stakes.** The 60 MMP cliff pairs (ΔpEC50 ≥ 2) will disproportionately penalise models under root-mean-squared-error metrics. Models with smooth, continuous representations are likely to underestimate these cliffs.

4. **Train–test distribution parity.** The near-identical similarity distributions remove one class of failure mode: the test set does not represent an out-of-distribution generalisation problem. This suggests that optimising in-distribution accuracy is the right objective, and that temporal or scaffold-based train–test splits would have been a harder but more realistic challenge design.

5. **Counter screen provides selectivity signal.** The low correlation (r = 0.107) between primary and counter-screen pEC50 values means the counter screen adds orthogonal information. Incorporating it as an auxiliary target in a multi-task model may improve primary activity predictions and, if the challenge eventually scores selectivity, it is directly relevant.

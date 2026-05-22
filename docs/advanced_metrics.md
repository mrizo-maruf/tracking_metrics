# Advanced Metrics (v0.4)

## HOTA

**Higher Order Tracking Accuracy** (Luiten et al., 2020) jointly measures detection and association quality via a threshold sweep.

### Algorithm

For each localization threshold α ∈ {0.05, 0.10, …, 0.95} (19 thresholds):

1. Run Hungarian matching on the raw similarity matrix with threshold α.
2. Compute **detection** scores:
   - **DetRe(α)** = TP / (TP + FN)
   - **DetPr(α)** = TP / (TP + FP)
   - **DetA(α)** = √(DetRe × DetPr)
3. Compute **association** scores via a per-GT-track Jaccard index:
   - For each GT track *i*, find the best matching predicted track *j\** = argmax_j A[i][j], where A[i][j] = number of frames where GT track i and pred track j were matched.
   - Jaccard(i) = A[i][j*] / (N_gt_i + N_pred_j* − A[i][j*])
   - **AssA(α)** = mean Jaccard over all TP detections
   - **AssRe(α)**, **AssPr(α)** — recall/precision analogues
4. **HOTA(α)** = √(DetA(α) × AssA(α))
5. **LocA(α)** = mean similarity of matched pairs (localization quality)
6. **OWTA(α)** = √(DetRe(α) × AssA(α)) — open-world variant

Final scores are the mean over all thresholds.

### Output keys

| Key | Description |
|-----|-------------|
| `HOTA` | Mean HOTA across thresholds |
| `DetA` | Mean detection accuracy |
| `AssA` | Mean association accuracy |
| `LocA` | Mean localization accuracy |
| `OWTA` | Mean open-world tracking accuracy |
| `DetRe` | Mean detection recall |
| `DetPr` | Mean detection precision |
| `AssRe` | Mean association recall |
| `AssPr` | Mean association precision |
| `HOTA_curve` | Per-threshold HOTA list (only if `return_curves=True`) |
| `DetA_curve` | Per-threshold DetA list |
| `AssA_curve` | Per-threshold AssA list |
| `LocA_curve` | Per-threshold LocA list |

### Usage

```python
from tracking_metrics.metrics.hota import HOTAMetric

# Default: 19 thresholds, no curves
metric = HOTAMetric()

# Custom thresholds + curves
metric = HOTAMetric(thresholds=[0.25, 0.5, 0.75], return_curves=True)
```

### Notes

- HOTA requires the full per-frame similarity matrix stored in `FrameResult.similarity_matrix`. This is populated automatically by the evaluator.
- HOTA is matcher-agnostic: the similarity matrix is provided by whichever matcher you use (IoU-based or distance-based).
- For center-distance matching the similarity is `max(0, 1 − d/max_distance)`, so HOTA thresholds are applied on this normalized scale.

---

## Fragmentations (`Frag`)

Counts how many times a GT track is "broken" (disappears then reappears) due to missed detections.

A **fragmentation** occurs when a matched segment ends and another begins for the same GT track. A track matched across all its frames has 0 fragmentations; a track with two separate segments has 1 fragmentation.

```python
from tracking_metrics.metrics.fragmentation import FragmentationsMetric

metric = FragmentationsMetric()
# Returns: {"Frag": int}
```

---

## Track Coverage (`MT`, `PT`, `ML`)

Classifies each GT track by the fraction of its frames that were successfully matched:

| Class | Condition | Description |
|-------|-----------|-------------|
| MT (Mostly Tracked) | coverage ≥ 0.8 | ≥ 80% of frames matched |
| PT (Partially Tracked) | 0.2 ≤ coverage < 0.8 | partially detected |
| ML (Mostly Lost) | coverage < 0.2 | < 20% of frames matched |

Thresholds are configurable.

```python
from tracking_metrics.metrics.track_coverage import TrackCoverageMetric

metric = TrackCoverageMetric(mt_threshold=0.8, ml_threshold=0.2)
# Returns: {"MT": int, "PT": int, "ML": int, "MT%": float, "PT%": float, "ML%": float}
```

---

## Track Survival Rate (`T-SR`)

Fraction of GT tracks that appear in at least one matched detection.

```python
from tracking_metrics.metrics.track_survival import TrackSurvivalRateMetric

metric = TrackSurvivalRateMetric()
# Returns: {"T-SR": float, "SurvivedTracks": int, "TotalGTTracks": int}
```

---

## ID Consistency (`IDCons`)

For each GT track that has at least one match, computes the fraction of its matched frames that share the most-common predicted ID. Averaged across all GT tracks.

A value of 1.0 means every GT track is always matched to the same predicted ID. A value of 0.5 means on average half the matched frames use the majority ID.

```python
from tracking_metrics.metrics.id_consistency import IDConsistencyMetric

metric = IDConsistencyMetric()
# Returns: {"IDCons": float}
```

---

## CLI

```bash
# Single advanced metric
track-metrics evaluate --gt gt.json --pred pred.json --metrics hota

# HOTA with per-threshold curves saved to JSON
track-metrics evaluate \
  --gt gt.json --pred pred.json \
  --metrics hota --hota-curves \
  --output results.json

# All advanced metrics
track-metrics evaluate \
  --gt gt.json --pred pred.json \
  --metrics hota --metrics frag \
  --metrics track-coverage --metrics t-sr --metrics id-cons

# Save as CSV or LaTeX
track-metrics evaluate --gt gt.json --pred pred.json --output results.csv
track-metrics evaluate --gt gt.json --pred pred.json --output results.tex
```

Supported metric names: `hota`, `frag`, `track-coverage`, `t-sr`, `id-cons`.

---

## Python reporting utilities

```python
from tracking_metrics.report.csv_report import save_csv_report
from tracking_metrics.report.latex_table import save_latex_report, results_to_latex
from tracking_metrics.report.summary import format_summary

save_csv_report(results, "results.csv")
save_latex_report(results, "results.tex", caption="Tracking results", label="tab:results")
print(format_summary(results, precision=4))
```

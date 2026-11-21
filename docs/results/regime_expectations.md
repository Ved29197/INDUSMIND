# Operating regimes and DBSCAN: expectations written before the evaluation

Setup: everything fitted on 8,000 training machines without labels; test machines (2,000) are
only assigned or scored. k chosen as the best silhouette among k >= 3. DBSCAN eps = 95th
percentile of training 16-nearest-neighbour distances. No tuning.

## Expectations
1. Structure is weak: best silhouette below 0.35. The regimes are a convenient partition of
   continuous operating conditions, not natural groups.
2. Failure rate differs strongly between regimes: the highest-failure regime is at least twice
   the lowest on the training set, and the same regimes are highest and lowest on the test set.
3. No regime is a time period: every regime has a largest-decile share below 0.25.
4. DBSCAN finds one main dense cluster and labels fewer than 3% of machines as noise, because
   noise requires being far from every dense region.
5. The k-distance score (top 5% flagged) has a test PR-AUC within 0.10 of Isolation Forest's
   0.302; the two methods flag overlapping but not identical machines (overlap 0.2 to 0.7).
6. Both detectors rarely flag heat-dissipation failures (under 30%), for the same reason as in
   Milestone 5.1.
# Datasheet: planted-leakage evaluation suite

This summarizes Appendix C of the paper, which is the authoritative version.

**Motivation.** The suite benchmarks leakage-audit tools and types of evidence on planted leakage whose ancestry is known. It includes benign look-alikes that defeat value-only rules. It separates two kinds of leak:
- **Identifiable:** exact copies on deduplicated tables, and a declared leaky scope.
- **Partly or not identifiable:** near copies, copies on raw tables, and a hidden scope.

**Composition.**
- **Families.** Ten source families, each in a raw and a deduplicated (dd) variant, under three seeds (104729, 130363 and 155921).
  - OpenML 44 (spambase), 1067 (kc1), 1462 (banknote), 1464 (blood-transfusion), 1487 (ozone-level), 1489 (phoneme) and 1494 (qsar-biodeg).
  - UCI 159 (MAGIC gamma), 329 (diabetic retinopathy) and 563 (Iranian churn).
- **Main study.** 21 arms (13 positive and 8 negative), giving 1,260 episodes: 780 positive and 480 negative, 60 per arm.
- **Matched arms.** With the two matched L5 arms there are 23 arms and 1,380 episodes.
- **Hidden-scope experiment.** It adds 240 dd episodes at three seeds and 800 at ten seeds.

**Arms.**
- **Positive:**
  - PEk: exact copies of k in {4, 16, 64} evaluation rows.
  - PJk: jittered copies, of 16 rows at sigma in {0.01, 0.05, 0.20} column sd and of 4 and 64 rows at sigma 0.05.
  - PL4h and PL4f: a scope that includes half or all of the evaluation rows.
  - PL5a: a proxy at univariate AUROC a in {0.80, 0.95, 0.995}.
  - PL5m: a proxy matched to the strongest predictor.
- **Negative:**
  - A: clean.
  - NB16 and NJ16: exact and jittered self-duplicates of 16 fit rows.
  - NR16 and NRJ16: exact and jittered copies of 16 reserve rows.
  - NL4R: a scope of fit plus reserve rows.
  - NL5S: a noisy copy of the strongest predictor.
  - NL5N: a noise predictor.
  - NL5X: the strongest predictor replaced by its noisy copy.

**Collection and processing.**
- Only numeric predictors are kept, and rows with a missing value are dropped; none were dropped in these tables.
- Deduplication removes duplicate feature rows and keeps the first.
- Each table is split 75/25 (stratified) into training and evaluation ids. 10% of the training ids form a reserve that is never fitted or evaluated.
- Each seed controls the split, the reserve draw, the construction draws, the audited estimator's random state and the placebo draws.

**Ground truth.** An episode is positive if training and evaluation share source records or, for L4 and L5, if the planted flow is present. Truth is computed from serialized source identifiers that are kept outside every method's input. Natural raw-table duplicates are labeled negative.

**Uses and misuse.** The suite is suitable for comparing types of audit evidence and audit tools on planted leakage. It is not evidence about how common natural leakage is. It is not confirmation data either, because all ten families were inspected during development.

**Distribution and licenses.**
- The source tables are not redistributed; see `README.md` and `DATA_HASHES.sha256`.
- UCI 159, 329 and 563 are CC BY 4.0. The OpenML tables are listed with licence "Public" in the OpenML API.
- The code and derived episode data will be released under open licenses upon acceptance.

**Maintenance.** The authors will maintain the suite through its public repository after review. Contact details are withheld for anonymous review.

**Known issues.**
1. **Distance-precision defect.** With more than 15 predictors, identical rows sit at a distance of about 1e-7, above the 1e-9 exact threshold. As a result the saved `exact_eval` undercounts in 183 episodes and `exact_loo` in 153, across five families. The corrected features use tuple equality.
2. **Natural duplicates.** Every raw family contains them.
3. **Hidden-scope result.** The supervised hidden-scope result depends mostly on added noise predictors that were selected with evaluation labels.

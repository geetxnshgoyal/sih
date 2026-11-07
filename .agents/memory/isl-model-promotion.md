---
name: ISL model promotion
description: Evidence required before replacing Setu's deployed recognition model.
---

Never promote a candidate model based on training or validation accuracy from clips that were also used by an initialization checkpoint. Compare against the deployed browser model on the same untouched signer-held-out evaluation.

**Why:** The ISL corpus is small and signer-diverse data is limited. Fine-tuning can reach near-perfect in-sample accuracy while offering no evidence of better recognition for a new signer.

**How to apply:** Keep unverified weights in an ignored candidate directory. Promote only after a fair held-out comparison, browser/Python parity, and confidence calibration checks pass.

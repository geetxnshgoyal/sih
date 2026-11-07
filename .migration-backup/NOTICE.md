# Notices and attributions

The MIT licence in `LICENSE` covers **the code in this repository**. It does not
and cannot cover the third-party corpora the model was trained on, which carry
their own terms. Those terms follow the data, not this project.

**Setu is free and always will be.** No fee, no subscription, no paid tier. That
is a product decision, and it also settles most of what follows: the corpora
below whose terms are limited to research or non-commercial use are satisfied by
a project that never charges. What survives regardless of price is
**attribution**, and that is not optional.

One caveat worth knowing rather than discovering: some licences define
"non-commercial" by who deploys the software rather than whether money changes
hands. A private hospital chain running a free tool can, on a strict reading,
count as commercial use. Nothing here is likely to attract that argument, but if
Setu is ever adopted by a paying institution it is worth a second look.

## Trained model weights

`app/public/model/` and `models/` are derived from the corpora below. A trained
model is a derivative of its training data, so redistributing the weights
inherits whatever the strictest of those licences requires.

| corpus | used for | licence | obligation |
|---|---|---|---|
| INCLUDE (AI4Bharat) | fine-tuning, 4,284 ISL clips | CC BY 4.0 | **attribution, below** |
| CISLR (Exploration-Lab) | fine-tuning, 612 ISL clips | AFL-3.0 | none for free use |
| Govt. of India ISL Dictionary | self-supervised pretraining | MIT | none |
| WLASL | pretraining, 21,083 ASL clips | C-UDA 1.0 | permits computational use |
| MS-ASL (Microsoft) | pretraining, 17,698 ASL clips | research use | satisfied while free |

If Setu ever stops being free, MS-ASL is the one to re-check first, and the
remedy is cheap: retrain the encoder on WLASL alone. C-UDA explicitly permits
commercial computational use and places no restriction on the resulting model,
and the measured cost of dropping MS-ASL is small.

## Attribution required by CC BY 4.0

> INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition.
> Advaith Sridhar, Rohith Gandhi Ganesan, Pratyush Kumar, Mitesh Khapra.
> ACM Multimedia 2020. Licensed CC BY 4.0.

## Runtime components

| | |
|---|---|
| MediaPipe (Google) | Apache 2.0 |
| TensorFlow.js | Apache 2.0 |
| React | MIT |
| NLLB-200 (Meta) | CC BY-NC 4.0, **non-commercial** |

NLLB is used **offline, once**, to generate the phrase translations that are
committed as JSON. It is not shipped and not called at runtime, and its
non-commercial term is satisfied by a project that never charges.

The translations still need replacing, for a better reason than licensing: they
are unreviewed machine output, and at least one known Hindi error is grammatical
nonsense. Human translation by native and Deaf signers is the fix, and it is on
the roadmap regardless of any licence.

## Not medical advice

Setu is a communication aid. It does not diagnose, treat, or make clinical
recommendations, and its sign recognition is unreliable enough that no clinical
decision should rest on it. See the accuracy figures in `ARCHITECTURE.md` §5.

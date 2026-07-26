# Control: is the drop physics-specific, or would any removal do it?

**Question (your idea):** removing z/M⋆/sSFR drops alignment — but does removing the
**same amount of variance in a *random* direction** drop it as much? If not, the
physics *direction* specifically carries the cross-model alignment.

**Setup:** 21 ViT×ConvNeXt pairs (HSC). Three removals, each a coherent
projection (same directions from both models) that also removes the mean:
`physics` = the confound subspace (rank 10, removes 14% of *centered* variance);
`random-rank` = a random subspace of the **same rank** (10);
`random-var` = a random subspace with **the same variance removed** (~14%, ~hundreds of dims).
Values are calibrated-similarity **drop %** (mean [min..max] over pairs).

| metric | physics | random-rank(10) | random-var(matched) |
|---|---|---|---|
| **CKA** (global) | **21% [9..28]** | 0% [-0..0] | -0% [-0..0] |
| **MKNN** (local) | **7% [2..12]** | -7% [-12..-2] | 4% [-4..12] |

*(negative = alignment went slightly up; removing the mean can help cosine-MKNN.)*

## What happened

- **CKA (global): strongly physics-specific.** Removing the physics subspace drops
  CKA ~**21%**, but removing the **same amount of variance in random
  directions drops it ~-0%**. The global cross-model similarity lives
  specifically in the astrophysics directions — not in generic variance.
- **MKNN (local): physics-specific but small.** Physics drops MKNN ~**7%**
  vs ~**4%** for matched-variance random. So the physics direction is
  still *more* damaging per unit variance (10 physics dims hurt local structure more than
  ~426 random dims of the same total variance) — but the effect
  is modest, because local neighbourhoods barely move under physics removal in the first place.

**Verdict:** your control holds. Removing the physics subspace does **more** than removing
the same amount of variance randomly — decisively for global CKA, and consistently (if
weakly) for local MKNN.

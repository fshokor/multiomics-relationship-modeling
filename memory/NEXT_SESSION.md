# NEXT SESSION

## Goal
nb04 — Joint DL relationship model for RNA-Protein (CITE-seq). Start with the
multimodal VAE originally scoped in NEXT_SESSION.md's nb04 plan, now applied
specifically to the CITE-seq track (RNA + ADT) using all features, not just
the 112 name-matched genes used for nb02's correlation analysis.

**This is NOT a prediction task.** The model is an analysis tool to learn
the shared/independent structure between RNA and protein, not to optimize
benchmark accuracy.

---

## Context: what's done, what this builds on

- **nb01_qc** — complete. Full data structure documented in DATA.md.
- **nb02_rna_protein** — complete. Statistical (linear) characterization of
  RNA-protein coupling on 112 matched genes. Key finding: lineage markers
  (CD3D, IL7R, MS4A1) strongly coupled; activation/checkpoint receptors
  (OX40, 4-1BB, CTLA4, LAG3, CD69) decoupled. Donor-consistent (r=0.758).
- **nb03_rna_atac** — complete but flagged with an open question (gene-activity
  sign-flip, parked — see SESSION_MEMORY.md). Not blocking nb04.
- **nb04 is now starting on the RNA-protein (CITE-seq) side first.** The
  RNA-ATAC (Multiome) VAE track comes later, after nb04 validates the
  approach on CITE-seq.

## Key design decision (already agreed)

Unlike nb02's correlation analysis, **the VAE uses the FULL feature sets**:
- RNA encoder input: all 13,953 GEX genes (not just the 112 matched ones)
- Protein encoder input: all 134 ADT proteins (not just the 112 matched ones)

No name-matching constraint — the model learns joint structure across all
available features in each modality, which is the whole point of using DL
instead of per-gene linear correlation. This was explicitly noted in
SESSION_MEMORY.md as the plan for nb04 and applies now.

---

## Architecture (from original NEXT_SESSION.md plan, adapt as needed)

```python
class MultimodalVAE(nn.Module):
    def __init__(self, rna_dim=13953, adt_dim=134, latent_dim=32):
        # Encoder 1: RNA -> latent (mean + logvar)
        # Encoder 2: ADT -> latent (mean + logvar)
        # Shared latent space
        # Decoder 1: latent -> RNA reconstruction
        # Decoder 2: latent -> ADT reconstruction
```

Scale: 90,261 cells is plenty for a small VAE (latent_dim=32, hidden=256).
Run on Colab GPU — should train in well under 30 minutes.

## What to extract from the trained model

- Latent representation per cell -> does it separate cell types cleanly
  (UMAP of latent space colored by `cell_type`)?
- Per-gene/per-protein reconstruction error -> features hard to reconstruct
  = independently regulated, candidate comparison against nb02's per-gene
  Pearson r list
- Cross-modal prediction: RNA encoder -> ADT decoder (RNA->protein
  prediction pathway through the shared latent space)
- Compare VAE cross-modal prediction quality against the simple linear
  baseline already computed in nb02 (per-gene Pearson r). If VAE doesn't
  outperform linear: relationship is linear, DL adds nothing for this axis.
  That is a valid, reportable finding either way.

## Suggested notebook steps

1. **Setup** — same `IN_COLAB` pattern as nb01-nb03, load CITE-seq, split GEX/ADT
2. **Prepare full-feature tensors** — RNA (13,953-d) and ADT (134-d) per cell,
   no name-matching needed this time. Decide on a sensible RNA dimensionality
   reduction or not (13,953 genes may be high-dimensional for a 256-hidden
   MLP encoder on Colab's free-tier GPU — test memory footprint early)
3. **Build the VAE** (rna_encoder, adt_encoder, shared latent, two decoders)
4. **Train** — track reconstruction loss per modality + KL term separately;
   plot per-epoch loss curve. Pause at this decision point before scaling up
   epochs, per the project's iterative/decision-point-driven convention.
5. **Latent space inspection** — UMAP/PCA of latent embeddings colored by
   cell_type, donor, batch -- check for batch effects vs real biology
6. **Cross-modal reconstruction quality** — per-protein reconstruction error,
   compare ranking against nb02's per-gene Pearson r for the 112 overlapping
   genes (does the VAE agree with which genes/proteins are "hard to predict"?)
7. **Linear vs VAE comparison** — does the VAE's cross-modal prediction beat
   simple per-gene linear regression (already implicit in nb02's r values)?

## Technical notes carried over

- Use `.X` (normalized), not raw counts — same rationale as nb02/nb03
  (avoids library-size confound, consistent representation across notebooks)
- `IN_COLAB` environment-detection pattern, `BASE_PATH` from `configs/paths.py`
- `src/` and `configs/` must be manually re-synced to Google Drive after any
  local edits (no auto-sync) — see Colab/Drive sync notes in SESSION_MEMORY.md
- If `ModuleNotFoundError: No module named 'src'` recurs despite files being
  on Drive: `drive.flush_and_unmount()`, remount with `force_remount=True`,
  restart runtime, re-run from top (known Drive FUSE caching issue)

## Do NOT do next session

- Do not redo nb01 or nb02 — both complete and locked in.
- Do not revisit GDSC/CCLE/ProCan data.
- Do not chase SOTA reconstruction performance — the VAE is an analysis tool.
- Do not pursue the promoter-proximal ATAC peak linkage yet (parked from
  this session, see SESSION_MEMORY.md "Open questions for later" — pick up
  after the CITE-seq VAE work is done, or whenever convenient).
- Do not work on the Genopole Shaker application narrative yet — separate
  task, deadline July 15, ~2 weeks out.

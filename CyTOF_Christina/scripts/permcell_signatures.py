"""PermCell signature sets for the 29-marker breast cancer cell-line panel.

Panel (core histones H3/H3.3/H4 excluded -- they were used for normalisation):
  Lineage/state (14): CD24 CD44 CD49f ER EZH2 EpCAM GATA3 KI67 KRT5 KRT8-18
                      Vimentin ZEB1 p53 pH2A.X
  Histone PTMs (15) : H2AK119ub H3K27ac H3K27me2 H3K27me3 H3K36me2 H3K36me3
                      H3K4me1 H3K4me3 H3K64ac H3K9ac H3K9me2 H3K9me3 H3S28p
                      H4K16ac H4K20me3

DESIGN NOTES -- these follow from how PermCell actually scores.

1. `sipsic_like_scores_v3` row-centres the matrix (`Xc = X - X.mean(axis=1)`) before scoring.
   Every score is therefore COMPOSITIONAL: "which markers dominate this cell", not "how much
   total signal the cell has". This is the same correction that was needed to see the chromatin
   result in the joint analysis, and it is why chromatin sets below are written as up-only where
   appropriate -- relative to the cell's own mean, an up-only set already is a composition score.

2. The null redraws `m` markers from the NON-set markers (`exclude_set=True`) keeping the same
   weight vector. With G=29 the pool is 29-m, so keep sets small: m <= 8 is comfortable
   (C(21,8) ~ 2e5 distinct draws), m > 14 is impossible. All sets below are 2-8 markers.

3. Row-centring assumes markers are on comparable scales. Run PermCell on a Z-SCORED layer
   (e.g. "norm_divide_z"), never on raw counts, or the centring is dominated by whichever
   channel happens to be brightest.

4. Signed sets ({"up": [...], "down": [...]}) give a directional contrast and are far more
   specific than up-only sets for lineage questions. Use `normalize_set_weights="l2"` so sets
   of different sizes are comparable.

CAVEAT ON ER: in this panel ER barely separates MCF7 (+0.65) from HCC1937 (+0.59), which is not
credible for a TNBC line and suggests background. ER is therefore given HALF weight in the
luminal signature and is never a sole driver.
"""

# ---------------------------------------------------------------------------------------------
# 1. LINEAGE / IDENTITY -- signed contrasts
# ---------------------------------------------------------------------------------------------
LINEAGE = {
    # Differentiated luminal. GATA3 is the strongest luminal discriminator in this panel
    # (+1.76 MCF7 vs TNBC mean); ER down-weighted per the caveat above.
    "Luminal": {
        "GATA3": 1.0, "KRT8-18": 1.0, "CD24": 1.0, "ER": 0.5,
        "KRT5": -1.0, "CD44": -1.0, "CD49f": -0.5,
    },
    # Basal-A: keratin-defined basal, CD44-low (the HCC70 phenotype).
    "Basal_A": {
        "KRT5": 1.0, "CD49f": 1.0,
        "GATA3": -1.0, "CD24": -0.5, "KRT8-18": -0.5,
    },
    # Basal-B / claudin-low: CD44-high mesenchymal-leaning (SUM149, MDA-MB-468).
    "Basal_B_claudin_low": {
        "CD44": 1.0, "Vimentin": 1.0, "CD49f": 0.5,
        "EpCAM": -1.0, "CD24": -1.0,
    },
    # Full EMT programme.
    "EMT": {
        "Vimentin": 1.0, "ZEB1": 1.0, "CD44": 0.5,
        "EpCAM": -1.0, "KRT8-18": -1.0, "CD24": -0.5,
    },
    # CD44+/CD24- stem/progenitor-like. This is the axis that isolates the minor basal-like
    # population inside MCF7 (~1.3% of cells); keep it separate from EMT, which it is not.
    "Stem_CD44p_CD24n": {
        "CD44": 1.0, "CD49f": 1.0,
        "CD24": -1.0, "GATA3": -1.0,
    },
    # Epithelial adhesion, the counterpart to EMT.
    "Epithelial_adhesion": {
        "EpCAM": 1.0, "KRT8-18": 1.0, "CD24": 0.5,
        "Vimentin": -1.0, "ZEB1": -1.0,
    },
}

# ---------------------------------------------------------------------------------------------
# 2. CELL CYCLE / STRESS
# ---------------------------------------------------------------------------------------------
CELL_CYCLE = {
    # Cycling cells. H3S28p is mitosis-specific, KI67 is cycle-wide.
    "Proliferation": {"KI67": 1.0, "H3S28p": 1.0},
    # Mitosis specifically -- H3S28p up while KI67 is only moderately informative alone.
    "Mitotic": {"H3S28p": 1.0, "KI67": 0.5, "H4K20me3": -0.5},
    # Replication stress / DNA damage OUTSIDE mitosis. The H3S28p-down term is what separates
    # this from "Mitotic"; it reproduces the joint-model A7 state (pH2A.X high, H3S28p low).
    "DNA_damage_response": {"pH2A.X": 1.0, "p53": 1.0, "H3S28p": -1.0},
    # Quiescent / non-cycling.
    "Quiescent": {"KI67": -1.0, "H3S28p": -1.0, "H4K20me3": 1.0},
}

# ---------------------------------------------------------------------------------------------
# 3. CHROMATIN STATE -- the panel's real strength (15 PTMs)
# ---------------------------------------------------------------------------------------------
CHROMATIN = {
    # Facultative repression: PRC2 (H3K27me2/3, EZH2) + PRC1 (H2AK119ub).
    "Polycomb_repression": {
        "H3K27me3": 1.0, "H3K27me2": 1.0, "H2AK119ub": 1.0, "EZH2": 1.0,
    },
    # Constitutive heterochromatin.
    "Constitutive_heterochromatin": {
        "H3K9me3": 1.0, "H4K20me3": 1.0, "H3K9me2": 0.5,
    },
    # Active promoters.
    "Active_promoter": {"H3K4me3": 1.0, "H3K9ac": 1.0, "H3K27ac": 1.0},
    # Primed / active enhancers. H3K4me1 without H3K4me3 is the enhancer signature.
    "Enhancer_primed": {"H3K4me1": 1.0, "H4K16ac": 1.0, "H3K4me3": -1.0},
    # Global acetylation / open chromatin, contrasted against repressive methylation.
    "Open_chromatin": {
        "H3K27ac": 1.0, "H3K9ac": 1.0, "H3K64ac": 1.0, "H4K16ac": 1.0,
        "H3K9me2": -1.0, "H3K9me3": -1.0, "H4K20me3": -1.0,
    },
    # Transcriptional elongation.
    "Elongation": {"H3K36me2": 1.0, "H3K36me3": 1.0},
    # Bivalent / poised promoters -- the classic plasticity signature.
    "Bivalent_poised": {"H3K4me3": 1.0, "H3K27me3": 1.0},
    # THE REPRESSIVE-MODE SWITCH found in this dataset: H3K9me2 gained while Polycomb is lost.
    # H3K9me2 rose monotonically across all deciles of basal-stem character in MCF7 and was
    # positive in all 5 lines; H3K27me3 was negative in all 5. This set encodes that contrast
    # directly and is the most dataset-specific signature here.
    "K9me2_over_Polycomb": {
        "H3K9me2": 1.0,
        "H3K27me3": -1.0, "H3K27me2": -0.5, "H2AK119ub": -0.5,
    },
    # Promoter- vs enhancer-weighted H3K4 balance.
    "H3K4_promoter_vs_enhancer": {"H3K4me3": 1.0, "H3K4me1": -1.0},
}

# ---------------------------------------------------------------------------------------------
# 4. POST-HOC COMPOSITE PHENOTYPES -- **NOT** part of the discovery set
#
# These are deliberately EXCLUDED from `SIGNATURES`. Two reasons:
#   * Circularity. `BasalStem_K9me2_slow` was constructed FROM the MCF7 result in this dataset.
#     Feeding it into archetype discovery would let a finding define the space used to rediscover
#     itself.
#   * Geometry. Each is a linear combination of the base sets above, so including them adds
#     strongly correlated dimensions that distort archetype vertices without adding information.
#
# Use them only AFTER archetypes are fitted, to score/label the result. The better route is to
# derive composites empirically -- read off which base signatures co-occur in each archetype.
# ---------------------------------------------------------------------------------------------
COMBINED_POSTHOC = {
    # The MCF7 minor population, encoded as a single signature: basal/stem surface phenotype
    # WITH the repressive-mode switch and slow cycling.
    "BasalStem_K9me2_slow": {
        "CD44": 1.0, "CD49f": 1.0, "H3K9me2": 1.0,
        "CD24": -1.0, "GATA3": -1.0, "H3K27me3": -1.0, "KI67": -0.5,
    },
    # De-differentiated / plastic: stem surface + bivalent chromatin.
    "Plastic_bivalent_stem": {
        "CD44": 1.0, "H3K4me3": 1.0, "H3K27me3": 1.0,
        "CD24": -1.0, "GATA3": -1.0,
    },
    # Proliferative luminal -- the dominant MCF7 state, as a positive control.
    "Luminal_proliferative": {
        "GATA3": 1.0, "KRT8-18": 1.0, "KI67": 1.0, "CD24": 1.0,
        "CD44": -1.0, "KRT5": -1.0,
    },
}

# ---------------------------------------------------------------------------------------------
# Assembled sets
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# 5. BUILDING THE DISCOVERY SET
#
# Measured on 230,352 pooled cells (SUM149 + MCF7 + HCC70), several sets are near-mirror images
# of each other BY CONSTRUCTION. Near-collinear features distort archetype geometry -- the simplex
# stretches along the duplicated direction, exactly the failure mode the global chromatin axis
# caused in marker space. Observed before pruning:
#
#     EMT            <-> Epithelial_adhesion   r = -0.97   (literal negatives)
#     Mitotic        <-> Quiescent             r = -0.93
#     Proliferation  <-> Mitotic               r = +0.89
#     Proliferation  <-> Quiescent             r = -0.81
#
# Effective dimensionality was only 7.5 of 19. Pruning keeps the positively-defined, biologically
# primary member of each mirror pair; the dropped member is exactly recoverable post-hoc as its
# negative, so no information is lost.
#
# `K9me2_over_Polycomb` is ALSO demoted, for the circularity reason rather than redundancy: it was
# written after seeing the H3K9me2 result in this dataset. It is reconstructable post-hoc from
# `Polycomb_repression` and `Constitutive_heterochromatin`, both of which are retained.
#
# Result: 15 signatures, max pairwise |r| = 0.83 (Luminal vs Basal_A -- a real biological
# opposition, and both are scientifically primary, so both are kept), effective dims 7.5 of 15.
# ---------------------------------------------------------------------------------------------
_MIRROR_OR_DERIVED = ["Epithelial_adhesion", "Quiescent", "Mitotic", "K9me2_over_Polycomb"]

#: Discovery set -- independent, mechanism-level signatures. Archetypes are fitted on THIS.
SIGNATURES = {k: v for k, v in {**LINEAGE, **CELL_CYCLE, **CHROMATIN}.items()
              if k not in _MIRROR_OR_DERIVED}

#: Dropped as mirror images / dataset-derived. Score these post-hoc only.
MIRRORED_POSTHOC = {k: v for k, v in {**LINEAGE, **CELL_CYCLE, **CHROMATIN}.items()
                    if k in _MIRROR_OR_DERIVED}

#: Everything, for post-hoc scoring and labelling only -- never for archetype discovery.
SIGNATURES_WITH_POSTHOC = {**SIGNATURES, **MIRRORED_POSTHOC, **COMBINED_POSTHOC}

# Recommended call. Note source_layer must be a Z-SCORED layer (see design note 3).
RECOMMENDED_PARAMS = dict(
    source_layer="norm_divide_z",
    positions_key="X_umap",
    n_perm=2000,
    seed=0,
    exclude_set=True,
    two_sided=True,            # these are contrasts; depletion is as meaningful as enrichment
    abs_variant=True,
    normalize_set_weights="l2",  # sets differ in size -- required for comparability
    use_sparse_W=False,
    prefer_permutation=True,     # keep True: the enumeration branch has a latent NameError
    perm_batch=1024,
    k=64,
    bandwidth=-1,
)


def validate(var_names, signatures=None, verbose=True):
    """Check every signature marker exists in the panel and that set sizes are safe.

    Returns (ok, problems).
    """
    signatures = SIGNATURES if signatures is None else signatures
    panel = set(map(str, var_names))
    G = len(panel)
    problems = []
    for name, payload in signatures.items():
        markers = list(payload.keys()) if isinstance(payload, dict) and not (
            set(payload.keys()) & {"up", "down"}
        ) else (list(payload.get("up", [])) + list(payload.get("down", [])))
        missing = [m for m in markers if m not in panel]
        m_size = len(markers)
        if missing:
            problems.append(f"{name}: markers not in panel -> {missing}")
        if m_size > G - m_size:
            problems.append(f"{name}: size {m_size} leaves pool {G-m_size} < m (null impossible)")
        if verbose:
            flag = "  <-- FIX" if missing or m_size > G - m_size else ""
            print(f"  {name:<28} m={m_size:<3} pool={G-m_size:<3}{flag}")
    if verbose:
        print(f"\n{len(signatures)} signatures, {len(problems)} problem(s)")
        for p in problems:
            print(f"  ! {p}")
    return (len(problems) == 0), problems

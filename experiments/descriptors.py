"""RDKit descriptor computation for augmenting HANNA's ChemBERTa representation.

HANNA represents each mixture component as a 384-dim ChemBERTa-2 [CLS] embedding.
This module computes physico-chemical RDKit descriptors per molecule (from SMILES)
so they can be *concatenated* onto that embedding, growing the per-component
feature vector from 384 to 384+D.

Why these descriptors? They are the same molecular properties classical
group-contribution models (UNIFAC) lean on to explain *why* a mixture deviates
from ideality:
  - polarity            -> TPSA, MolLogP
  - hydrogen bonding    -> NumHDonors, NumHAcceptors, NHOHCount, NOCount
  - size / shape        -> MolWt, HeavyAtomCount, ring counts, FractionCSP3
  - polarizability      -> MolMR (molar refractivity), LabuteASA
Activity coefficients are driven by exactly these effects, so the hypothesis is
that handing them to the network explicitly (rather than hoping ChemBERTa has
encoded them) could help — especially the cosine-distance gate, which now sees
an interpretable similarity axis.

Two sets are provided:
  - "curated":  18 interpretable, thermodynamically-motivated descriptors.
  - "full":     every descriptor in rdkit Descriptors.descList (~210).

Sanitisation. RDKit descriptors can be NaN/inf for some molecules (partial-charge
descriptors on odd valence states; Ipc information content overflowing). We
(1) replace NaN/inf with the per-column median computed on TRAIN components only,
and (2) winsorise each column to its [0.5, 99.5] train percentile so a few
pathological molecules cannot dominate the downstream StandardScaler. The final
standardisation is left to the pipeline's CustomScaler, so descriptors are
normalised on exactly the same footing as the BERT dimensions.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")  # descriptor failures are handled explicitly below

# ── Descriptor sets ──────────────────────────────────────────────────────────

# Thermodynamically-motivated subset. Every name must exist in Descriptors.descList.
CURATED = [
    "MolWt",            # molecular size
    "HeavyAtomCount",   # size (atom count)
    "MolLogP",          # hydrophobicity / polarity (Crippen logP)
    "MolMR",            # molar refractivity ~ polarizability (Crippen)
    "TPSA",             # topological polar surface area
    "LabuteASA",        # accessible surface area
    "NumHDonors",       # H-bond donors
    "NumHAcceptors",    # H-bond acceptors
    "NHOHCount",        # N-H / O-H count (H-bonding)
    "NOCount",          # N + O count (H-bonding sites)
    "NumRotatableBonds",  # flexibility
    "FractionCSP3",     # sp3 fraction (saturation)
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "RingCount",
    "NumHeteroatoms",
    "NumValenceElectrons",
]


def get_descriptor_names(set_name: str) -> list[str]:
    """Return the ordered list of descriptor names for a set."""
    all_names = [name for name, _ in Descriptors.descList]
    if set_name == "curated":
        missing = [n for n in CURATED if n not in all_names]
        if missing:
            raise ValueError(f"Curated descriptors missing from this rdkit build: {missing}")
        return list(CURATED)
    if set_name == "full":
        return list(all_names)
    raise ValueError(f"Unknown descriptor set: {set_name!r} (expected 'curated' or 'full')")


def _descriptor_functions(names: list[str]):
    table = dict(Descriptors.descList)
    return [(n, table[n]) for n in names]


def compute_raw_descriptors(smiles_list: list[str], names: list[str]) -> np.ndarray:
    """Compute the raw [n_molecules, D] descriptor matrix. Failures become NaN."""
    funcs = _descriptor_functions(names)
    out = np.full((len(smiles_list), len(names)), np.nan, dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for j, (_, fn) in enumerate(funcs):
            try:
                out[i, j] = float(fn(mol))
            except Exception:
                out[i, j] = np.nan
    return out


class DescriptorSanitizer:
    """Fit per-column median + winsor bounds on TRAIN rows, apply to any rows.

    Leaves final standardisation to the downstream CustomScaler — this only makes
    descriptors finite and outlier-robust.
    """

    def __init__(self, lo_pct: float = 0.5, hi_pct: float = 99.5):
        self.lo_pct = lo_pct
        self.hi_pct = hi_pct
        self.median_: np.ndarray | None = None
        self.lo_: np.ndarray | None = None
        self.hi_: np.ndarray | None = None

    def fit(self, raw_train: np.ndarray) -> "DescriptorSanitizer":
        finite = np.where(np.isfinite(raw_train), raw_train, np.nan)
        self.median_ = np.nanmedian(finite, axis=0)
        # columns that are entirely NaN -> median 0
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        filled = np.where(np.isfinite(finite), finite, self.median_)
        self.lo_ = np.nanpercentile(filled, self.lo_pct, axis=0)
        self.hi_ = np.nanpercentile(filled, self.hi_pct, axis=0)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self.median_ is None:
            raise RuntimeError("DescriptorSanitizer.fit must be called first")
        x = np.where(np.isfinite(raw), raw, self.median_)
        x = np.clip(x, self.lo_, self.hi_)
        return x.astype(np.float32)

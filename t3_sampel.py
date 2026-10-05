"""T3 v2 — ambil 100 komentar berstrata tingkat kesepakatan (35 / 35 / 30) untuk validasi manual.

  python pipeline_v2/t3_sampel.py sample   -> sampel100_v2_LEMBAR_LABEL.xlsx (tanpa label otomatis, untuk penilai)
                                             pipeline_v2/out/t3_sampel_v2.csv (dengan label otomatis)
  python pipeline_v2/t3_sampel.py eval     -> baca label_manual dari sampel100_v2_KUNCI.xlsx, hitung metrik
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
OUTD = ROOT / "pipeline_v2" / "out"
SEED = 42
L = ["negative", "neutral", "positive"]
QUOTA = {3: 35, 2: 35, 1: 30}


def sample():
    df = pd.read_csv(ROOT / "dataset_labelled_v2.csv")
    parts = []
    for ag, n in QUOTA.items():
        g = df[df.agreement == ag]
        # seimbang platform di dalam tiap strata (dataset v2 sudah 50:50)
        for plat, x in g.groupby("platform"):
            k = n // 2 + (n % 2 if plat == "youtube" else 0)
            parts.append(x.sample(min(len(x), k), random_state=SEED))
    s = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    s.insert(0, "no", range(1, len(s) + 1))
    s.to_csv(OUTD / "t3_sampel_v2.csv", index=False)
    lembar = s[["no", "id", "platform", "event", "source_title", "text_raw"]].copy()
    lembar["label_manual"] = ""
    lembar["catatan"] = ""
    lembar.to_excel(ROOT / "sampel100_v2_LEMBAR_LABEL.xlsx", index=False)
    print(s.agreement.value_counts().to_dict(), s.platform.value_counts().to_dict())


def evaluate():
    k = pd.read_excel(ROOT / "sampel100_v2_KUNCI.xlsx")
    k = k[k.label_manual.isin(L)]
    out = {"n": len(k), "manual_dist": k.label_manual.value_counts().to_dict(),
           "platform": k.platform.value_counts().to_dict()}
    for name, c in [("IndoBERT", "label_indobert"), ("InSet", "label_inset"), ("VADER", "label_vader"),
                    ("Voting", "label_vote")]:
        p, r, f, _ = precision_recall_fscore_support(k.label_manual, k[c], labels=L, average="macro", zero_division=0)
        out[name] = dict(acc=accuracy_score(k.label_manual, k[c]), p=p, r=r, f1=f,
                         kappa=cohen_kappa_score(k.label_manual, k[c]))
    pc, rc, fc, _ = precision_recall_fscore_support(k.label_manual, k.label_vote, labels=L, zero_division=0)
    out["vote_per_kelas"] = {l: dict(p=a, r=b, f1=c) for l, a, b, c in zip(L, pc, rc, fc)}
    out["cm_vote"] = confusion_matrix(k.label_manual, k.label_vote, labels=L).tolist()
    k["benar"] = k.label_vote == k.label_manual
    out["acc_per_agreement"] = k.groupby("agreement").benar.mean().to_dict()
    out["acc_per_platform"] = k.groupby("platform").benar.mean().to_dict()
    pop = pd.read_csv(ROOT / "dataset_labelled_v2.csv").agreement.value_counts(normalize=True)
    out["acc_populasi_terbobot"] = sum(pop[a] * out["acc_per_agreement"][a] for a in pop.index)
    mis = k[k.label_vote != k.label_manual]
    out["n_tidak_sesuai"] = len(mis)
    if "kategori" in k.columns:
        out["kategori"] = mis.kategori.value_counts().to_dict()
    (OUTD / "t3_ringkasan.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    {"sample": sample, "eval": evaluate}[sys.argv[1]]()

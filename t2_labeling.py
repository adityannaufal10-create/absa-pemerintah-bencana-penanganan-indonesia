"""T2 v2 — pelabelan otomatis (IndoBERT, InSet, VADER+OPUS-MT) + majority voting + split 80/10/10.
Port langsung dari NLP_Labeling.ipynb (logika & ambang sama), dijalankan pada dataset_clean_v2.csv.
Output: dataset_labelled_v2.csv, pipeline_v2/out/t2_*.csv|json
"""
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parent.parent
OUTD = ROOT / "pipeline_v2" / "out"
SEED = 42
DEV = 0 if torch.cuda.is_available() else -1

df = pd.read_csv(ROOT / "dataset_clean_v2.csv")
df = df.dropna(subset=["text_light", "text_norm"]).reset_index(drop=True)
print(df.shape)

# ---- IndoBERT
clf = pipeline("text-classification", model="mdhugol/indonesia-bert-sentiment-classification", device=DEV)
MAP_BERT = {"LABEL_0": "positive", "LABEL_1": "neutral", "LABEL_2": "negative"}
tes = clf(["sangat bahagia hari ini", "dasar pejabat tidak becus"])
assert MAP_BERT[tes[0]["label"]] == "positive" and MAP_BERT[tes[1]["label"]] == "negative"
out = clf(df.text_light.tolist(), batch_size=32, truncation=True, max_length=512)
df["label_indobert"] = [MAP_BERT[o["label"]] for o in out]
df["score_indobert"] = [round(o["score"], 4) for o in out]

# ---- InSet
base = "https://raw.githubusercontent.com/fajri91/InSet/master/"
lex = {}
for f in ["positive.tsv", "negative.tsv"]:
    p = ROOT / "pipeline_v2" / "resources" / f"inset_{f}"
    if not p.exists():
        p.write_bytes(requests.get(base + f, timeout=60).content)
    d = pd.read_csv(p, sep="\t")
    for w, s in zip(d.word, d.weight):
        lex[w] = lex.get(w, 0) + s
MULTI = {w: s for w, s in lex.items() if " " in w}
SINGLE = {w: s for w, s in lex.items() if " " not in w}
NEGASI = {"tidak", "tak", "bukan", "belum", "jangan", "gak", "nggak", "ga", "enggak", "ndak", "kurang"}


def inset_score(text):
    t, score = f" {text} ", 0
    for w, s in MULTI.items():
        k = t.count(f" {w} ")
        if k:
            score += s * k
            t = t.replace(f" {w} ", " ")
    tok = t.split()
    for i, w in enumerate(tok):
        if w in SINGLE:
            s = SINGLE[w]
            if i > 0 and tok[i - 1] in NEGASI:
                s = -s
            score += s
    return score


df["score_inset"] = df.text_norm.map(inset_score)
df["label_inset"] = df.score_inset.map(lambda s: "positive" if s > 0 else "negative" if s < 0 else "neutral")

# ---- VADER + terjemahan OPUS-MT
MODEL_MT = "Helsinki-NLP/opus-mt-id-en"
tok_mt = AutoTokenizer.from_pretrained(MODEL_MT)
dev = "cuda" if DEV == 0 else "cpu"
model_mt = AutoModelForSeq2SeqLM.from_pretrained(MODEL_MT, use_safetensors=True).to(dev)
if dev == "cuda":
    model_mt = model_mt.half()
model_mt.eval()


def translate_batch(texts, batch_size=64, max_length=64):
    hasil = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i + batch_size]]
        enc = tok_mt(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(dev)
        with torch.no_grad():
            o = model_mt.generate(**enc, max_length=max_length, num_beams=1)
        hasil.extend(tok_mt.batch_decode(o, skip_special_tokens=True))
    return hasil


df["text_en"] = translate_batch(df.text_norm.tolist())
vader = SentimentIntensityAnalyzer()
df["score_vader"] = df.text_en.map(lambda t: vader.polarity_scores(str(t))["compound"])
df["label_vader"] = df.score_vader.map(lambda c: "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral")


# ---- voting + split
def vote(r):
    (lab, n), = Counter([r.label_indobert, r.label_inset, r.label_vader]).most_common(1)
    return (lab, n) if n >= 2 else (r.label_indobert, 1)


df[["label_vote", "agreement"]] = df.apply(lambda r: pd.Series(vote(r)), axis=1)
tr, tmp = train_test_split(df, test_size=0.2, stratify=df.label_vote, random_state=SEED)
va, te = train_test_split(tmp, test_size=0.5, stratify=tmp.label_vote, random_state=SEED)
df["split"] = "train"
df.loc[va.index, "split"] = "val"
df.loc[te.index, "split"] = "test"
df.to_csv(ROOT / "dataset_labelled_v2.csv", index=False)

# ---- statistik untuk laporan
M = ["label_indobert", "label_inset", "label_vader"]
L = ["negative", "neutral", "positive"]
summ = {"n": len(df), "split": df.split.value_counts().to_dict()}
summ["dist"] = {m: df[c].value_counts().reindex(L).fillna(0).astype(int).to_dict()
                for m, c in [("IndoBERT", "label_indobert"), ("InSet", "label_inset"), ("VADER", "label_vader"),
                             ("Voting", "label_vote")]}
summ["cohen"] = {f"{a}|{b}": round(cohen_kappa_score(df[a], df[b]), 3) for i, a in enumerate(M) for b in M[i + 1:]}
tab, _ = aggregate_raters(df[M].replace({"negative": 0, "neutral": 1, "positive": 2}).astype(int).values)
summ["fleiss"] = round(fleiss_kappa(tab), 3)
summ["agreement"] = df.agreement.value_counts().to_dict()
summ["platform_pct"] = (pd.crosstab(df.platform, df.label_vote, normalize="index") * 100).round(1).to_dict("index")
summ["event_pct"] = (pd.crosstab(df.event, df.label_vote, normalize="index") * 100).round(1).to_dict("index")
rows = []
for ev, g in df.groupby("event"):
    p = [cohen_kappa_score(g[a], g[b]) for i, a in enumerate(M) for b in M[i + 1:]]
    rows.append(dict(event=ev, n=len(g), avg_pairwise_kappa=round(np.nanmean(p), 3),
                     pct_agreement_3_3=round((g.agreement == 3).mean() * 100, 1)))
pd.DataFrame(rows).sort_values("n", ascending=False).to_csv(OUTD / "t2_kesepakatan_per_event.csv", index=False)
(OUTD / "t2_ringkasan.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
print(json.dumps(summ, indent=1, default=str))

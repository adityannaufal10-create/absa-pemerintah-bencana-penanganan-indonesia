"""T4 v2 — klasifikasi sentimen 5 model + analisis kesalahan pada dataset_labelled_v2.csv.
Port dari T4_Klasifikasi_Sentimen.ipynb (arsitektur, grid hyperparameter, dan seed sama); output ke t4_output_v2/.
"""
import matplotlib
matplotlib.use("Agg")
ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
OUT = __import__("os").path.join(ROOT, "t4_output_v2")
import os, time, random, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
if DEVICE == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} | VRAM {p.total_memory/1e9:.1f} GB")
else:
    print("PERINGATAN: tidak ada GPU. Model konvensional & BiLSTM tetap jalan, IndoBERTweet akan sangat lambat.")

# ---- pengaturan ----
BERT_BS = 32          # turunkan ke 16/8 kalau CUDA out of memory
BERT_EPOCHS = 4
BERT_MAXLEN = 128

LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
pd.set_option("display.max_colwidth", 150)
os.makedirs(OUT, exist_ok=True)

PATH = os.path.join(ROOT, "dataset_labelled_v2.csv")

df = pd.read_csv(PATH)
for c in ["text_ml", "text_norm", "text_light"]:
    df[c] = df[c].fillna("").astype(str)
df["y"] = df.label_vote.map(L2I)

tr = df[df.split == "train"].reset_index(drop=True)
va = df[df.split == "val"].reset_index(drop=True)
te = df[df.split == "test"].reset_index(drop=True)
print(df.shape, "| train/val/test:", len(tr), len(va), len(te))
print(pd.crosstab(df.split, df.label_vote, normalize="index").round(3))

from scipy.sparse import hstack

vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=60000)
Xtr = hstack([vec_w.fit_transform(tr.text_ml), vec_c.fit_transform(tr.text_ml)]).tocsr()
Xva = hstack([vec_w.transform(va.text_ml), vec_c.transform(va.text_ml)]).tocsr()
Xte = hstack([vec_w.transform(te.text_ml), vec_c.transform(te.text_ml)]).tocsr()
print("Dimensi fitur:", Xtr.shape)

def mf1(y, p): return f1_score(y, p, average="macro")

grids = {
    "NaiveBayes":  (lambda a: MultinomialNB(alpha=a), [0.01, 0.05, 0.1, 0.3, 1.0]),
    "LogReg":      (lambda c: LogisticRegression(C=c, class_weight="balanced", max_iter=3000), [0.5, 1, 2, 5, 10]),
    "LinearSVM":   (lambda c: LinearSVC(C=c, class_weight="balanced"), [0.05, 0.1, 0.3, 0.5, 1.0]),
}
PRED, INFO, conv_models = {}, {}, {}
for name, (make, params) in grids.items():
    t0 = time.time()
    scores = {p: mf1(va.y, make(p).fit(Xtr, tr.y).predict(Xva)) for p in params}
    best = max(scores, key=scores.get)
    m = make(best).fit(Xtr, tr.y)
    conv_models[name] = m
    PRED[name] = m.predict(Xte)
    INFO[name] = dict(param=best, val_macroF1=round(scores[best], 4), waktu_s=round(time.time() - t0, 1))
    print(f"{name:10s} best={best} | val macro-F1 {scores[best]:.4f} | test macro-F1 {mf1(te.y, PRED[name]):.4f}")

m = conv_models["LogReg"]
fn = np.array(list(vec_w.get_feature_names_out()) + ["[c]" + f for f in vec_c.get_feature_names_out()])
nw = len(vec_w.vocabulary_)
top = {}
for k, lab in enumerate(LABELS):
    coef = m.coef_[k][:nw]                     # hanya fitur kata agar mudah dibaca
    top[lab] = fn[:nw][np.argsort(coef)[::-1][:20]]
top_words = pd.DataFrame(top)
top_words.to_csv(OUT + "/t4_top_kata_logreg.csv", index=False)


MAX_LEN_LSTM = 64
cnt = Counter(w for t in tr.text_norm for w in t.split())
vocab = {w: i + 2 for i, (w, c) in enumerate(cnt.most_common()) if c >= 2}   # 0=PAD, 1=UNK
print("Vocab:", len(vocab) + 2)

def encode(texts):
    out = np.zeros((len(texts), MAX_LEN_LSTM), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in t.split()][:MAX_LEN_LSTM]
        out[i, :len(ids)] = ids
    return torch.tensor(out)

class BiLSTM(nn.Module):
    def __init__(self, V, E=128, H=128, n_cls=3, drop=0.4):
        super().__init__()
        self.emb = nn.Embedding(V, E, padding_idx=0)
        self.lstm = nn.LSTM(E, H, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(2 * H, n_cls)
    def forward(self, x):
        h, _ = self.lstm(self.drop(self.emb(x)))
        h = h.masked_fill((x == 0).unsqueeze(-1), -1e4)   # abaikan padding saat pooling
        return self.fc(self.drop(h.max(1).values))

cw = torch.tensor(compute_class_weight("balanced", classes=np.arange(3), y=tr.y), dtype=torch.float).to(DEVICE)

def run_lstm(epochs=15, patience=3, bs=64, lr=2e-3):
    set_seed()
    dl_tr = DataLoader(TensorDataset(encode(tr.text_norm), torch.tensor(tr.y.values)), batch_size=bs, shuffle=True)
    Xv, Xt = encode(va.text_norm).to(DEVICE), encode(te.text_norm).to(DEVICE)
    model = BiLSTM(len(vocab) + 2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(weight=cw)
    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(epochs):
        model.train(); tot = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item()
        model.eval()
        with torch.no_grad(): pv = model(Xv).argmax(1).cpu().numpy()
        f = mf1(va.y, pv); hist.append((ep + 1, tot / len(dl_tr), f))
        print(f"epoch {ep+1:2d} | loss {tot/len(dl_tr):.4f} | val macro-F1 {f:.4f}")
        if f > best: best, bad, best_state = f, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: print("early stop"); break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): pt = model(Xt).argmax(1).cpu().numpy()
    return pt, best, hist

t0 = time.time()
PRED["BiLSTM"], v, hist_lstm = run_lstm()
INFO["BiLSTM"] = dict(param="E128-H128-drop0.4", val_macroF1=round(v, 4), waktu_s=round(time.time() - t0, 1))
print("BiLSTM test macro-F1:", round(mf1(te.y, PRED["BiLSTM"]), 4))

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

BERT_NAME = "indolem/indobertweet-base-uncased"

def prep_tweet(t):                                   # format yang dipakai saat pretraining IndoBERTweet
    t = t.lower()
    t = re.sub(r"@\w+", "@USER", t)
    t = re.sub(r"http\S+|www\.\S+", "HTTPURL", t)
    return t

def run_bert(model_name=BERT_NAME, epochs=BERT_EPOCHS, bs=BERT_BS, lr=2e-5, max_len=BERT_MAXLEN):
    set_seed()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(DEVICE)
    def enc(d):
        e = tok([prep_tweet(t) for t in d.text_light], truncation=True, max_length=max_len,
                padding="max_length", return_tensors="pt")
        return TensorDataset(e["input_ids"], e["attention_mask"], torch.tensor(d.y.values))
    dl_tr = DataLoader(enc(tr), batch_size=bs, shuffle=True)
    dl_va, dl_te = DataLoader(enc(va), batch_size=64), DataLoader(enc(te), batch_size=64)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(dl_tr)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    lossf = nn.CrossEntropyLoss(weight=cw)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    def predict(dl):
        model.eval(); P, PR = [], []
        with torch.no_grad():
            for ids, am, _ in dl:
                with torch.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                    lg = model(input_ids=ids.to(DEVICE), attention_mask=am.to(DEVICE)).logits
                PR.append(torch.softmax(lg.float(), -1).cpu().numpy()); P.append(lg.argmax(1).cpu().numpy())
        return np.concatenate(P), np.concatenate(PR)

    best, best_state, hist = -1, None, []
    for ep in range(epochs):
        model.train(); tot = 0
        for ids, am, yb in dl_tr:
            opt.zero_grad()
            with torch.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                lg = model(input_ids=ids.to(DEVICE), attention_mask=am.to(DEVICE)).logits
            loss = lossf(lg.float(), yb.to(DEVICE))
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); tot += loss.item()
        pv, _ = predict(dl_va); f = mf1(va.y, pv); hist.append((ep + 1, tot / len(dl_tr), f))
        print(f"epoch {ep+1} | loss {tot/len(dl_tr):.4f} | val macro-F1 {f:.4f}")
        if f > best: best, best_state = f, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pt, prob = predict(dl_te)
    return pt, prob, best, hist

t0 = time.time()
PRED["IndoBERTweet"], PROB_BERT, v, hist_bert = run_bert()
INFO["IndoBERTweet"] = dict(param="lr2e-5, ep4, len128", val_macroF1=round(v, 4), waktu_s=round(time.time() - t0, 1))
print("IndoBERTweet test macro-F1:", round(mf1(te.y, PRED["IndoBERTweet"]), 4))

MODELS = ["NaiveBayes", "LogReg", "LinearSVM", "BiLSTM", "IndoBERTweet"]
JENIS = {"NaiveBayes": "Konvensional", "LogReg": "Konvensional", "LinearSVM": "Konvensional",
         "BiLSTM": "Deep learning", "IndoBERTweet": "Deep learning"}
rows = []
for m in MODELS:
    p = PRED[m]
    pr, rc, f, _ = precision_recall_fscore_support(te.y, p, labels=[0, 1, 2], zero_division=0)
    rows.append(dict(model=m, jenis=JENIS[m], accuracy=accuracy_score(te.y, p), macro_F1=mf1(te.y, p),
                     weighted_F1=f1_score(te.y, p, average="weighted"),
                     F1_negative=f[0], F1_neutral=f[1], F1_positive=f[2], **INFO[m]))
hasil = pd.DataFrame(rows).round(4).sort_values("macro_F1", ascending=False)
hasil.to_csv(OUT + "/t4_hasil_model.csv", index=False)
BEST = hasil.iloc[0].model
print("Model terbaik (macro-F1):", BEST)


ax = hasil.set_index("model")[["accuracy", "macro_F1", "F1_negative", "F1_neutral", "F1_positive"]].plot(
    kind="bar", figsize=(10, 5), rot=0)
ax.set_ylim(0, 1); ax.set_title("Perbandingan 5 model — test set"); ax.legend(ncol=5, loc="upper center", fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "/t4_perbandingan_model.png", dpi=300); plt.close()

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, m in zip(axes, MODELS):
    cm = confusion_matrix(te.y, PRED[m], labels=[0, 1, 2])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, cbar=False, ax=ax)
    ax.set_title(f"{m}\nmacro-F1 {mf1(te.y, PRED[m]):.3f}"); ax.set_xlabel("prediksi"); ax.set_ylabel("label_vote")
plt.tight_layout(); plt.savefig(OUT + "/t4_confusion_matrix_5model.png", dpi=300); plt.close()

for m in MODELS:
    print(f"===== {m} ====="); print(classification_report(te.y, PRED[m], target_names=LABELS, digits=3, zero_division=0))

r = []
for ag in [3, 2, 1]:
    idx = te.agreement == ag
    r.append({"agreement": ag, "n": int(idx.sum()),
              **{m: round(mf1(te.y[idx], PRED[m][idx]), 3) for m in MODELS}})
by_agree = pd.DataFrame(r); by_agree.to_csv(OUT + "/t4_f1_per_agreement.csv", index=False)


def slice_eval(col):
    out = []
    for k, g in te.groupby(col):
        i = g.index
        out.append({col: k, "n": len(g), "macro_F1": round(mf1(te.y[i], PRED[BEST][i]), 3),
                    "accuracy": round(accuracy_score(te.y[i], PRED[BEST][i]), 3),
                    "catatan": "n<100, hati-hati" if len(g) < 100 else ""})
    return pd.DataFrame(out).sort_values("n", ascending=False)

per_platform, per_event = slice_eval("platform"), slice_eval("event")
per_platform.to_csv(OUT + "/t4_per_platform.csv", index=False); per_event.to_csv(OUT + "/t4_per_event.csv", index=False)
print(per_platform); print(per_event)

circ = pd.DataFrame({m: {lab: round(accuracy_score(te[lab].map(L2I), PRED[m]), 3)
                         for lab in ["label_vote", "label_indobert", "label_inset", "label_vader"]} for m in MODELS}).T
circ.to_csv(OUT + "/t4_kemiripan_dengan_labeler.csv")

N_ERR = 60
NEGASI = {"tidak", "tak", "bukan", "belum", "jangan", "enggak", "nggak", "gak", "ga", "kurang"}
te_err = te.copy()
te_err["pred_best"] = [LABELS[i] for i in PRED[BEST]]
te_err["n_model_salah"] = sum((PRED[m] != te.y.values).astype(int) for m in MODELS)
for m in MODELS: te_err[f"pred_{m}"] = [LABELS[i] for i in PRED[m]]

def hint(r):
    h, w = [], set(r.text_norm.split())
    if r.agreement == 1: h.append("label T2 ragu (3 metode beda)")
    if r.pred_best == r.label_indobert and r.pred_best != r.label_vote: h.append("prediksi = IndoBERT, label kalah voting")
    if w & NEGASI: h.append("ada negasi")
    if r.n_tokens <= 4: h.append("teks sangat pendek")
    if re.search(r"semoga|aamiin|amin|doa|ya allah", r.text_norm): h.append("doa/harapan")
    if re.search(r"😂|🤣|wkwk|haha|hebat ya|mantap ya", r.text_raw.lower()): h.append("kemungkinan sarkasme")
    if r.is_reply: h.append("balasan komentar lain")
    return "; ".join(h)

salah = te_err[te_err.pred_best != te_err.label_vote].copy()
print(f"Total salah {BEST}: {len(salah)} dari {len(te)}")
print(pd.crosstab(salah.label_vote, salah.pred_best))

pairs = salah.groupby(["label_vote", "pred_best"])
per = int(np.ceil(N_ERR / pairs.ngroups))
sampel = pd.concat([g.sample(min(len(g), per), random_state=SEED) for _, g in pairs])
if len(sampel) < N_ERR:
    sisa = salah.drop(sampel.index); sampel = pd.concat([sampel, sisa.sample(min(len(sisa), N_ERR - len(sampel)), random_state=SEED)])
sampel["hint_otomatis"] = sampel.apply(hint, axis=1)
sampel["kategori_error"] = ""; sampel["catatan"] = ""
cols = ["id", "platform", "event", "source_title", "text_raw", "label_vote", "pred_best", "n_model_salah",
        "label_indobert", "label_inset", "label_vader", "agreement"] + [f"pred_{m}" for m in MODELS] + \
       ["hint_otomatis", "kategori_error", "catatan"]
sampel = sampel[cols].sort_values(["label_vote", "pred_best"])
sampel.to_csv(OUT + "/t4_analisis_kesalahan.csv", index=False)
print("Sampel analisis kesalahan:", len(sampel))


susah = te_err[te_err.n_model_salah == 5][["id", "text_raw", "label_vote", "pred_best", "label_indobert", "label_inset", "label_vader", "agreement"]]
print("Salah di 5 model:", len(susah), "| proporsi dengan agreement=1:", round((susah.agreement == 1).mean(), 3))
susah.to_csv(OUT + "/t4_salah_semua_model.csv", index=False)


pred_out = te[["id", "platform", "event", "text_raw", "label_vote", "agreement"]].copy()
for m in MODELS: pred_out[f"pred_{m}"] = [LABELS[i] for i in PRED[m]]
pred_out.to_csv(OUT + "/t4_prediksi_test.csv", index=False)
print(sorted(os.listdir(OUT)))

# ---- tambahan v2: metrik per kelas, confusion matrix model terbaik, INFO waktu
import json
per_kelas = {}
for m in MODELS:
    pr, rc, f, _ = precision_recall_fscore_support(te.y, PRED[m], labels=[0, 1, 2], zero_division=0)
    pr_m, rc_m, f_m, _ = precision_recall_fscore_support(te.y, PRED[m], average="macro", zero_division=0)
    per_kelas[m] = dict(precision_macro=pr_m, recall_macro=rc_m, f1_macro=f_m, accuracy=accuracy_score(te.y, PRED[m]),
                        f1=dict(zip(LABELS, f)), recall=dict(zip(LABELS, rc)), precision=dict(zip(LABELS, pr)),
                        cm=confusion_matrix(te.y, PRED[m], labels=[0, 1, 2]).tolist(), **INFO[m])
per_kelas["_meta"] = dict(best=BEST, n_test=len(te), n_train=len(tr), n_val=len(va), n_fitur=int(Xtr.shape[1]),
                          vocab_lstm=len(vocab) + 2, salah_best=int((PRED[BEST] != te.y.values).sum()),
                          salah_semua=int((te_err.n_model_salah == 5).sum()),
                          salah_semua_pct_agree1=float((te_err[te_err.n_model_salah == 5].agreement == 1).mean()))
open(OUT + "/t4_ringkasan.json", "w").write(json.dumps(per_kelas, indent=1, default=float))
print(json.dumps(per_kelas["_meta"], indent=1))

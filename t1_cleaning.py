"""T1 v2 — pembersihan data + pengambilan 2.500 komentar (1.250 YouTube + 1.250 Instagram).

Input : scraping/data/youtube_raw_v2.csv, scraping/data/instagram_raw_v2.csv
        (semua komentar dalam periode + jumlah followers/subscriber komentator)
Seleksi: prioritas komentar dari akun followers/subscriber > 1.000; bila per platform kurang dari 1.250,
         sisanya diisi sampel acak berstrata peristiwa dari komentar lain (followers <= 1.000 / tidak terbaca).
Output: dataset_clean_v2.csv  (skema sama dengan dataset_clean_v1.csv + author_followers)
        pipeline_v2/out/t1_ringkasan.json (angka untuk laporan)

Tiga versi teks (sama seperti v1):
  text_light : huruf kecil, emoji & URL dibuang, mention -> @user          (transformer)
  text_norm  : text_light tanpa mention/angka/tanda baca + normalisasi slang (leksikon, terjemahan, BiLSTM)
  text_ml    : text_norm tanpa stopword + stemming Sastrawi                 (TF-IDF)
Kamus slang & stopword diturunkan dari pasangan teks dataset v1 (resources/), dengan perbaikan
pemetaan yang merusak makna di v1 (mis. "mbg" -> "mbak").
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import emoji
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "pipeline_v2" / "resources"
OUTD = ROOT / "pipeline_v2" / "out"
OUTD.mkdir(parents=True, exist_ok=True)
SEED = 42
TARGET = 1250
PERIOD = ("2026-01-01", "2026-09-30T23:59:59Z")

SLANG = json.loads((RES / "slang_map.json").read_text(encoding="utf-8"))
STOP = set((RES / "stopwords.txt").read_text(encoding="utf-8").split())
stemmer = StemmerFactory().create_stemmer()

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MENTION = re.compile(r"@[\w.]+")
RE_SPAM = re.compile(
    r"slot|gacor|judi|togel|maxwin|deposit|\bwd\b|link di bio|cek bio|klik link|dm (aja|ya|kak)|"
    r"wa\.me|whatsapp|\b08\d{8,}|follow (balik|back)|promo|endorse|jual|diskon|pinjol|investasi", re.I)
RE_DOA = re.compile(
    r"^(\W*)(semoga|smg|aamiin|amin|amiin|ya allah|innalillahi|turut berduka|berduka|doa|tetap semangat|"
    r"yang sabar|sabar ya|lekas pulih|cepat pulih|get well|pray for|prayfor|al fatihah|alfatihah)", re.I)
RE_STANCE = re.compile(
    r"pemerintah|presiden|prabowo|gibran|bnpb|bpbd|menteri|gubernur|bupati|pejabat|negara|dpr|anggaran|"
    r"bantuan|lambat|cepat tanggap|terima kasih|makasih|korup|tolong|harus|kapan|mana|kok|kenapa|"
    r"tidak|belum|enggak|gak|ga\b|kurang|salut|mantap|hebat|bagus|becus", re.I)


def text_light(t):
    t = str(t).lower().replace("​", " ")
    t = RE_URL.sub(" ", t)
    t = emoji.replace_emoji(t, " ")
    t = RE_MENTION.sub(" @user ", t)
    return re.sub(r"\s+", " ", t).strip()


def text_norm(light):
    t = light.replace("@user", " ")
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\w\s]|_", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)          # "pakkk" -> "pak", "eeee" -> "e"
    toks = [SLANG.get(w, w) for w in t.split()]
    return " ".join(" ".join(toks).split())


@lru_cache(maxsize=None)
def stem(w):
    return stemmer.stem(w)


def text_ml(norm):
    toks = [stem(w) for w in norm.split() if w not in STOP]
    return " ".join(w for w in toks if len(w) > 1 and w not in STOP)


def is_doa(raw, norm, n):
    return bool(RE_DOA.search(norm)) and n <= 15 and not RE_STANCE.search(norm)


def main():
    yt = pd.read_csv(ROOT / "scraping/data/youtube_raw_v2.csv", dtype={"comment_id": str, "parent_id": str})
    ig = pd.read_csv(ROOT / "scraping/data/instagram_raw_v2.csv", dtype={"comment_id": str, "parent_id": str})
    extra = ROOT / "scraping/data/instagram_extra_uts_clean.csv"   # sumber IG tambahan (uts_instagram_clean.csv, followers tidak diketahui)
    ig_x = pd.read_csv(extra, dtype={"comment_id": str, "parent_id": str}) if extra.exists() else ig.iloc[:0]
    raw = pd.concat([yt, ig, ig_x], ignore_index=True)
    summ = {"raw": raw.platform.value_counts().to_dict()}

    raw["text_raw"] = raw.text_raw.fillna("").astype(str)
    raw["text_light"] = raw.text_raw.map(text_light)
    raw["text_norm"] = raw.text_light.map(text_norm)
    raw["n_tokens"] = raw.text_norm.str.split().str.len().fillna(0).astype(int)
    raw["in_period"] = raw.published_at.between(*PERIOD)
    raw["is_spam"] = raw.text_raw.str.contains(RE_SPAM) | (raw.text_norm.str.len() == 0)
    raw["is_doa"] = [is_doa(r, n, k) for r, n, k in zip(raw.text_raw, raw.text_norm, raw.n_tokens)]
    raw["is_dup"] = raw.duplicated(["platform", "text_norm"], keep="first") | raw.duplicated("comment_id", keep="first")
    raw["is_short"] = raw.n_tokens < 3

    steps = {}
    df = raw
    for flag, name in [("in_period", "di luar periode"), ("is_spam", "spam/promosi/kosong"),
                       ("is_dup", "duplikat"), ("is_short", "< 3 kata"), ("is_doa", "doa tanpa sikap")]:
        drop = ~df[flag] if flag == "in_period" else df[flag]
        steps[name] = df[drop].platform.value_counts().to_dict()
        df = df[~drop]
    summ["dibuang"] = steps
    summ["lolos_cleaning"] = df.platform.value_counts().to_dict()
    print("Lolos cleaning:", summ["lolos_cleaning"])

    # ambil 1.250 per platform: prioritas followers > 1.000, sisanya diisi sampel berstrata peristiwa
    def strat(g, n):
        if n <= 0:
            return g.iloc[:0]
        if len(g) <= n:
            return g
        s = g.groupby("event", dropna=False).sample(frac=n / len(g), random_state=SEED)
        if len(s) < n:
            s = pd.concat([s, g.drop(s.index).sample(n - len(s), random_state=SEED)])
        return s.sample(n, random_state=SEED) if len(s) > n else s

    df["followers_gt_1000"] = df.author_followers > 1000
    parts, summ["seleksi"] = [], {}
    for plat, g in df.groupby("platform"):
        a = g[g.followers_gt_1000]
        pick_a = strat(a, TARGET)
        pick_b = strat(g[~g.followers_gt_1000], TARGET - len(pick_a))
        parts.append(pd.concat([pick_a, pick_b]))
        summ["seleksi"][plat] = dict(tersedia_gt_1000=len(a), diambil_gt_1000=len(pick_a), diisi_lainnya=len(pick_b),
                                     lainnya_followers_diketahui=int(pick_b.author_followers.notna().sum()))
        if len(pick_a) + len(pick_b) < TARGET:
            print(f"PERINGATAN: {plat} hanya {len(pick_a) + len(pick_b)} komentar (< {TARGET})")
    df = pd.concat(parts)
    df = df.sort_values(["platform", "id"], ascending=[False, True]).reset_index(drop=True)
    df["text_ml"] = df.text_norm.map(text_ml)
    df["is_spam"] = False; df["is_doa"] = False; df["is_dup"] = False; df["in_period"] = True

    cols = ["id", "platform", "source_id", "source_url", "source_title", "source_account", "source_date", "event",
            "scraped_at", "comment_id", "parent_id", "is_reply", "author_hash", "text_raw", "like_count",
            "published_at", "text_light", "text_norm", "text_ml", "n_tokens", "is_spam", "is_doa", "is_dup",
            "in_period", "author_followers", "followers_gt_1000"]
    df[cols].to_csv(ROOT / "dataset_clean_v2.csv", index=False)

    summ["final"] = df.platform.value_counts().to_dict()
    summ["final_total"] = len(df)
    summ["n_konten"] = df.groupby("platform").source_id.nunique().to_dict()
    summ["event_platform"] = pd.crosstab(df.event, df.platform).to_dict()
    summ["event_konten"] = df.groupby("event").source_id.nunique().to_dict()
    summ["pct_reply"] = round(df.is_reply.astype(bool).mean() * 100, 1)
    summ["median_tokens"] = float(df.n_tokens.median())
    m = pd.to_datetime(df.published_at, utc=True, errors="coerce").dt.month
    summ["pct_aug_sep"] = round(m.isin([8, 9]).mean() * 100, 1)
    summ["followers_median"] = df.groupby("platform").author_followers.median().to_dict()
    summ["periode"] = [str(df.published_at.min()), str(df.published_at.max())]
    (OUTD / "t1_ringkasan.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

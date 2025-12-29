import re
import pandas as pd

# =========================
# KONFIGURASI
# =========================
INPUT_CSV = "nyarigawe_reviews.csv"
OUTPUT_CSV = "nyarigawe_reviews_clean.csv"

TEXT_COL = "content"
LABEL_COL = "label_sentiment"

DROP_DUPLICATES = True
MIN_WORDS = 2  # buang teks yang terlalu pendek setelah cleaning

# =========================
# STOPWORDS (fallback jika Sastrawi tidak ada)
# =========================
STOPWORDS_FALLBACK = {
    "yang","dan","di","ke","dari","ini","itu","ada","atau","untuk","dengan","pada","saya","aku",
    "kamu","dia","mereka","kami","kita","nya","lah","kok","sih","nih","ya","iya","aja","deh",
    "dong","min","admin","aplikasi","app","nya","ga","gak","nggak","tdk","tidak","bukan","udah",
    "sudah","belum","bgt","banget","bisa","tidak","dapat","dapet","kalo","kalau","karena","jadi",
    "sebagai","juga","lagi","lg","pun","atau","akan","lebih","masih","sangat","harus","mau","mohon",
    "tolong","terima","kasih","terimakasih","halo","selamat"
}

# =========================
# Coba pakai Sastrawi (opsional)
# =========================
HAS_SASTRAWI = False
stemmer = None
stopwords = set(STOPWORDS_FALLBACK)

try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    stop_factory = StopWordRemoverFactory()
    stopwords = set(stop_factory.get_stop_words()) | set(STOPWORDS_FALLBACK)

    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()

    HAS_SASTRAWI = True
except ImportError:
    HAS_SASTRAWI = False


def basic_clean(text: str) -> str:
    """
    Cleaning dasar:
    - lowercase
    - hapus URL & email
    - hapus angka, tanda baca, emoji (sisakan huruf a-z dan spasi)
    - rapikan spasi
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("\n", " ").replace("\t", " ")

    # hapus url & email
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # sisakan huruf a-z dan spasi saja (menghapus emoji/tanda baca/angka)
    text = re.sub(r"[^a-z\s]", " ", text)

    # rapikan spasi
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    if not text:
        return ""
    tokens = [t for t in text.split() if t not in stopwords and len(t) > 1]
    return " ".join(tokens)


def do_stemming(text: str) -> str:
    if (not text) or (stemmer is None):
        return text
    return stemmer.stem(text)


def clean_pipeline(text: str) -> str:
    text = basic_clean(text)
    text = remove_stopwords(text)
    text = do_stemming(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv(INPUT_CSV)

    # validasi kolom penting
    if TEXT_COL not in df.columns:
        raise ValueError(f"Kolom teks '{TEXT_COL}' tidak ditemukan. Kolom yang ada: {list(df.columns)}")
    if LABEL_COL not in df.columns:
        raise ValueError(f"Kolom label '{LABEL_COL}' tidak ditemukan. Kolom yang ada: {list(df.columns)}")

    # cleaning
    df["text_clean"] = df[TEXT_COL].apply(clean_pipeline)

    # buang yang kosong / terlalu pendek
    df["word_count"] = df["text_clean"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    df = df[df["word_count"] >= MIN_WORDS].copy()

    # buang duplikat
    if DROP_DUPLICATES:
        df = df.drop_duplicates(subset=["text_clean", LABEL_COL]).copy()

    # output final (lebih rapi buat modeling)
    out = df[["text_clean", LABEL_COL, "score"]].copy() if "score" in df.columns else df[["text_clean", LABEL_COL]].copy()
    out = out.rename(columns={"text_clean": "text", LABEL_COL: "label"})

    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Selesai ✅")
    print("Pakai Sastrawi:", HAS_SASTRAWI)
    print("Output:", OUTPUT_CSV)
    print("Jumlah data:", len(out))
    print("\nDistribusi label:")
    print(out["label"].value_counts())
    print("\nContoh 5 baris (before → after):")
    preview = df[[TEXT_COL, "text_clean"]].head(5)
    for i, row in preview.iterrows():
        print("-" * 60)
        print("RAW :", row[TEXT_COL])
        print("CLEAN:", row["text_clean"])


if __name__ == "__main__":
    main()

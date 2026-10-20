import os
import re
import numpy as np
from preprocessing import clean_text
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FEATURE_NAMES = [
    "minilm_cosine", "tfidf_cosine", "keypoint_coverage",
    "word_overlap", "log_student_len", "log_reference_len", "length_ratio",
]
V3_FEATURE_NAMES = [
    "minilm_cosine", "tfidf_cosine", "keypoint_min_coverage",
    "word_overlap", "log_student_len", "log_reference_len",
    "length_ratio", "negation_mismatch",
]


def split_key_points(raw_text):
    """
    Split an answer into short "key points".
    Reference answers in EngSAF are often written as bullet lists or several
    sentences, so we split on new lines and on sentence endings, then clean
    each piece with the Week 2 clean_text(). Pieces shorter than 2 words
    (e.g. a lone "FALSE") are dropped.

    Example:
        "- Write to CR3 register\n- Access kernel VA"
        -> ["write to cr3 register", "access kernel va"]
    """
    pieces = re.split(r"\n+|(?<=[.;!?])\s+", str(raw_text))
    points = [clean_text(p) for p in pieces]
    points = [p for p in points if len(p.split()) >= 2]
    if not points:
        points = [clean_text(raw_text)]
    return points


class TextEncoder:
    """
    Small wrapper around SentenceTransformer that remembers every text it has
    already encoded. Many rows share the same reference answer, so this makes
    feature building much faster.
    """

    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache = {}

    def encode(self, texts):
        """Return an (n, dim) array of L2-normalised embeddings."""
        texts = list(texts)
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            vectors = self.model.encode(
                missing,
                batch_size=64,
                normalize_embeddings=True,  # dot product == cosine similarity
                show_progress_bar=len(missing) > 500,
            )
            self.cache.update(zip(missing, vectors))
        return np.vstack([self.cache[t] for t in texts])


def tfidf_cosine(vectorizer, student_clean, reference_clean):
    """Row-by-row TF-IDF cosine similarity (same idea as the Week 2 notebook)."""
    s = vectorizer.transform(student_clean)
    r = vectorizer.transform(reference_clean)
    # TfidfVectorizer rows are L2-normalised, so the row-wise dot product is the cosine.
    return np.asarray(s.multiply(r).sum(axis=1)).ravel()


def _build_features(student_raw, reference_raw, vectorizer, encoder,
    coverage_reducer,include_negation, return_details=False,):
    """
    Build the 7-feature matrix for a list of (student answer, reference answer) pairs.

    student_raw / reference_raw : lists of ORIGINAL (uncleaned) answers.
        We need the original text to split it into key points; the cleaned
        versions are recomputed here with clean_text().
    vectorizer : the TF-IDF vectorizer fitted on the training data.
    encoder    : a TextEncoder.
    return_details : if True, also return, for every row, a list of
        (key_point, best_similarity) pairs - used to write feedback.
    """
    student_raw = ["" if x is None else str(x) for x in student_raw]
    reference_raw = ["" if x is None else str(x) for x in reference_raw]
    student_clean = [clean_text(x) for x in student_raw]
    reference_clean = [clean_text(x) for x in reference_raw]

    if len(student_raw) != len(reference_raw):
        raise ValueError("student_raw and reference_raw must have the same length")

    stud_emb = encoder.encode(student_clean)
    ref_emb = encoder.encode(reference_clean)
    minilm_cos = np.sum(stud_emb * ref_emb, axis=1)
    tfidf_cos = tfidf_cosine(vectorizer, student_clean, reference_clean)

    # Key-point coverage: encode every key point / student sentence once
    ref_points = [split_key_points(x) for x in reference_raw]
    stud_points = [split_key_points(x) for x in student_raw]
    encoder.encode([p for pts in ref_points + stud_points for p in pts])

    rows, details = [], []
    for i in range(len(student_raw)):
        k = encoder.encode(ref_points[i])   # (num_key_points, dim)
        s = encoder.encode(stud_points[i])  # (num_student_sentences, dim)
        best = (k @ s.T).max(axis=1)        # best match for each key point
        coverage = float(coverage_reducer(best))

        s_words = set(student_clean[i].split())
        r_words = set(reference_clean[i].split())
        overlap = len(s_words & r_words) / max(len(r_words), 1)

        s_len = len(student_clean[i].split())
        r_len = len(reference_clean[i].split())

        row = [
            float(minilm_cos[i]),
            float(tfidf_cos[i]),
            coverage,
            overlap,
            np.log1p(s_len),
            np.log1p(r_len),
            min(s_len / max(r_len, 1), 5.0),
        ]
        if include_negation:
            row.append(negation_mismatch(student_clean[i], reference_clean[i]))
        rows.append(row)
        if return_details:
            details.append(list(zip(ref_points[i], best.round(3).tolist())))

    features = np.array(rows, dtype=float)
    return (features, details) if return_details else features


def build_features(student_raw, reference_raw, vectorizer, encoder, return_details=False):
    """Build the seven V2 hybrid features."""
    return _build_features(
        student_raw,
        reference_raw,
        vectorizer,
        encoder,
        coverage_reducer=np.mean,
        include_negation=False,
        return_details=return_details,
)

_NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "cannot", "cant",
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
    "wont", "wouldnt", "shouldnt", "couldnt", "without", "false",
    "incorrect", "untrue", "unable", "lack", "lacks", "lacking", "nothing",
}


def negation_mismatch(student_clean, reference_clean):
    """
    Return 1.0 if one answer contains a negation word and the other doesn't, else 0.0.
    This is a simple feature that helps the model avoid false positives where
    the student answer is the opposite of the reference answer.
    """
    s_words = set(student_clean.split())
    r_words = set(reference_clean.split())
    s_neg = bool(s_words & _NEGATION_WORDS)
    r_neg = bool(r_words & _NEGATION_WORDS)
    return 1.0 if s_neg != r_neg else 0.0


def build_features_v3(student_raw, reference_raw, vectorizer, encoder, return_details=False):
    """Build the eight V3 hybrid features."""
    return _build_features(
        student_raw,
        reference_raw,
        vectorizer,
        encoder,
        coverage_reducer=np.min,
        include_negation=True,
        return_details=return_details,
    )
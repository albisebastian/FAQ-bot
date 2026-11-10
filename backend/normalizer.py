"""Text normalisation for the FAQ matcher: stop-words -> lemmatisation -> stemming.

    "I was refunded for the returned parcels"  ->  "refund return parcel"

Why: TF-IDF only matches identical tokens. Normalising maps different forms of a word to one token, so a question
that uses a form the FAQ never contained ("refunding", "cancelled", "coupons") still matches.

Pipeline (the same code runs in the backend, the Colab notebook and - ported line by line - in the browser):
  1. lower-case, strip accents, drop apostrophes ("can't" -> "cant")
  2. split into word tokens of 2+ characters
  3. drop English stop-words (before and after lemmatising)
  4. lemmatise: irregular forms via a dictionary ("bought" -> "buy", "children" -> "child", "paid" -> "pay")
  5. stem: Porter algorithm ("shipping"/"shipped"/"ships" -> "ship", "codes" -> "code")

Dependencies: scikit-learn (only for its stop-word list). Bump PREPROCESS if you change anything below so that
models trained with an older normaliser are not mixed with this one.
"""
import re
import unicodedata

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

PREPROCESS = "stem_lemma_v2"
# scikit-learn's list misses some very common function and chat words ("does" appears in many FAQs and would make
# unrelated questions match them), so add those.
EXTRA_STOP_WORDS = frozenset({
    "does", "did", "doing", "doesn", "didn", "don", "isn", "wasn", "aren", "weren", "hasn", "haven", "hadn",
    "wouldn", "couldn", "shouldn", "please", "pls", "plz", "hi", "hello", "hey", "thanks", "thank", "thx",
    "want", "wanted", "wants", "need", "needed", "needs", "like", "tell", "let", "ok", "okay", "just",
    "really", "also",
})
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) | EXTRA_STOP_WORDS

# ---------------------------------------------------------------------------------------------------------------
# Lemmatisation: irregular forms that no suffix rule can handle (dictionary-based, like WordNet's exception lists)
# ---------------------------------------------------------------------------------------------------------------
LEMMAS = {
    # irregular verbs
    "arose": "arise", "arisen": "arise", "ate": "eat", "eaten": "eat", "began": "begin", "begun": "begin",
    "bitten": "bite", "blew": "blow", "blown": "blow", "bought": "buy", "brought": "bring",
    "built": "build", "broke": "break", "broken": "break", "came": "come", "caught": "catch", "chose": "choose",
    "chosen": "choose", "dealt": "deal", "drew": "draw", "drawn": "draw", "drove": "drive", "driven": "drive",
    "fell": "fall", "fallen": "fall", "felt": "feel", "flew": "fly", "flown": "fly", "forgot": "forget",
    "forgotten": "forget", "found": "find", "gave": "give", "given": "give", "goes": "go", "went": "go",
    "gone": "go", "got": "get", "gotten": "get", "grew": "grow", "grown": "grow", "held": "hold", "hid": "hide",
    "hidden": "hide", "kept": "keep", "knew": "know", "known": "know", "laid": "lay", "lent": "lend", "lost": "lose", "made": "make", "met": "meet", "paid": "pay", "ran": "run",
    "rang": "ring", "rung": "ring", "risen": "rise", "saw": "see", "seen": "see", "sent": "send",
    "sold": "sell", "spent": "spend", "spoke": "speak", "spoken": "speak", "stole": "steal", "stolen": "steal",
    "swore": "swear", "sworn": "swear", "taught": "teach", "told": "tell", "took": "take", "taken": "take",
    "thought": "think", "threw": "throw", "thrown": "throw", "understood": "understand", "woke": "wake",
    "woken": "wake", "won": "win", "wore": "wear", "worn": "wear", "wrote": "write", "written": "write",
    "bigger": "big", "biggest": "big",
    # irregular nouns
    "children": "child", "men": "man", "women": "woman", "people": "person", "feet": "foot", "teeth": "tooth",
    "mice": "mouse", "geese": "goose", "knives": "knife", "wives": "wife", "shelves": "shelf", "halves": "half",
    "wolves": "wolf", "analyses": "analysis", "indices": "index", "criteria": "criterion",
    "selves": "self", "loaves": "loaf", "thieves": "thief",
    # spelling variants (UK / US) that Porter would stem differently
    "colour": "color", "colours": "color", "favourite": "favorite", "cheque": "check", "grey": "gray",
    "cancelled": "cancel", "cancelling": "cancel", "labelled": "label", "travelled": "travel", "jewellery": "jewelry",
    "instalment": "installment", "instalments": "installment", "centre": "center", "litre": "liter",
    "harbour": "harbor", "neighbour": "neighbor", "neighbours": "neighbor", "programme": "program",
    "personalise": "personalize", "personalised": "personalize", "personalising": "personalize",
    "customise": "customize", "customised": "customize", "customising": "customize",
    "organise": "organize", "organised": "organize", "organising": "organize",
    "recognise": "recognize", "recognised": "recognize", "authorise": "authorize", "authorised": "authorize",
    "apologise": "apologize", "apologised": "apologize", "catalogue": "catalog", "licence": "license",
}

# ---------------------------------------------------------------------------------------------------------------
# Porter stemmer (regex formulation of M. Porter's 1980 algorithm). The JavaScript port in standalone/tfidf.js
# mirrors this code exactly; tests/test_standalone.py checks both give the same output.
# ---------------------------------------------------------------------------------------------------------------
_STEP2 = {
    "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance", "izer": "ize", "bli": "ble", "alli": "al",
    "entli": "ent", "eli": "e", "ousli": "ous", "ization": "ize", "ation": "ate", "ator": "ate", "alism": "al",
    "iveness": "ive", "fulness": "ful", "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble",
    "logi": "log",
}
_STEP3 = {"icate": "ic", "ative": "", "alize": "al", "iciti": "ic", "ical": "ic", "ful": "", "ness": ""}

_c = "[^aeiou]"
_v = "[aeiouy]"
_C = _c + "[^aeiouy]*"
_V = _v + "[aeiou]*"
_MGR0 = re.compile("^(" + _C + ")?" + _V + _C)
_MEQ1 = re.compile("^(" + _C + ")?" + _V + _C + "(" + _V + ")?$")
_MGR1 = re.compile("^(" + _C + ")?" + _V + _C + _V + _C)
_S_V = re.compile("^(" + _C + ")?" + _v)
_CVC = re.compile("^" + _C + _v + "[^aeiouwxy]$")
_DOUBLE = re.compile("([^aeiouylsz])\\1$")
_STEP2_RE = re.compile("^(.+?)(" + "|".join(_STEP2) + ")$")
_STEP3_RE = re.compile("^(.+?)(" + "|".join(_STEP3) + ")$")
_STEP4_RE = re.compile("^(.+?)(al|ance|ence|er|ic|able|ible|ant|ement|ment|ent|ou|ism|ate|iti|ous|ive|ize)$")
_STEP4_ION_RE = re.compile("^(.+?)(s|t)(ion)$")


def stem(w: str) -> str:
    """Porter-stem one lower-case word. Anything that is not purely a-z is returned unchanged."""
    if len(w) < 3 or not re.fullmatch("[a-z]+", w):
        return w
    first = w[0]
    if first == "y":
        w = "Y" + w[1:]          # an initial y is a consonant

    # step 1a: plurals
    m = re.search("^(.+?)(ss|i)es$", w)
    if m:
        w = m.group(1) + m.group(2)
    else:
        m = re.search("^(.+?)([^s])s$", w)
        if m:
            w = m.group(1) + m.group(2)

    # step 1b: -eed / -ed / -ing
    m = re.search("^(.+?)eed$", w)
    if m:
        if _MGR0.search(m.group(1)):
            w = w[:-1]
    else:
        m = re.search("^(.+?)(ed|ing)$", w)
        if m and _S_V.search(m.group(1)):
            w = m.group(1)
            if re.search("(at|bl|iz)$", w):
                w += "e"
            elif _DOUBLE.search(w):
                w = w[:-1]
            elif _CVC.search(w):
                w += "e"

    # step 1c: y -> i
    m = re.search("^(.+?)y$", w)
    if m and _S_V.search(m.group(1)):
        w = m.group(1) + "i"

    # step 2
    m = _STEP2_RE.search(w)
    if m and _MGR0.search(m.group(1)):
        w = m.group(1) + _STEP2[m.group(2)]

    # step 3
    m = _STEP3_RE.search(w)
    if m and _MGR0.search(m.group(1)):
        w = m.group(1) + _STEP3[m.group(2)]

    # step 4
    m = _STEP4_RE.search(w)
    if m:
        if _MGR1.search(m.group(1)):
            w = m.group(1)
    else:
        m = _STEP4_ION_RE.search(w)
        if m and _MGR1.search(m.group(1) + m.group(2)):
            w = m.group(1) + m.group(2)

    # step 5
    m = re.search("^(.+?)e$", w)
    if m:
        s = m.group(1)
        if _MGR1.search(s) or (_MEQ1.search(s) and not _CVC.search(s)):
            w = s
    if re.search("ll$", w) and _MGR1.search(w):
        w = w[:-1]

    return "y" + w[1:] if first == "y" else w


# ---------------------------------------------------------------------------------------------------------------
def _strip_accents(s: str) -> str:
    normalized = unicodedata.normalize("NFKD", s)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def lemmatize(token: str) -> str:
    return LEMMAS.get(token, token)


def tokens(text: str) -> list:
    """Normalised tokens of a text: stop-words removed, lemmatised, stemmed."""
    text = _strip_accents(str(text).lower()).replace("'", "").replace("’", "")
    out = []
    for tok in re.findall(r"\w\w+", text):
        lemma = lemmatize(tok)
        if tok in STOP_WORDS or lemma in STOP_WORDS:     # e.g. "got" -> "get" is a stop-word too
            continue
        out.append(stem(lemma))
    return out


def normalize(text: str) -> str:
    """Normalised text, tokens separated by spaces (feed this to a whitespace-tokenising TfidfVectorizer)."""
    return " ".join(tokens(text))

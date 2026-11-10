import pytest

import normalizer as n


@pytest.mark.parametrize("word,expected", [
    # classic Porter test vectors
    ("caresses", "caress"), ("ponies", "poni"), ("cats", "cat"), ("feed", "feed"), ("agreed", "agre"),
    ("plastered", "plaster"), ("motoring", "motor"), ("sing", "sing"), ("conflated", "conflat"),
    ("troubled", "troubl"), ("sized", "size"), ("hopping", "hop"), ("falling", "fall"), ("hissing", "hiss"),
    ("happy", "happi"), ("sky", "sky"), ("relational", "relat"), ("rational", "ration"), ("digitizer", "digit"),
    ("operator", "oper"), ("feudalism", "feudal"), ("triplicate", "triplic"), ("goodness", "good"),
    ("revival", "reviv"), ("allowance", "allow"), ("adoption", "adopt"), ("controll", "control"), ("roll", "roll"),
    # words that matter for this project
    ("shipping", "ship"), ("shipped", "ship"), ("ships", "ship"), ("codes", "code"), ("refunding", "refund"),
    ("refunded", "refund"), ("deliveries", "deliveri"), ("delivery", "deliveri"), ("cancelled", "cancel"),
])
def test_porter_stemmer(word, expected):
    assert n.stem(word) == expected


def test_stem_leaves_short_and_non_alphabetic_tokens_alone():
    for w in ["a", "is", "y", "abc123", "x_y", "2024", ""]:
        assert n.stem(w) == w


def test_lemmatize_irregular_forms():
    assert n.lemmatize("bought") == "buy" and n.lemmatize("children") == "child" and n.lemmatize("paid") == "pay"
    assert n.lemmatize("parcel") == "parcel"
    # forms collapse to the same token after lemmatising + stemming
    assert n.tokens("bought") == n.tokens("buying") == n.tokens("buys")
    assert n.tokens("lost") == n.tokens("losing")


def test_inflected_forms_collapse_to_one_token():
    groups = [["refund", "refunds", "refunded", "refunding"], ["ship", "ships", "shipped", "shipping"],
              ["coupon", "coupons"], ["cancel", "cancelled", "cancelling", "canceled"], ["code", "codes"]]
    for g in groups:
        assert len({tuple(n.tokens(w)) for w in g}) == 1, g


def test_stop_words_removed_before_and_after_lemmatising():
    assert n.tokens("the and of") == []
    assert n.tokens("does it work please") == ["work"]          # "does", "please" are in EXTRA_STOP_WORDS
    assert n.tokens("I got the wrong product") == ["wrong", "product"]   # "got" -> "get" is a stop-word


def test_normalize_text():
    assert n.normalize("I was refunded for the returned parcels") == "refund return parcel"
    assert n.normalize("Can't track my parcel's shipping!") == "track parcel ship"
    assert n.normalize("CAFÉ déjà vu") == "cafe deja vu"
    assert n.normalize("") == "" and n.normalize("   ") == ""


def test_unambiguous_words_not_mangled_by_lemma_table():
    # words with a common non-verb meaning must not be in the irregular-forms table
    for w in ["left", "rose", "bit", "led", "media", "lives"]:
        assert w not in n.LEMMAS

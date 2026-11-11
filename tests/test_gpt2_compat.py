import os
import pytest
from src.tokenizers.bpe import load_gpt2


GPT2_DIR = "data/gpt2"
VERDICT = "data/the-verdict.txt"


needs_gpt2 = pytest.mark.skipif(
    not os.path.exists(f"{GPT2_DIR}/vocab.bpe")
    or not os.path.exists(f"{GPT2_DIR}/encoder.json"),
    reason = "GPT-2 vocab files not downloaded",
)

needs_verdict = pytest.mark.skipif(
    not os.path.exists(VERDICT), reason = "the-verdict.txt not downloaded"
)

BOOK_TEXT = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
BOOK_IDS = [
    15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252,
    18250, 8812, 2114, 286, 617, 34680, 27271, 13,
]

STRINGS = [
    "hello world",
    "naïve café 😀🚀",
    "  leading and trailing  ",
    "",
    "a\n\nb\tc",
    "don't we'll they've I'm she'd it's",
    "numbers 12345 and 3.14159",
    "日本語のテキスト",
]


@pytest.fixture(scope = "module")
def gpt2():
    return load_gpt2(f"{GPT2_DIR}/encoder.json", f"{GPT2_DIR}/vocab.bpe")


@pytest.fixture(scope = "module")
def verdict():
    with open(VERDICT, encoding = "utf-8") as f:
        return f.read()


@needs_gpt2
def test_vocab_size_is_50257(gpt2):
    assert gpt2.n_vocab == 50257


@needs_gpt2
def test_endoftext_id(gpt2):
    assert gpt2.special_tokens["<|endoftext|>"] == 50256


@needs_gpt2
def test_book_example(gpt2):
    assert gpt2.encode(BOOK_TEXT, allowed_special = {"<|endoftext|>"}) == BOOK_IDS


@needs_gpt2
def test_book_example_decodes_back(gpt2):
    assert gpt2.decode(BOOK_IDS) == BOOK_TEXT


@needs_gpt2
def test_exercise_2_1_unknown_word(gpt2):
    ids = gpt2.encode("Akwirw ier")
    assert ids == [33901, 86, 343, 86, 220, 959]
    assert gpt2.decode(ids) == "Akwirw ier"


@needs_gpt2
@pytest.mark.parametrize("s", STRINGS)
def test_roundtrip_is_lossless(gpt2, s):
    assert gpt2.decode(gpt2.encode(s)) == s


@needs_gpt2
@needs_verdict
def test_verdict_token_count_and_roundtrip(gpt2, verdict):
    ids = gpt2.encode(verdict)
    assert len(ids) == 5145
    assert gpt2.decode(ids) == verdict


@needs_gpt2
@needs_verdict
def test_first_tokens_of_verdict(gpt2, verdict):
    # Book section 2.6: the first batch with max_length=4 starts [40, 367, 2885, 1464]
    assert gpt2.encode(verdict)[:4] == [40, 367, 2885, 1464]


@needs_gpt2
@needs_verdict
def test_matches_tiktoken(gpt2, verdict):
    tiktoken = pytest.importorskip("tiktoken")  # skipped if it can't be installed
    ref = tiktoken.get_encoding("gpt2")
    for s in STRINGS + [verdict]:
        assert gpt2.encode(s) == ref.encode(s), f"mismatch on {s[:40]!r}"
    assert gpt2.encode(BOOK_TEXT, allowed_special = {"<|endoftext|>"}) == ref.encode(
        BOOK_TEXT, allowed_special = {"<|endoftext|>"}
    )

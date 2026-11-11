"""
    Word-level tokenizer (SimpleTokenizerV2), following the approach in
    "Build a Large Language Model (From Scratch)", sections 2.2-2.4.

    Pipeline:
        raw text --split_words--> tokens --build_vocab--> {token: id}
        encode: text -> token ids
        decode: ids  -> text

    What V2 adds over V1:
        V1 raised a KeyError whenever it met a word that was not in the vocabulary.
        V2 maps such words to a special "<|unk|>" (unknown) token instead, so it
        can encode ANY text, even text it has never seen.
"""

import os
import re


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Splits text on punctuation, double dashes, and whitespace.
# The surrounding parentheses make it a *capturing group*, so re.split()
# keeps the delimiters in its output instead of throwing them away.
# Example: "Hello, world" -> ["Hello", ",", "", " ", "world"]
# This is what turns punctuation into separate tokens.
# Note: "<", ">" and "|" are NOT in this pattern, so special tokens such as
# "<|unk|>" stay together as one single token.
WORD_SPLIT = r'([,.:;?_!"()\']|--|\s)'
# Used in decode(): matches whitespace sitting directly BEFORE punctuation,
# so it can be removed (e.g. "Hello , world" -> "Hello, world").
# The punctuation is captured in group 1 so r"\1" can put it back.
DETOK = r'\s+([,.:;?!"()\'])'
# Special tokens. They are added to the vocabulary on purpose (they don't come
# from the training text).
UNK_TOKEN = "<|unk|>"              # stands in for any word not in the vocab
EOT_TOKEN = "<|endoftext|>"        # marks the boundary between two documents
SPECIAL_TOKENS = (EOT_TOKEN, UNK_TOKEN)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_words(text: str) -> list[str]:
    """
        Split text into word and punctuation tokens.

        Steps:
        1. re.split() with a capturing group returns words AND delimiters,
            including empty strings and bare whitespace pieces.
        2. Strip each piece and drop anything that becomes empty,
            leaving only real tokens.

        Example:
            "Hello, world!" -> ["Hello", ",", "world", "!"]
    """
    return [t.strip() for t in re.split(WORD_SPLIT, text) if t.strip()]


def build_vocab(text: str, special_tokens=()) -> dict[str, int]:
    """
        Build a vocabulary mapping each unique token to an integer id.

        - set() removes duplicates; sorted() makes ids deterministic
        (the same text always gives the same vocabulary).
        - special_tokens (e.g. "<|unk|>") are appended at the END, so they get the
        highest ids and don't shift the ids of the normal tokens.
    """
    tokens = sorted(set(split_words(text=text)))
    tokens.extend(special_tokens)

    # enumerate gives ids 0..N-1 in list order
    return {tok: i for i, tok in enumerate(tokens)}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class SimpleTokenizerV2:
    """
        Word-level tokenizer.

        Words missing from the vocabulary are mapped to "<|unk|>", so the vocab
        must contain that token (build it with special_tokens=SPECIAL_TOKENS).
    """

    UNK = UNK_TOKEN

    def __init__(self, vocab: dict[str, int]):

        # Fail early with a clear message instead of a confusing KeyError later
        if self.UNK not in vocab:
            raise ValueError(
                f"vocab must contain {self.UNK!r}; "
                "build it with build_vocab(text, special_tokens = SPECIAL_TOKENS)"
            )
        
        # token -> id, used for encoding
        self.str_to_int = vocab
        # id -> token, the inverse mapping, used for decoding
        self.int_to_str = {i: s for s, i in vocab.items()}


    def encode(self, text: str) -> list[int]:
        """
            Convert text to a list of token ids.

            dict.get(token, unk_id) returns the token's id if it is in the vocab,
            otherwise the id of "<|unk|>". This is the only change from V1.
        """
        unk_id = self.str_to_int[self.UNK]
        return [self.str_to_int.get(t, unk_id) for t in split_words(text = text)]


    def decode(self, ids: list[int]) -> str:
        """
            Convert token ids back to text.

            1. Look up each id and join the tokens with single spaces.
            2. Remove the space that join() put before punctuation, using DETOK.

            Limitations (decoding is not perfectly lossless):
            - Original spacing and line breaks are not preserved.
            - Quotes are treated like closing punctuation, so spacing around them
                can come out slightly off (e.g. '" Who' instead of '"Who').
            - Unknown words come back as "<|unk|>"; the original word is lost.
        """
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(DETOK, r"\1", text)

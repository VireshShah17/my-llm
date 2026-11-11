"""
Word-level tokenizer (SimpleTokenizerV1), following the approach in
"Build a Large Language Model (From Scratch)", sections 2.2-2.4.

Pipeline:
    raw text --split_words--> tokens --build_vocab--> {token: id}
    encode: text   -> token ids
    decode: ids    -> text

Note: the unused imports from the original (`regex`, `Counter`, `defaultdict`)
were removed. Only `re` is needed here.
"""

import re


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Splits text on punctuation, double-dashes, and whitespace.
# The surrounding parentheses make it a *capturing group*, so re.split()
# keeps the delimiters in the output instead of throwing them away.
# That way "Hello, world" -> ["Hello", ",", "", " ", "world"] and punctuation
# becomes its own token.
WORD_SPLIT = r'([,.:;?_!"()\']|--|\s)'

# Used in decode(): matches whitespace that sits directly BEFORE punctuation,
# so we can remove it (e.g. "Hello , world" -> "Hello, world").
# The punctuation is captured in group 1 so we can put it back with r"\1".
DETOK = r'\s+([,.:;?!"()\'])'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_words(text: str) -> list[str]:
    """
        Split text into word and punctuation tokens.

        Steps:
        1. re.split() with a capturing group returns words AND delimiters,
            including empty strings and bare whitespace pieces.
        2. We strip each piece and drop anything that becomes empty,
            leaving only real tokens.

        Example:
            "Hello, world!" -> ["Hello", ",", "world", "!"]
    """
    return [t.strip() for t in re.split(WORD_SPLIT, text) if t.strip()]


def build_vocab(text: str, special_tokens=()) -> dict[str, int]:
    """
        Build a vocabulary mapping each unique token to an integer id.

        - set() removes duplicates, sorted() makes ids deterministic
        (same text always gives the same vocabulary).
        - special_tokens (e.g. "<|unk|>", "<|endoftext|>") are appended at the
        END, so they get the highest ids and don't shift existing ones.
    """
    tokens = sorted(set(split_words(text=text)))
    tokens.extend(special_tokens)
    # enumerate gives ids 0..N-1 in list order
    return {tok: i for i, tok in enumerate(tokens)}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class SimpleTokenizerV1:
    """
        Word-level tokenizer. Cannot handle words missing from the vocab.
    """

    def __init__(self, vocab: dict[str, int]):
        # token -> id, used for encoding
        self.str_to_int = vocab
        # id -> token, the inverse mapping, used for decoding
        self.int_to_str = {i: s for s, i in vocab.items()}


    def encode(self, text: str) -> list[int]:
        """
            Convert text to a list of token ids.

            Raises KeyError if a token was not in the training text
            (V1's known limitation; V2 in the book fixes it with an <|unk|> token).
        """
        return [self.str_to_int[t] for t in split_words(text = text)]

    
    def decode(self, ids: list[int]) -> str:
        """
            Convert token ids back to text.

            1. Look up each id and join tokens with spaces.
            2. Remove the space that join() inserted before punctuation, using DETOK.

            Limitation: decoding is not perfectly lossless. Original spacing and
            line breaks are not preserved, and spaces before an opening bracket or
            quote are also removed (e.g. 'said ("hi")' may decode as 'said("hi")').
        """
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(DETOK, r"\1", text)

"""
    Byte-level BPE tokenizer (GPT-2 style).

    =====================================================================
    WHAT IS BPE (Byte Pair Encoding)?
    =====================================================================
    BPE is a way to build a vocabulary of "sub-word" tokens from data, instead of
    fixing the vocabulary by hand (like whole words in SimpleTokenizerV1/V2).

    The idea:
    1. START with the smallest possible pieces. Here those are the 256 possible
        BYTES. Any text in any language (even emoji) is just a sequence of bytes,
        so nothing can ever be "unknown" (no <|unk|> needed).
    2. COUNT how often each pair of neighbouring tokens appears in the training
        text.
    3. MERGE the most frequent pair into ONE new token and give it a new id.
        Record this merge (e.g. "(t, h) -> th").
    4. REPEAT steps 2-3 until the vocabulary reaches the size you want.

    Tiny example, training text "low low low lower":
        start      : l o w   l o w   l o w   l o w e r      (bytes)
        merge 1    : most frequent pair is (l, o) -> "lo"
        merge 2    : next is (lo, w)              -> "low"
        result     : "low low low low" + "e" "r"  -> common words become 1 token,
                    rare words are split into smaller known pieces.

    ENCODING new text = split into bytes, then apply the learned merges in the
    SAME ORDER they were learned (earliest merge first). That is why `merges`
    insertion order matters: it is the merge "rank".

    DECODING = look up the bytes of every token id, join them, decode as UTF-8.

    PRE-TOKENIZATION: before merging, text is first cut into chunks (words,
    numbers, punctuation, spaces) with a regex. Merges only happen INSIDE a chunk,
    never across chunk boundaries, so tokens like "dog." or "the cat" are never
    learned as a single piece.
    =====================================================================
"""

import json
import regex
from collections import Counter, defaultdict
from collections.abc import Set


# GPT-2 pre-tokenization regex. Each match is one chunk; merges never cross
# chunk boundaries. Alternatives, tried left to right:
#   's|'t|'re|'ve|'m|'ll|'d   English contractions ("don't" -> "don" + "'t")
#    ?\p{L}+                  an optional leading space + a run of letters
#    ?\p{N}+                  an optional leading space + a run of digits
#    ?[^\s\p{L}\p{N}]+        an optional leading space + a run of symbols/punctuation
#   \s+(?!\S)                 whitespace that is NOT followed by a non-space
#                             (so the last space before a word stays with the word)
#   \s+                       any remaining whitespace
GPT2_PATTERN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """
        Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`.

        Example: ids=[5, 1, 2, 7, 1, 2], pair=(1, 2), new_id=9  ->  [5, 9, 7, 9]
        Scans left to right, so for [1, 1, 1] with pair (1, 1) the result is
        [new_id, 1] (the first two are merged, the last one is left alone).
    """
    out, i = [], 0

    while i < len(ids):
        # Is there a pair starting at position i? (needs at least 2 items left)
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)   # write the merged token...
            i += 2               # ...and skip BOTH original tokens
        else:
            # FIX: the original code had `ids[1]` here, which copies the wrong
            # element (always the second one). It must copy the current one.
            out.append(ids[i])
            i += 1

    return out


class BPETokenizer:


    def __init__(self, merges = None, special_tokens = None, byte_to_id = None):
        # merges: {(id_a, id_b): new_id}; Insertion order == rank (earliest = highest priority when encoding).
        self.merges = dict(merges) if merges else {}
        # special_tokens: {"<|endoftext|>": id}. These are never produced by; merging; they are inserted only when explicitly allowed in encode().
        self.special_tokens = dict(special_tokens) if special_tokens else {}
        # byte_to_id[b] = token id of the single byte b.
        # For tokenizers trained here it is simply 0..255. GPT-2's pretrained
        # vocab uses different ids for bytes, hence this indirection.
        self.byte_to_id = list(byte_to_id) if byte_to_id else list(range(256))
        self._build()


    def _build(self):
        """
            Derive the lookup tables from merges / special_tokens / byte_to_id.
        """
        # id -> bytes. First the 256 base tokens (one byte each)...
        self.vocab = {self.byte_to_id[b]: bytes([b]) for b in range(256)}

        # ...then every merged token is the concatenation of its two parts.
        # Merges are stored in learning order, so both parts already exist.
        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

        # pair -> rank (0 = learned first). Used to pick which merge to apply first.
        self.ranks = {pair: r for r, pair in enumerate(self.merges)}
        # id -> special token string (for decoding)
        self.inverse_special = {i: s for s, i in self.special_tokens.items()}
        # word bytes -> token ids; encoding the same word again is then free
        self._cache: dict[bytes, list[int]] = {}


    @property
    def n_vocab(self) -> int:
        """
            Total vocabulary size: bytes + merged tokens + special tokens.
        """
        return len(self.vocab) + len(self.special_tokens)


    def train(self, texts, vocab_size: int, special_tokens = ("<|endoftext|>",)):
        """
            Learn the merges from data.

            texts:      iterable of str. Stream it; don't hold a huge corpus as one string.
            vocab_size: final size = 256 bytes + learned merges + special tokens.
        """
        # How many merges we are allowed to learn
        n_merges = vocab_size - 256 - len(special_tokens)
        assert n_merges > 0, "vocab_size too small"

        # 1) Count unique pre-tokens (words) as byte sequences.
        #    Training on UNIQUE words with a frequency is far faster than
        #    re-scanning the whole corpus on every merge.
        word_freq: Counter[bytes] = Counter()

        for text in texts:
            for w in GPT2_PATTERN.findall(text):
                word_freq[w.encode("utf-8")] += 1

        words = [list(w) for w in word_freq]      # each word = list of byte ids (0..255)
        freqs = list(word_freq.values())          # freqs[i] = how often words[i] occurs

        # 2) Count adjacent pairs, weighted by word frequency, and remember which words contain each pair (so a merge only touches those words).
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)

        for i, ids in enumerate(words):
            for p in zip(ids, ids[1:]):
                pair_counts[p] += freqs[i]
                pair_words[p].add(i)

        # 3) Greedy merge loop: repeatedly merge the most frequent pair.
        merges: dict[tuple[int, int], int] = {}
        next_id = 256                              # ids 0..255 are the raw bytes

        for _ in range(n_merges):

            if not pair_counts:
                break                              # corpus exhausted; vocab saturates early

            # Most frequent pair. Ties are broken by the pair itself so the result is deterministic (same data -> same tokenizer).
            best = max(pair_counts, key = lambda p: (pair_counts[p], p))
            merges[best] = next_id

            # Update only the words that contain `best`
            for i in list(pair_words[best]):
                ids, f = words[i], freqs[i]

                for p in zip(ids, ids[1:]):         # remove this word's old pairs
                    pair_counts[p] -= f

                    if pair_counts[p] <= 0:
                        del pair_counts[p]

                new = _merge(ids, best, next_id)    # apply the merge inside the word

                for p in zip(new, new[1:]):         # add its new pairs
                    pair_counts[p] += f
                    pair_words[p].add(i)

                words[i] = new

            pair_counts.pop(best, None)             # `best` no longer exists anywhere
            next_id += 1

        # Save the result. Special tokens get the ids right after the last merge.
        self.merges = merges
        self.special_tokens = {s: next_id + k for k, s in enumerate(special_tokens)}
        self.byte_to_id = list(range(256))
        self._build()


    def _encode_word(self, wb: bytes) -> list[int]:
        """
            Encode ONE pre-token (as bytes) by applying the learned merges.
        """
        if wb in self._cache:
            return self._cache[wb]

        # Start from single bytes
        ids = [self.byte_to_id[b] for b in wb]

        while len(ids) >= 2:
            # Among all adjacent pairs, pick the one learned EARLIEST (lowest rank), not the most frequent one as in training.
            best = min(zip(ids, ids[1:]), key = lambda p: self.ranks.get(p, float("inf")))

            if best not in self.ranks:
                break                              # no known merge applies any more

            ids = _merge(ids, best, self.merges[best])
        self._cache[wb] = ids

        return ids


    def encode(self, text: str, allowed_special: Set[str] = frozenset()) -> list[int]:
        """
            Convert text to token ids.

            allowed_special: special tokens (e.g. "<|endoftext|>") that should be
            turned into their special id. If not allowed, that string is encoded
            as ordinary text (a safety feature against injected special tokens).
        """
        if allowed_special:
            # Split the text on the allowed special tokens. The capturing group keeps the special tokens themselves in the result list.
            pat = "(" + "|".join(regex.escape(s) for s in allowed_special) + ")"
            parts = regex.split(pat, text)
        else:
            parts = [text]

        out: list[int] = []

        for part in parts:
            if part in allowed_special:
                out.append(self.special_tokens[part])
            else:
                # Normal text: pre-tokenize, then BPE-encode each chunk
                for w in GPT2_PATTERN.findall(part):
                    out.extend(self._encode_word(w.encode("utf-8")))

        return out


    def decode(self, ids: list[int]) -> str:
        """
            Convert token ids back to text (exact inverse of encode).
        """
        chunks = [
            self.inverse_special[i].encode("utf-8") if i in self.inverse_special else self.vocab[i]
            for i in ids
        ]

        # errors="replace": a broken sequence of bytes becomes the U+FFFD character instead of crashing (a model can generate ids that cut a character in half).
        return b"".join(chunks).decode("utf-8", errors="replace")


    def save(self, path: str) -> None:
        """
            Save the tokenizer to a JSON file.
        """
        obj = {
            "merges": [[a, b] for (a, b) in self.merges],   # list order = rank
            "special_tokens": self.special_tokens,
            "byte_to_id": self.byte_to_id,
        }

        with open(path, "w", encoding = "utf-8") as f:
            json.dump(obj, f)


    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """
            For tokenizers trained with this class (ids assigned sequentially from 256).
            (Not for GPT-2's pretrained files; use load_gpt2 for those.)
        """

        with open(path, encoding = "utf-8") as f:
            obj = json.load(f)

        # The i-th merge in the list got id 256 + i during training
        merges = {(a, b): 256 + k for k, (a, b) in enumerate(obj["merges"])}
        return cls(merges, obj["special_tokens"], obj["byte_to_id"])


def bytes_to_unicode() -> dict[int, str]:
    """
        GPT-2's reversible byte -> printable-unicode-char map (used in its vocab files).

        GPT-2 stores its vocab as text, but many bytes (space, newline, control
        characters) are not printable. So every byte is mapped to a visible
        character: printable bytes map to themselves, the rest are shifted to
        unused code points starting at 256 (this is why a space shows up as "Ġ").
    """
    # Bytes that are already printable and keep their own character
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = bs[:], 0

    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)      # map to an unused printable code point
            n += 1

    return dict(zip(bs, map(chr, cs)))


def load_gpt2(encoder_json: str, vocab_bpe: str) -> BPETokenizer:
    """
        Build a tokenizer from OpenAI's GPT-2 files.

        encoder_json: token string -> id  (the whole vocabulary)
        vocab_bpe:    one merge per line "tokenA tokenB", in rank order
    """
    with open(encoder_json, encoding = "utf-8") as f:
        enc = json.load(f)

    b2u = bytes_to_unicode()
    byte_to_id = [enc[b2u[b]] for b in range(256)]   # GPT-2's byte ids are NOT 0..255
    merges = {}

    with open(vocab_bpe, encoding = "utf-8") as f:
        for line in f.read().split("\n")[1:]:         # skip "#version" header line
            if line:
                a, b = line.split(" ")
                # id of the merged token = vocab id of the concatenated string
                merges[(enc[a], enc[b])] = enc[a + b]

    return BPETokenizer(merges, {"<|endoftext|>": enc["<|endoftext|>"]}, byte_to_id)

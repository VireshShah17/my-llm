import json
import os
import tempfile
import unittest
from src.tokenizers.bpe import BPETokenizer, _merge, bytes_to_unicode, load_gpt2


EOT = "<|endoftext|>"
# Tiny corpus whose training result can be worked out by hand:
#   pre-tokens:  "ab" x1, " ab" x3, " cd" x1
#   merge 1: (a, b)      = (97, 98)  appears 1 + 3 = 4 times -> id 256
#   merge 2: (space,256) = (32, 256) appears 3 times         -> id 257
TINY_CORPUS = ["ab ab ab ab cd"]
TINY_VOCAB_SIZE = 256 + 2 + 1          # 256 bytes + 2 merges + 1 special token


def make_tiny_tokenizer() -> BPETokenizer:
    tok = BPETokenizer()
    tok.train(TINY_CORPUS, vocab_size = TINY_VOCAB_SIZE)
    return tok


class TestMerge(unittest.TestCase):
    """
        _merge(ids, pair, new_id): the helper used by both training and encoding.
    """

    def test_replaces_pair_in_the_middle(self):
        self.assertEqual(_merge([5, 1, 2, 7], (1, 2), 9), [5, 9, 7])


    def test_replaces_every_occurrence(self):
        self.assertEqual(_merge([1, 2, 1, 2], (1, 2), 9), [9, 9])
        self.assertEqual(_merge([5, 1, 2, 7, 1, 2], (1, 2), 9), [5, 9, 7, 9])


    def test_overlapping_pairs_are_merged_left_to_right(self):
        # [1, 1, 1] with pair (1, 1): first two merge, the last one stays
        self.assertEqual(_merge([1, 1, 1], (1, 1), 9), [9, 1])


    def test_no_match_returns_same_content(self):
        # Regression test: the old code copied ids[1] instead of ids[i],
        # which turned [4, 5, 6] into [5, 5, 5].
        self.assertEqual(_merge([4, 5, 6], (1, 2), 9), [4, 5, 6])


    def test_empty_and_single_item(self):
        self.assertEqual(_merge([], (1, 2), 9), [])
        self.assertEqual(_merge([1], (1, 2), 9), [1])


class TestTraining(unittest.TestCase):

    def test_learns_expected_merges_in_order(self):
        tok = make_tiny_tokenizer()
        # dict order == rank, so compare as lists of items
        self.assertEqual(list(tok.merges.items()), [((97, 98), 256), ((32, 256), 257)])


    def test_vocab_bytes_of_merged_tokens(self):
        tok = make_tiny_tokenizer()
        self.assertEqual(tok.vocab[256], b"ab")
        self.assertEqual(tok.vocab[257], b" ab")


    def test_special_tokens_get_ids_after_last_merge(self):
        tok = make_tiny_tokenizer()
        self.assertEqual(tok.special_tokens, {EOT: 258})


    def test_n_vocab_matches_requested_size(self):
        tok = make_tiny_tokenizer()
        self.assertEqual(tok.n_vocab, TINY_VOCAB_SIZE)


    def test_training_is_deterministic(self):
        a, b = make_tiny_tokenizer(), make_tiny_tokenizer()
        self.assertEqual(list(a.merges.items()), list(b.merges.items()))


    def test_vocab_too_small_is_rejected(self):
        tok = BPETokenizer()
        
        with self.assertRaises(AssertionError):
            tok.train(TINY_CORPUS, vocab_size = 256)      # no room for any merge
        
        with self.assertRaises(AssertionError):
            tok.train(TINY_CORPUS, vocab_size = 257)      # room only for the special token


    def test_stops_early_when_no_pairs_are_left(self):
        # "ab" can only produce ONE merge, even though we ask for many
        tok = BPETokenizer()
        tok.train(["ab"], vocab_size = 1000)
        self.assertEqual(len(tok.merges), 1)
        self.assertEqual(tok.special_tokens[EOT], 257)  # right after the last merge
        self.assertEqual(tok.n_vocab, 256 + 1 + 1)


    def test_merges_never_cross_chunk_boundaries(self):
        corpus = ["the cat sat on the mat. the cat ate. the dog sat."] * 5
        tok = BPETokenizer()
        tok.train(corpus, vocab_size = 320)
        self.assertGreater(len(tok.merges), 10)         # make sure real merging happened

        for idx in tok.merges.values():
            piece = tok.vocab[idx]
            # a space is only ever allowed as the FIRST byte of a token
            self.assertNotIn(b" ", piece[1:], piece)
            # letters and punctuation are different chunks, so never merged together
            has_letter = any(chr(c).isalpha() for c in piece)
            self.assertFalse(has_letter and b"." in piece, piece)


class TestEncodeDecode(unittest.TestCase):

    def setUp(self):
        self.tok = make_tiny_tokenizer()


    def test_encode_applies_merges(self):
        # "ab"  -> [256];  " ab" -> [32, 97, 98] -> [32, 256] -> [257]
        self.assertEqual(self.tok.encode("ab ab"), [256, 257])


    def test_untouched_bytes_stay_as_byte_ids(self):
        # "xyz" has no learned merges, so it is just its UTF-8 bytes
        self.assertEqual(self.tok.encode("xyz"), [ord("x"), ord("y"), ord("z")])


    def test_empty_string(self):
        self.assertEqual(self.tok.encode(""), [])
        self.assertEqual(self.tok.decode([]), "")


    def test_untrained_tokenizer_is_plain_utf8_bytes(self):
        tok = BPETokenizer()
        self.assertEqual(tok.encode("hi é"), list("hi é".encode("utf-8")))
        self.assertEqual(tok.n_vocab, 256)


    def test_earliest_learned_merge_wins(self):
        # Both (a,b) and (b,c) can apply to "abc"; (a,b) has the lower rank
        # (learned first), so the result must be [ab, c] and not [a, bc].
        tok = BPETokenizer(merges = {(97, 98): 256, (98, 99): 257})
        self.assertEqual(tok.encode("abc"), [256, 99])


    def test_roundtrip_on_text_the_tokenizer_never_saw(self):
        texts = [
            "ab ab cd",
            "héllo wörld",                 # accented characters (multi-byte)
            "你好，世界",                    # Chinese
            "emoji 🙂🚀 test",              # 4-byte characters
            "line1\nline2\n\n  indented\ttab",
            "don't can't we're I'll",      # contractions (regex special case)
            "numbers 12345 and 6.78",
            "   leading and trailing   ",
        ]

        for text in texts:
            with self.subTest(text=text):
                self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_roundtrip_after_training_on_real_looking_text(self):
        corpus = ["the quick brown fox jumps over the lazy dog. " * 20,
                  "Don't panic! It's 2024 and the café is open.\n" * 20]
        tok = BPETokenizer()
        tok.train(corpus, vocab_size = 400)
        for text in corpus + ["something completely new: 🙂 ñ 日本語"]:
            self.assertEqual(tok.decode(tok.encode(text)), text)


    def test_encoding_is_repeatable_with_cache(self):
        first = self.tok.encode("ab ab ab")
        second = self.tok.encode("ab ab ab")        # served from the word cache
        self.assertEqual(first, second)


    def test_trained_tokenizer_compresses_repeated_text(self):
        text = "ab ab ab ab"
        self.assertLess(len(self.tok.encode(text)), len(text.encode("utf-8")))


    def test_decode_of_broken_utf8_does_not_crash(self):
        # id 0xC3 alone is half of a 2-byte character -> replacement character
        self.assertEqual(self.tok.decode([0xC3]), "�")


class TestSpecialTokens(unittest.TestCase):

    def setUp(self):
        self.tok = make_tiny_tokenizer()
        self.eot_id = self.tok.special_tokens[EOT]


    def test_allowed_special_becomes_one_id(self):
        ids = self.tok.encode(f"ab{EOT}cd", allowed_special = {EOT})
        self.assertEqual(ids.count(self.eot_id), 1)
        self.assertEqual(ids[1], self.eot_id)           # [256, EOT, ...]


    def test_special_token_only(self):
        self.assertEqual(self.tok.encode(EOT, allowed_special = {EOT}), [self.eot_id])


    def test_not_allowed_special_is_treated_as_normal_text(self):
        ids = self.tok.encode(f"ab{EOT}cd")             # allowed_special not given
        self.assertNotIn(self.eot_id, ids)
        self.assertEqual(self.tok.decode(ids), f"ab{EOT}cd")


    def test_special_token_roundtrip(self):
        text = f"ab ab{EOT}cd cd{EOT}"
        ids = self.tok.encode(text, allowed_special = {EOT})
        self.assertEqual(self.tok.decode(ids), text)


class TestSaveLoad(unittest.TestCase):

    def test_saved_tokenizer_behaves_identically(self):
        tok = BPETokenizer()
        tok.train(["the cat sat on the mat. " * 10], vocab_size = 300)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tok.json")
            tok.save(path)
            loaded = BPETokenizer.load(path)

        self.assertEqual(list(loaded.merges.items()), list(tok.merges.items()))
        self.assertEqual(loaded.special_tokens, tok.special_tokens)
        self.assertEqual(loaded.n_vocab, tok.n_vocab)

        text = f"the cat 🙂 sat{EOT}"
        self.assertEqual(
            loaded.encode(text, allowed_special={EOT}),
            tok.encode(text, allowed_special={EOT}),
        )


class TestBytesToUnicode(unittest.TestCase):

    def test_covers_all_256_bytes_with_unique_chars(self):
        b2u = bytes_to_unicode()
        self.assertEqual(sorted(b2u), list(range(256)))
        self.assertEqual(len(set(b2u.values())), 256)   # reversible (no collisions)


    def test_printable_ascii_maps_to_itself(self):
        b2u = bytes_to_unicode()
        self.assertEqual(b2u[ord("A")], "A")
        self.assertEqual(b2u[ord("!")], "!")


    def test_space_and_newline_use_gpt2_symbols(self):
        b2u = bytes_to_unicode()
        self.assertEqual(b2u[ord(" ")], "Ġ")       # "Ġ"
        self.assertEqual(b2u[ord("\n")], "Ċ")      # "Ċ"


class TestLoadGPT2(unittest.TestCase):
    """
        Uses a tiny fake GPT-2 style vocab with NON-identity byte ids (1000 + byte).
    """

    def test_load_from_gpt2_style_files(self):
        b2u = bytes_to_unicode()
        enc = {b2u[b]: 1000 + b for b in range(256)}    # byte ids are NOT 0..255
        enc[b2u[ord("a")] + b2u[ord("b")]] = 2000       # merged token "ab"
        enc[EOT] = 5000

        with tempfile.TemporaryDirectory() as d:
            enc_path = os.path.join(d, "encoder.json")
            bpe_path = os.path.join(d, "vocab.bpe")

            with open(enc_path, "w", encoding = "utf-8") as f:
                json.dump(enc, f)
            
            with open(bpe_path, "w", encoding = "utf-8") as f:
                f.write("#version: 0.2\n")
                f.write(f"{b2u[ord('a')]} {b2u[ord('b')]}\n")

            tok = load_gpt2(enc_path, bpe_path)

        self.assertEqual(tok.encode("ab"), [2000])              # merge applied
        self.assertEqual(tok.encode("a"), [1000 + ord("a")])    # custom byte ids used
        self.assertEqual(tok.decode([2000, 1000 + ord("c")]), "abc")
        self.assertEqual(tok.encode(EOT, allowed_special = {EOT}), [5000])
        self.assertEqual(tok.n_vocab, 256 + 1 + 1)


if __name__ == "__main__":
    unittest.main(verbosity = 2)

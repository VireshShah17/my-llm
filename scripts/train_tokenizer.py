import glob, resource, time
import re
from src.tokenizers.bpe import BPETokenizer


def clean(line: str) -> str:
    return re.sub(r"[ \t]{4,}", "   ", line)   # cap runs of spaces/tabs at 3


def stream_lines(pattern = "data/corpus/*.txt"):
    for path in sorted(glob.glob(pattern)):

        with open(path, encoding = "utf-8") as f:
            inside = False

            for line in f:

                if line.startswith("*** START"):
                    inside = True
                    continue
                
                if line.startswith("*** END"):
                    break

                if inside:
                    yield clean(line = line)


if __name__ == "__main__":
    tok = BPETokenizer()
    t0 = time.time()
    tok.train(stream_lines(), vocab_size = 8192)
    print(f"train time: {time.time()-t0:.1f}s")
    print(f"peak mem:   {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
    print(f"n_vocab:    {tok.n_vocab}")
    tok.save("data/tok_8k.json")

    nbytes = ntok = 0
    for line in stream_lines():
        nbytes += len(line.encode("utf-8"))
        ntok += len(tok.encode(line))
    print(f"bytes/token: {nbytes/ntok:.2f}  (tokens: {ntok:,})")
    print("longest tokens:", sorted(tok.vocab.values(), key = len)[-20:])

from __future__ import annotations
import argparse
from pathlib import Path
import sentencepiece as spm

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--vocab_size", type=int, default=4000)
    p.add_argument("--model_prefix", default="data/spm_unigram_4k")
    args = p.parse_args()

    inp = Path(args.input)
    assert inp.exists(), f"not found: {inp}"

    spm.SentencePieceTrainer.Train(
        input=str(inp),
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0
    )
    print("saved:", args.model_prefix + ".model", args.model_prefix + ".vocab")

if __name__ == "__main__":
    main()
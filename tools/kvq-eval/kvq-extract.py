#!/usr/bin/env python3
# extract one row of a tokenized jsonl into an int32 token file for llama-kvq-eval
# output: ids[0..e] where ids[e] = <|im_end|>; prints s (prefix length) and counts only

import argparse
import json
import struct
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("row", type=int, help="0-based line number")
    ap.add_argument("out")
    ap.add_argument("--im-start", type=int, required=True, help="token id of <|im_start|>")
    ap.add_argument("--im-end", type=int, required=True, help="token id of <|im_end|>")
    args = ap.parse_args()

    with open(args.jsonl, "rb") as f:
        for i, line in enumerate(f):
            if i == args.row:
                d = json.loads(line)
                break
        else:
            print(f"row {args.row} not found", file=sys.stderr)
            return 1

    ids = d["input_ids"]
    n_prompt = d["metadata"]["prompt_tokens"]

    s = max(i for i in range(n_prompt) if ids[i] == args.im_start)
    appended = ids[-1] != args.im_end
    if appended:
        ids = ids + [args.im_end]
    e = len(ids) - 1

    with open(args.out, "wb") as f:
        f.write(struct.pack(f"<{len(ids)}i", *ids))

    print(f"row={args.row} n_seq={len(d['input_ids'])} prompt_tokens={n_prompt} s={s} e={e} fed={e - s} scored={e - s - 2} appended_im_end={appended}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

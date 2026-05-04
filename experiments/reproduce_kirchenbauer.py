"""Reproduce a slice of paper Tab. 2 / Fig. 4: TPR/FPR for soft watermark on OPT.

Generate N watermarked vs. non-watermarked completions on C4 RealNews-like
prompts, run the detector on both, and report the confusion matrix at
z >= z_threshold.

Defaults pick a small model (`facebook/opt-125m`) so this can be smoked
on CPU. For the headline 99.4% TPR @ 0% FPR scrap the model name to
`facebook/opt-1.3b` and the prompt count to ~500.
"""

from __future__ import annotations

import argparse

from datasets import load_dataset
from loguru import logger

from runic.core.config import WatermarkConfig
from runic.detection.detector import WatermarkDetector
from runic.generation.generator import WatermarkGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=4.0)
    args = parser.parse_args()

    cfg = WatermarkConfig(
        gamma=args.gamma,
        delta=args.delta,
        z_threshold=args.threshold,
    )
    gen = WatermarkGenerator(args.model, config=cfg)
    det = WatermarkDetector(tokenizer=gen.tokenizer, config=cfg)

    ds = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
    prompts: list[str] = []
    for example in ds:
        text = example["text"]
        words = text.split()
        if len(words) >= 100:
            prompts.append(" ".join(words[:50]))
        if len(prompts) >= args.n_prompts:
            break

    tp = fn = fp = tn = 0
    for i, prompt in enumerate(prompts):
        wm = gen.generate(prompt, max_new_tokens=args.max_new_tokens, watermark=True, seed=i)
        no_wm = gen.generate(prompt, max_new_tokens=args.max_new_tokens, watermark=False, seed=i)

        wm_stats = det.detect(wm)
        nowm_stats = det.detect(no_wm)
        tp += int(wm_stats.is_watermarked)
        fn += int(not wm_stats.is_watermarked)
        fp += int(nowm_stats.is_watermarked)
        tn += int(not nowm_stats.is_watermarked)
        logger.info(
            f"[{i + 1}/{len(prompts)}] watermarked z={wm_stats.z_score:.2f} | "
            f"clean z={nowm_stats.z_score:.2f}"
        )

    n = len(prompts)
    logger.info(
        f"\nTPR = {tp}/{n} = {tp / n:.2%}    FPR = {fp}/{n} = {fp / n:.2%}\n"
        f"FN  = {fn}/{n}       TN  = {tn}/{n}"
    )


if __name__ == "__main__":
    main()

"""Sweep T5 span-replacement budget vs. watermark detection (paper Fig. 5).

Generate one watermarked sequence, run the attacker at increasing
budgets, and watch z-score and detection probability degrade.
"""

from __future__ import annotations

import argparse

from loguru import logger

from runic.attacks.span_replacement import SpanReplacementAttack
from runic.core.config import WatermarkConfig
from runic.detection.detector import WatermarkDetector
from runic.generation.generator import WatermarkGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--prompt", default="The history of cryptography began with")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--replacement-model", default="t5-small")
    args = parser.parse_args()

    cfg = WatermarkConfig()
    gen = WatermarkGenerator(args.model, config=cfg)
    det = WatermarkDetector(tokenizer=gen.tokenizer, config=cfg)
    attacker = SpanReplacementAttack(replacement_model=args.replacement_model)

    watermarked = gen.generate(args.prompt, max_new_tokens=args.max_new_tokens, seed=0)
    logger.info(f"Original z = {det.detect(watermarked).z_score:.2f}")

    for budget in args.budgets:
        attacked = (
            watermarked if budget == 0 else attacker.attack(watermarked, budget=budget, seed=0)
        )
        result = det.detect(attacked)
        logger.info(
            f"budget={budget:.2f}  z={result.z_score:.2f}  p={result.p_value:.3g}  "
            f"watermarked={result.is_watermarked}"
        )


if __name__ == "__main__":
    main()

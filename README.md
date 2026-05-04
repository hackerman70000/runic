# runic

Statistical watermarking for large language models. Implements the soft
green-list watermark from Kirchenbauer et al. (ICML 2023) plus the T5
span-replacement attack used to stress-test it.

## Why

LLM provenance: embed an algorithmically detectable but
human-imperceptible signal into generated text, with rigorous p-values
and detection that needs only the tokenizer plus a shared secret.

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,attacks]"
pre-commit install
```

`attacks` extra pulls `sentencepiece` for the T5 tokenizer.

## Quickstart

```python
from runic import WatermarkConfig, WatermarkGenerator, WatermarkDetector

cfg = WatermarkConfig(gamma=0.25, delta=2.0, hash_key=42)

gen = WatermarkGenerator("facebook/opt-125m", config=cfg)
text = gen.generate("The history of cryptography began with", max_new_tokens=200)

det = WatermarkDetector(tokenizer=gen.tokenizer, config=cfg)
result = det.detect(text)
print(f"z={result.z_score:.2f}  p={result.p_value:.3g}  watermarked={result.is_watermarked}")
```

CLI:

```bash
runic generate facebook/opt-125m "The history of cryptography began with" --max-new-tokens 200
runic detect facebook/opt-125m "<text or path/to/file.txt>"
runic attack <text or path/to/file.txt> --budget 0.1 --replacement-model t5-small
```

## What it does

- `WatermarkConfig` — gamma, delta, hash_key, z_threshold.
- `WatermarkLogitsProcessor` — HuggingFace `LogitsProcessor` injecting the
  green-list bias at sampling time. Drops into any `model.generate()`.
- `WatermarkGenerator` — convenience wrapper around `AutoModelForCausalLM`.
- `WatermarkDetector` — model-free; tokenises a string and runs the z-test.
- `SpanReplacementAttack` — paper §7.1 attack using a T5 model to refill
  masked words.

## Layout

```
src/runic/
    core/         WatermarkConfig, vocab partition, z-test
    generation/   LogitsProcessor + Generator wrapper
    detection/    WatermarkDetector (model-free verification)
    attacks/      T5 span-replacement attack
    cli/          typer commands: generate / detect / attack
tests/            pytest suite (no GPU needed; uses StubTokenizer)
experiments/      reproduce_kirchenbauer.py + attack_budget_sweep.py
docs/             METHOD.md — algorithm reference
```

## Development

```bash
pytest                  # tests
ruff check . && ruff format .
mypy src
pre-commit run --all
```

CI runs all of the above on push.

## References

- Kirchenbauer, Geiping, Wen, Katz, Miers, Goldstein. *A Watermark for Large Language Models.* ICML 2023.

See `docs/METHOD.md` for the formal algorithm reference.

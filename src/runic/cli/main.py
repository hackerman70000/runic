from __future__ import annotations

from pathlib import Path

import typer
from transformers import AutoTokenizer

from runic.attacks.span_replacement import SpanReplacementAttack
from runic.core.config import WatermarkConfig
from runic.detection.detector import WatermarkDetector
from runic.generation.generator import WatermarkGenerator

app = typer.Typer(
    help="runic — statistical watermarking for LLMs (Kirchenbauer et al. 2023).",
    no_args_is_help=True,
)


def _build_config(gamma: float, delta: float, hash_key: int, threshold: float) -> WatermarkConfig:
    return WatermarkConfig(gamma=gamma, delta=delta, hash_key=hash_key, z_threshold=threshold)


@app.command()
def generate(
    model_name: str = typer.Argument(..., help="HuggingFace model identifier."),
    prompt: str = typer.Argument(..., help="Prompt to continue."),
    max_new_tokens: int = typer.Option(200, "--max-new-tokens", "-n"),
    gamma: float = typer.Option(0.25, "--gamma", "-g"),
    delta: float = typer.Option(2.0, "--delta", "-d"),
    hash_key: int = typer.Option(15485863, "--hash-key", "-k"),
    threshold: float = typer.Option(4.0, "--threshold"),
    no_watermark: bool = typer.Option(False, "--no-watermark"),
    seed: int | None = typer.Option(None, "--seed"),
    device: str = typer.Option("auto"),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Generate text with the watermark logits processor injected."""
    cfg = _build_config(gamma, delta, hash_key, threshold)
    gen = WatermarkGenerator(model_name, config=cfg, device=device)
    text = gen.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        watermark=not no_watermark,
        seed=seed,
    )

    typer.echo(text)
    if output:
        output.write_text(text)
        typer.echo(f"\nSaved to {output}")


@app.command()
def detect(
    model_name: str = typer.Argument(..., help="Tokenizer source (HF model id)."),
    text: str = typer.Argument(..., help="Text to test, or path to a .txt file."),
    gamma: float = typer.Option(0.25, "--gamma", "-g"),
    delta: float = typer.Option(2.0, "--delta", "-d"),
    hash_key: int = typer.Option(15485863, "--hash-key", "-k"),
    threshold: float = typer.Option(4.0, "--threshold"),
) -> None:
    """Test whether `text` carries the watermark — model-free, just tokenizer."""
    cfg = _build_config(gamma, delta, hash_key, threshold)
    payload = Path(text).read_text() if Path(text).exists() else text

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    det = WatermarkDetector(tokenizer=tokenizer, config=cfg)
    result = det.detect(payload)

    verdict = "WATERMARKED" if result.is_watermarked else "no signal"
    color = typer.colors.RED if result.is_watermarked else typer.colors.GREEN
    typer.secho(
        f"\n{verdict}: z={result.z_score:.3f} p={result.p_value:.3g} "
        f"green={result.green_count}/{result.total_tokens} "
        f"({100 * result.green_fraction:.1f}%)",
        fg=color,
    )


@app.command()
def attack(
    text: str = typer.Argument(..., help="Watermarked text or path to a .txt file."),
    budget: float = typer.Option(0.1, "--budget", "-b"),
    replacement_model: str = typer.Option("t5-large", "--replacement-model"),
    seed: int | None = typer.Option(None, "--seed"),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Run T5 span-replacement attack on watermarked text (paper section 7.1)."""
    payload = Path(text).read_text() if Path(text).exists() else text
    attacker = SpanReplacementAttack(replacement_model=replacement_model)
    attacked = attacker.attack(payload, budget=budget, seed=seed)

    typer.echo(attacked)
    if output:
        output.write_text(attacked)
        typer.echo(f"\nSaved to {output}")


if __name__ == "__main__":
    app()

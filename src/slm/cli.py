"""Command-line interface for BYO-SLM.

Subcommands::

    slm prepare-data --config configs/tiny.yaml   # tokenizer + token binaries
    slm train        --config configs/tiny.yaml   # train (optionally --resume)
    slm generate     --model-dir checkpoints/tiny --prompt "Hello"
    slm serve        --host 0.0.0.0 --port 8000
    slm info                                       # show effective settings
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from slm.config import get_settings, load_experiment_config
from slm.logging_config import configure_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Build, train, and serve your own Small Language Model.",
)
console = Console()


@app.command("prepare-data")
def prepare_data(
    config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML path."),
    retrain_tokenizer: bool = typer.Option(True, help="Retrain the tokenizer from scratch."),
) -> None:
    """Train the tokenizer and write train/val token binaries."""
    from slm.data import prepare_dataset

    configure_logging()
    cfg = load_experiment_config(config)
    stats = prepare_dataset(cfg, retrain_tokenizer=retrain_tokenizer)
    console.print(
        f"[green]Prepared[/green] {stats.train_tokens:,} train / {stats.val_tokens:,} val "
        f"tokens (vocab={stats.vocab_size})."
    )


@app.command("train")
def train(
    config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML path."),
    resume: bool = typer.Option(False, help="Resume from an existing checkpoint."),
) -> None:
    """Train the model defined by an experiment config."""
    from slm.training.trainer import Trainer

    configure_logging()
    cfg = load_experiment_config(config)
    trainer = Trainer(cfg, resume=resume)
    state = trainer.train()
    console.print(
        f"[green]Training complete.[/green] step={state.step} "
        f"best_val_loss={state.best_val_loss:.4f}"
    )


@app.command("generate")
def generate(
    model_dir: Path = typer.Option(..., "--model-dir", "-m", help="Directory with model.pt."),
    prompt: str = typer.Option("", "--prompt", "-p", help="Prompt text."),
    max_tokens: int = typer.Option(128, help="Maximum new tokens."),
    temperature: float = typer.Option(0.8, help="Sampling temperature (0 = greedy)."),
    top_k: int = typer.Option(40, help="Top-k filtering."),
    top_p: float = typer.Option(0.95, help="Top-p (nucleus) filtering."),
    seed: int = typer.Option(-1, help="RNG seed (<0 for nondeterministic)."),
    device: str = typer.Option("auto", help="auto | cpu | cuda | mps."),
    stream: bool = typer.Option(True, help="Stream tokens to stdout."),
) -> None:
    """Generate text from a trained model."""
    from slm.generation.sampler import GenerationConfig
    from slm.inference.engine import InferenceEngine

    configure_logging(level="WARNING")
    engine = InferenceEngine.from_pretrained(
        model_dir, device=device, max_new_tokens_limit=max(max_tokens, 8192)
    )
    gen_config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=None if seed < 0 else seed,
    )
    console.print(f"[dim]{prompt}[/dim]", end="")
    if stream:
        for chunk in engine.stream(prompt, gen_config):
            console.print(chunk, end="", style="bold")
        console.print()
    else:
        result = engine.generate(prompt, gen_config)
        console.print(result.text)


@app.command("serve")
def serve(
    host: str | None = typer.Option(None, help="Bind host (defaults to SLM_API_HOST)."),
    port: int | None = typer.Option(None, help="Bind port (defaults to SLM_API_PORT)."),
    workers: int = typer.Option(1, help="Number of uvicorn workers."),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev only)."),
) -> None:
    """Run the HTTP inference server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(  # pragma: no cover - blocking server entrypoint
        "slm.api.app:create_app",
        factory=True,
        host=host or settings.api_host,
        port=port or settings.api_port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


@app.command("info")
def info() -> None:
    """Print the effective runtime settings."""
    from slm import __version__

    settings = get_settings()
    table = Table(title=f"BYO-SLM v{__version__}")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    for field_name in settings.__class__.model_fields:
        value = getattr(settings, field_name)
        if field_name == "api_keys":
            value = f"<{len(value)} configured>"
        table.add_row(field_name, str(value))
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()

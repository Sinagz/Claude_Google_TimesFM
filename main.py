"""
Multi-Agent Stock Analysis & Forecasting System
─────────────────────────────────────────────────
Entry point.  Usage:

    python main.py                       # uses config.yaml
    python main.py --config my.yaml      # custom config path
    python main.py --tickers AAPL MSFT   # override ticker list
    python main.py --no-cache            # force re-fetch all data
"""

import argparse
import sys

from models.model_loader import print_system_info, report_model_availability, set_hf_cache
from utils.helpers        import load_config, setup_logger

logger = setup_logger("main")


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-Agent Stock Analysis & Forecasting System"
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    p.add_argument(
        "--tickers", nargs="+", default=None,
        help="Override ticker list (e.g. --tickers AAPL MSFT NVDA)",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Ignore cached CSVs and re-fetch all data",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Run data + features only (skip models — fast sanity check)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    # Apply CLI overrides
    if args.tickers:
        config["tickers"] = {"us": args.tickers, "canada": []}
        logger.info("Ticker override: %s", args.tickers)

    if args.no_cache:
        config["data"]["cache_hours"] = 0
        logger.info("Cache disabled — will re-fetch all data")

    # ── Setup logging level from config ──────────────────────────────────────
    log_level = config.get("logging_level", "INFO")
    setup_logger("main", log_level)

    # ── Point HuggingFace to local cache ─────────────────────────────────────
    set_hf_cache("./models/cache")

    # ── Print environment info ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Multi-Agent Stock Forecasting System")
    logger.info("=" * 60)
    print_system_info()
    logger.info("Checking model availability …")
    report_model_availability()

    if args.dry_run:
        logger.info("--dry-run: running data + features only")
        _dry_run(config)
        return

    # ── Run full pipeline via Orchestrator ────────────────────────────────────
    from agents.chronos_agent  import ModelLoadError as ChronosLoadError
    from agents.orchestrator   import Orchestrator
    from agents.timesfm_agent  import ModelLoadError as TimesFMLoadError

    orchestrator = Orchestrator(config)
    try:
        results = orchestrator.run()
    except (TimesFMLoadError, ChronosLoadError) as exc:
        logger.critical("")
        logger.critical("=" * 60)
        logger.critical("  PIPELINE ABORTED — core model failed to load")
        logger.critical("  %s", exc)
        logger.critical("  Fix the issue above and re-run.")
        logger.critical("=" * 60)
        sys.exit(1)

    if not results:
        logger.error("Pipeline produced no results")
        sys.exit(1)

    # ── Rich visual report ────────────────────────────────────────────────────
    from utils.reporter import generate_report
    generate_report(results, config)

    # ── Interactive HTML dashboard ────────────────────────────────────────────
    try:
        from utils.dashboard import generate_dashboard
        dash_path = generate_dashboard(results, config, getattr(orchestrator, "raw_data", {}))
        if dash_path:
            logger.info("Dashboard -> %s", dash_path)
    except Exception as exc:
        logger.warning("Dashboard generation failed (non-critical): %s", exc)


def _dry_run(config: dict):
    """Quick sanity check: data + features only."""
    from agents.data_agent    import DataAgent
    from agents.feature_agent import FeatureAgent

    raw   = DataAgent(config).run()
    feat, scores, arrays = FeatureAgent(config).run(raw)
    print(f"\nDry-run complete: {len(feat)} tickers loaded & featured.")
    for t, s in list(scores.items())[:5]:
        print(f"  {t:12s}  tech_score={s:.3f}")


if __name__ == "__main__":
    main()

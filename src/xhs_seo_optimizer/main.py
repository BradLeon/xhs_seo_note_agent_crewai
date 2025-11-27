#!/usr/bin/env python
"""Entry point for XHS SEO Optimizer Flow.

This script loads input data and runs the complete optimization pipeline
using CrewAI Flow.

Usage:
    python -m xhs_seo_optimizer.main

    Or from project root:
    PYTHONPATH=src:$PYTHONPATH python src/xhs_seo_optimizer/main.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src to path if running directly
project_root = Path(__file__).parent.parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
except ImportError:
    print("python-dotenv not installed, using shell environment variables")


def load_json(path: str) -> Any:
    """Load JSON file and return parsed content.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON content
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save
        path: Path to output file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        if hasattr(data, 'model_dump_json'):
            f.write(data.model_dump_json(indent=2, ensure_ascii=False))
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)


def run_flow(
    keyword: str,
    target_notes: List[Dict[str, Any]],
    owned_note: Dict[str, Any],
    save_outputs: bool = True
) -> Any:
    """Run the XHS SEO optimization flow.

    Args:
        keyword: Target SEO keyword
        target_notes: List of competitor note dicts
        owned_note: Client's note dict to optimize
        save_outputs: Whether to save intermediate outputs to files

    Returns:
        Final flow state with optimized note
    """
    from xhs_seo_optimizer.flow import XhsSeoOptimizerFlow
    from xhs_seo_optimizer.flow_state import XhsSeoFlowState

    # Initialize flow
    flow = XhsSeoOptimizerFlow()

    # Set initial state
    flow.state = XhsSeoFlowState(
        keyword=keyword,
        target_notes=target_notes,
        owned_note=owned_note,
    )

    # Execute flow
    result = flow.kickoff()

    # Optionally save outputs
    if save_outputs and result:
        output_dir = project_root / "outputs"

        if result.success_profile_report:
            save_json(
                result.success_profile_report,
                str(output_dir / "success_profile_report.json")
            )
            print(f"Saved: outputs/success_profile_report.json")

        if result.audit_report:
            save_json(
                result.audit_report,
                str(output_dir / "audit_report.json")
            )
            print(f"Saved: outputs/audit_report.json")

        if result.gap_report:
            save_json(
                result.gap_report,
                str(output_dir / "gap_report.json")
            )
            print(f"Saved: outputs/gap_report.json")

        if result.optimized_note:
            save_json(
                result.optimized_note,
                str(output_dir / "optimized_note.json")
            )
            print(f"Saved: outputs/optimized_note.json")

    return result


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("XHS SEO Optimizer - Flow Mode")
    print("=" * 60 + "\n")

    # Check for required environment variable
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set it before running:")
        print("  export OPENROUTER_API_KEY='your-api-key'")
        sys.exit(1)

    # Load input data
    docs_dir = project_root / "docs"

    print("Loading input data...")

    # Load keyword
    keyword_path = docs_dir / "keyword.json"
    if keyword_path.exists():
        keyword_data = load_json(str(keyword_path))
        keyword = keyword_data if isinstance(keyword_data, str) else keyword_data.get("keyword", "")
    else:
        keyword = "老爸测评dha推荐哪几款"
    print(f"  Keyword: {keyword}")

    # Load target notes
    target_notes_path = docs_dir / "target_notes.json"
    if not target_notes_path.exists():
        print(f"Error: Target notes file not found: {target_notes_path}")
        sys.exit(1)
    target_notes = load_json(str(target_notes_path))
    print(f"  Target Notes: {len(target_notes)} notes loaded")

    # Load owned note
    owned_note_path = docs_dir / "owned_note.json"
    if not owned_note_path.exists():
        print(f"Error: Owned note file not found: {owned_note_path}")
        sys.exit(1)
    owned_note = load_json(str(owned_note_path))
    print(f"  Owned Note: {owned_note.get('note_id', 'unknown')}")

    print("\n" + "-" * 60)
    print("Starting optimization flow...")
    print("-" * 60 + "\n")

    # Run flow
    result = run_flow(
        keyword=keyword,
        target_notes=target_notes,
        owned_note=owned_note,
        save_outputs=True
    )

    # Print summary
    if result and result.optimized_note:
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Original Note ID: {result.optimized_note.original_note_id}")
        print(f"Optimized Note ID: {result.optimized_note.note_id}")
        print(f"New Title: {result.optimized_note.title}")
        print(f"Cover Image Source: {result.optimized_note.cover_image_source}")
        print(f"Inner Images Source: {result.optimized_note.inner_images_source}")
        print(f"\nOptimization Summary:")
        print(result.optimized_note.optimization_summary)
        print("=" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print("OPTIMIZATION FAILED")
        print("=" * 60)
        if result and result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  - {error}")
        print("=" * 60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

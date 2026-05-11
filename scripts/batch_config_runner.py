#!/usr/bin/env python3
"""
Batch run all YAML config files in a directory.

This script recursively finds all .yaml files in a given directory and runs them
through the cf_bench pipeline. Useful for re-running experiments like predictors_vs_threshold.

By default, continues running even if individual configs fail, so you can batch process
large experiment sets without manual intervention.

Usage:
    python scripts/batch_config_runner.py configs/predictors_vs_threshold
    python scripts/batch_config_runner.py configs/predictors_vs_threshold --dry-run
    python scripts/batch_config_runner.py configs/predictors_vs_threshold/baseline
    python scripts/batch_config_runner.py configs/predictors_vs_threshold --debug
"""

# ============================================================================
# IMPORTS
# ============================================================================

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

# ============================================================================
# CONFIG BATCH RUNNER CLASS
# ============================================================================


class ConfigBatchRunner:
    """
    Handles batch execution of YAML config files.

    This class manages the entire batch run process including:
    - Finding config files
    - Creating output directories
    - Running each config
    - Tracking results
    - Generating summaries
    """

    def __init__(self, args):
        """
        Initialize the batch runner.

        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.config_dir = args.directory
        self.debug = args.debug
        self.dry_run = args.dry_run
        self.stop_on_error = args.stop_on_error

        # Timestamp for this batch run
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Setup output paths
        config_dir_name = (
            self.config_dir.name if self.config_dir.name else self.config_dir.parts[-1]
        )
        batch_folder_name = f"{config_dir_name}_{self.timestamp}"
        self.output_dir = Path("cf_outputs") / "batch_runs" / batch_folder_name
        self.log_file_path = self.output_dir / "batch_run.log"
        self.summary_file_path = self.output_dir / "summary.txt"

        # Results tracking
        self.config_files = []
        self.results = []  # List of True/False for each config
        self.failed = []  # List of failed config paths
        self.failed_details = {}  # Map: config_path -> error_message

    # ------------------------------------------------------------------------
    # Find config files
    # ------------------------------------------------------------------------

    def find_configs(self):
        """Find all YAML config files in the directory."""
        yaml_files = sorted(self.config_dir.glob("**/*.yaml"))
        # Filter out README files
        self.config_files = [f for f in yaml_files if not f.name.startswith("README")]
        return len(self.config_files) > 0

    # ------------------------------------------------------------------------
    # Display preview of what will run
    # ------------------------------------------------------------------------

    def display_preview(self):
        """Show user what configs will be run."""
        print(f"\nFound {len(self.config_files)} config file(s) in {self.config_dir}")
        print("\nConfigs to run:")
        for i, cfg in enumerate(self.config_files, 1):
            try:
                rel_path = cfg.relative_to(self.config_dir)
            except ValueError:
                rel_path = cfg.name
            print(f"  {i:2d}. {rel_path}")

        if self.dry_run:
            print("\n[DRY RUN MODE - No configs will actually be executed]")
        else:
            print("\n[BATCH MODE - Will continue on errors by default]")

    # ------------------------------------------------------------------------
    # Ask for user confirmation
    # ------------------------------------------------------------------------

    def confirm_execution(self):
        """
        Ask user to confirm batch run (unless dry-run).

        Returns:
            True if should proceed, False otherwise
        """
        if self.dry_run:
            return True

        try:
            response = input(
                f"\nProceed with running {len(self.config_files)} configs? [y/N]: "
            )
            return response.lower() in ["y", "yes"]
        except KeyboardInterrupt:
            print("\nAborted.")
            return False

    # ------------------------------------------------------------------------
    # Setup output directory
    # ------------------------------------------------------------------------

    def setup_output(self):
        """Create output directory and display paths."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nBatch run logs will be saved to: {self.output_dir}")
        print(f"  - Full log: {self.log_file_path.name}")
        print(f"  - Summary: {self.summary_file_path.name}")
        print(
            "\nNote: Individual experiment results go to their configured output_dir\n"
        )

    # ------------------------------------------------------------------------
    # Run a single config file
    # ------------------------------------------------------------------------

    def run_single_config(
        self, config_path: Path, log_file: TextIO
    ) -> tuple[bool, str]:
        """
        Run a single config file.

        Args:
            config_path: Path to the config file
            log_file: Open log file handle

        Returns:
            Tuple of (success: bool, error_message: str or None)
        """
        cmd = [sys.executable, "-m", "cf_bench.cli", "--config", str(config_path)]

        if self.debug:
            cmd.append("--debug")

        try:
            rel_path = config_path.relative_to(Path.cwd())
        except ValueError:
            rel_path = config_path

        # Print and log header
        header = f"\n{'=' * 80}\nRunning: {rel_path}\n{'=' * 80}"
        print(header)
        log_file.write(header + "\n")
        log_file.flush()

        if self.dry_run:
            msg = f"[DRY RUN] Command: {' '.join(cmd)}"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()
            return True, None

        try:
            # Run the config with streaming output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
            )

            # Stream output in real-time
            for line in process.stdout:
                log_file.write(line)
                print(line, end="")
                log_file.flush()

            # Wait for process to complete
            return_code = process.wait()

            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)

            success_msg = f"[SUCCESS] {config_path.name}"
            print(success_msg)
            log_file.write(success_msg + "\n")
            log_file.flush()
            return True, None

        except subprocess.CalledProcessError as e:
            error_msg = f"[FAILED] {config_path.name}\n  Error code: {e.returncode}"
            print(error_msg)
            log_file.write(error_msg + "\n")
            if e.stdout:
                log_file.write("STDOUT:\n" + e.stdout + "\n")
            if e.stderr:
                log_file.write("STDERR:\n" + e.stderr + "\n")
            log_file.flush()
            return False, f"Exit code {e.returncode}"

        except Exception as e:
            error_msg = f"[ERROR] {config_path.name}\n  {type(e).__name__}: {e}"
            print(error_msg)
            log_file.write(error_msg + "\n")
            log_file.flush()
            return False, f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------------
    # Execute all configs
    # ------------------------------------------------------------------------

    def run_all_configs(self):
        """Execute all config files and track results."""
        with open(self.log_file_path, "w", encoding="utf-8") as log_file:
            # Write log header
            log_file.write(f"Batch Run Started: {self.timestamp}\n")
            log_file.write(f"Directory: {self.config_dir}\n")
            log_file.write(f"Total configs: {len(self.config_files)}\n")
            log_file.write("=" * 80 + "\n\n")
            log_file.flush()

            # Run each config
            for i, config_path in enumerate(self.config_files, 1):
                progress = f"\n[{i}/{len(self.config_files)}]"
                print(progress)
                log_file.write(progress + "\n")
                log_file.flush()

                success, error_msg = self.run_single_config(config_path, log_file)
                self.results.append(success)

                if not success:
                    self.failed.append(config_path)
                    self.failed_details[config_path] = error_msg

                    if self.stop_on_error and not self.dry_run:
                        stop_msg = (
                            "\nStopping due to failure (--stop-on-error flag set)."
                        )
                        print(stop_msg)
                        log_file.write(stop_msg + "\n")
                        break

    # ------------------------------------------------------------------------
    # Generate summary report
    # ------------------------------------------------------------------------

    def generate_summary(self) -> str:
        """
        Generate the summary report text.

        Returns:
            Summary text as string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("BATCH RUN SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {self.timestamp}")
        lines.append(f"Config Directory: {self.config_dir}")
        lines.append("")

        if self.dry_run:
            lines.append(
                f"Dry run completed. {len(self.config_files)} config(s) would be executed."
            )
        else:
            successful = sum(self.results)
            total = len(self.results)
            lines.append(f"Completed: {successful}/{total} successful")
            lines.append("")

            # Failed configs
            if self.failed:
                lines.append(f"Failed configs ({len(self.failed)}):")
                for cfg in self.failed:
                    try:
                        rel_path = cfg.relative_to(self.config_dir)
                    except ValueError:
                        rel_path = cfg.name
                    error_detail = self.failed_details.get(cfg, "Unknown error")
                    lines.append(f"  [X] {rel_path}")
                    lines.append(f"      Reason: {error_detail}")
            else:
                lines.append("[SUCCESS] All configs completed successfully!")

            # Successful configs
            lines.append("")
            lines.append("Successful configs:")
            for cfg, success in zip(self.config_files, self.results):
                if success:
                    try:
                        rel_path = cfg.relative_to(self.config_dir)
                    except ValueError:
                        rel_path = cfg.name
                    lines.append(f"  [OK] {rel_path}")

        lines.append("=" * 80)
        lines.append(f"\nFull log: {self.log_file_path}")
        lines.append(f"Summary: {self.summary_file_path}")

        return "\n".join(lines)

    # ------------------------------------------------------------------------
    # Save and display summary
    # ------------------------------------------------------------------------

    def save_summary(self):
        """Generate, save, and display the summary report."""
        summary_text = self.generate_summary()

        # Display to console
        print("\n" + summary_text)

        # Save to file
        with open(self.summary_file_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        print(f"\n[OK] Logs saved to: {self.output_dir}\n")

    # ------------------------------------------------------------------------
    # Main execution flow
    # ------------------------------------------------------------------------

    def run(self):
        """
        Execute the complete batch run workflow.

        Returns:
            Exit code (0 for success, 1 if any configs failed)
        """
        # Validate directory
        if not self.config_dir.exists():
            print(f"Error: Directory not found: {self.config_dir}", file=sys.stderr)
            return 1

        if not self.config_dir.is_dir():
            print(f"Error: Not a directory: {self.config_dir}", file=sys.stderr)
            return 1

        # Find configs
        if not self.find_configs():
            print(f"No YAML config files found in {self.config_dir}", file=sys.stderr)
            return 1

        # Show what will be run
        self.display_preview()

        # Get confirmation
        if not self.confirm_execution():
            print("Aborted.")
            return 0

        # Setup output
        self.setup_output()

        # Execute all configs
        self.run_all_configs()

        # Generate and save summary
        self.save_summary()

        # Return exit code
        return 1 if (self.failed and not self.dry_run) else 0


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Parse arguments and run the config batch runner."""
    parser = argparse.ArgumentParser(
        description="Batch run all YAML configs in a directory (continues on errors by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all configs in predictors_vs_threshold
  python scripts/batch_config_runner.py configs/predictors_vs_threshold

  # Run only baseline configs
  python scripts/batch_config_runner.py configs/predictors_vs_threshold/baseline

  # Dry run to see what would be executed
  python scripts/batch_config_runner.py configs/predictors_vs_threshold --dry-run

  # Run with debug output
  python scripts/batch_config_runner.py configs/predictors_vs_threshold --debug

  # Stop on first failure (not recommended for batch runs)
  python scripts/batch_config_runner.py configs/predictors_vs_threshold --stop-on-error
        """,
    )

    parser.add_argument(
        "directory", type=Path, help="Directory containing YAML config files to run"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output for each run"
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop execution on first failure (default: continue on errors)",
    )

    args = parser.parse_args()

    # Create and run the config batch runner
    runner = ConfigBatchRunner(args)
    exit_code = runner.run()
    sys.exit(exit_code)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

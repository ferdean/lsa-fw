#!/dolfinx-env/bin/python3

"""MPI performance analysis for cube.py: measures wall-clock time and memory usage."""

import argparse
import logging
import pathlib
import time
from subprocess import PIPE

import psutil
import csv
import json

from lib.loggingutils import log_global, setup_logging

logger = logging.getLogger("mpi-analysis")
setup_logging(verbose=True, output_path=pathlib.Path(".") / "log_parallel.log")

MPI_EXEC = "mpirun"
MPI_FLAG = "-n"
INTERPRETER = "/dolfinx-env/bin/python3"
SCRIPT = "/workspaces/lsa-fw/.examples/cube.py"
RESULTS_FILE = pathlib.Path(__file__).parent / "data" / "mpi_analysis.csv"


def _parse_args():
    p = argparse.ArgumentParser(description="MPI performance analysis for cube.py")
    p.add_argument(
        "--cores",
        nargs="+",
        type=int,
        default=list(range(1, 13)),
        help="MPI core counts to test (default: 1-12).",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Number of runs per core count (default: 20).",
    )
    return p.parse_args()


def sample_tree_memory(proc: psutil.Popen) -> tuple[int, int]:
    """Aggregate RSS and VMS (bytes) of the MPI tree (parent + children).

    - Resident Set Size (RSS) is the portion of the process' memory that is held in RAM. It shoes the amount of
    physical RAM that the MPI processes actually occupy.
    - Virtual Memory Size (VMS) is the total address space that the process has reserved, including mapped files,
    libraries, heap, stack, etc.
    """
    try:
        root = psutil.Process(proc.pid)
        processes = [root] + root.children(recursive=True)
        rss = sum(p.memory_info().rss for p in processes)
        vms = sum(p.memory_info().vms for p in processes)
        return rss, vms
    except psutil.Error:
        return 0, 0


args = _parse_args()

# Write header if first time
if not RESULTS_FILE.exists():
    with RESULTS_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "cores",
                "run",
                "time_ns",
                "time_mesh_gen_ns",
                "time_spaces_def_ns",
                "time_bcs_def_ns",
                "time_baseflow_compute_ns",
                "time_assemble_ns",
                "peak_rss_MB",
                "peak_vms_MB",
            ]
        )

for n in args.cores:
    for run_idx in range(1, args.repeats + 1):
        log_global(
            logger,
            logging.INFO,
            "Starting run %d/%d with %d core(s)",
            run_idx,
            args.repeats,
            n,
        )
        cmd = [MPI_EXEC, MPI_FLAG, str(n), INTERPRETER, SCRIPT]

        # Launch MPI run under psutil to monitor memory
        proc = psutil.Popen(
            cmd,
            stdout=PIPE,
            stderr=PIPE,
            text=True,
        )
        start_ns = time.perf_counter_ns()
        peak_rss = peak_vms = 0

        try:
            # Poll until completion, sampling memory every 0.2 seconds
            while True:
                try:
                    proc.wait(timeout=0.2)
                    stdout, _ = proc.communicate()
                    if stdout is None:
                        raise RuntimeError("MPI run produced no stdout")
                    stage_times = json.loads(stdout.strip())
                    rss, vms = sample_tree_memory(proc)
                    peak_rss = max(peak_rss, rss)
                    peak_vms = max(peak_vms, vms)
                    break
                except psutil.TimeoutExpired:
                    rss, vms = sample_tree_memory(proc)
                    peak_rss = max(peak_rss, rss)
                    peak_vms = max(peak_vms, vms)
        except Exception as e:
            log_global(logger, logging.ERROR, "Monitoring failed: %s", str(e))
            continue

        elapsed_ns = time.perf_counter_ns() - start_ns

        # Convert bytes to megabytes
        peak_rss_mb = peak_rss / (1024**2)
        peak_vms_mb = peak_vms / (1024**2)

        # Append results
        with RESULTS_FILE.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    n,
                    run_idx,
                    elapsed_ns,
                    stage_times["mesh_gen_ns"],
                    stage_times["spaces_def_ns"],
                    stage_times["bcs_def_ns"],
                    stage_times["baseflow_compute_ns"],
                    stage_times["assemble_ns"],
                    f"{peak_rss_mb:.1f}",
                    f"{peak_vms_mb:.1f}",
                ]
            )

        log_global(
            logger,
            logging.INFO,
            "Completed run %d/%d with %d core(s): time=%d ns, peak_rss=%.1f MB, peak_vms=%.1f MB",
            run_idx,
            args.repeats,
            n,
            elapsed_ns,
            peak_rss_mb,
            peak_vms_mb,
        )

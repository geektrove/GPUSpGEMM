#!/usr/bin/env python3

import multiprocessing
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from urllib.request import urlretrieve

import invoke
import numpy as np
import pandas as pd
import scipy.io

# Path to the directory containing the matrices
DATA_DIR = Path("data")

# Template for the binary path and target names
BIN_TEMPLATE = "build/{preset}/src/apps/{target}"
TARGETS = ["proposal", "opsparse", "cusparse1", "cusparse2"]

# List of representative matrices
MATRICES = [
    "JGD_Homology/m133-b3",
    "Williams/webbase-1M",
    "Williams/mc2depi",
    "GHS_indef/mario002",
    "QLi/majorbasis",
    "DNVS/shipsec1",
    "Bova/rma10",
    "Williams/pdb1HYS",
    "Chen/pkustk12",
    "PARSEC/SiO2",
    "QY/case39",
    "Gupta/gupta3",
]


@invoke.task
def download_matrices(
    _: invoke.Context,
    force: bool = False,
    threads: int = multiprocessing.cpu_count() * 2,
):
    BASE_URL = "https://sparse.tamu.edu/mat"

    def download_matrix(url: str, force: bool) -> None:
        mat_name = Path(url).stem
        csr_filename = DATA_DIR / f"{mat_name}.csr"

        # Skip if the file already exists and we are not forcing a download
        if not force and csr_filename.exists():
            return

        # Download the matrix
        mat_filename, _ = urlretrieve(url)

        # Load the matrix from the .mat file
        mat = scipy.io.loadmat(mat_filename)
        csr = mat["Problem"]["A"][0][0].tocsr()
        with suppress(Exception):
            zeros = mat["Problem"]["Zeros"][0][0].tocsr()
            csr += zeros

        # Save the matrix to a custom .csr binary format
        with csr_filename.open("wb") as fp:
            header = np.array([csr.nnz, csr.shape[0], csr.shape[1]], dtype=np.int32)
            header.tofile(fp)
            csr.indptr.tofile(fp)
            csr.indices.tofile(fp)
            csr.data.tofile(fp)

        # Remove the .mat file
        Path(mat_filename).unlink()

        print(f"Downloaded {url} to {csr_filename}")

    # Create the data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download the matrices in parallel
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(download_matrix, f"{BASE_URL}/{matrix}.mat", force)
            for matrix in MATRICES
        ]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error downloading {future}: {e}")
    print("Download complete")


@invoke.task
def benchmark(
    context: invoke.Context,
    target: str = "proposal",
    output: Path | None = None,
    runs: int = 10,
    warmups: int = 10,
    pause: int = 50,
):
    RUN_PATTERN = re.compile(
        r"Run\s+(?P<id>\d+): (?P<time>.+)ns [|] (?P<flops>\d+[.]\d*) FLOPS"
    )
    BIN = BIN_TEMPLATE.format(preset="release", target=target)
    FORMAT_WIDTH = 7

    # Get the matrix names and filenames
    matrix_names = [matrix.split("/")[-1] for matrix in MATRICES]
    filenames = [DATA_DIR / f"{name}.csr" for name in matrix_names]
    matrix_column_width = max(len(name) for name in matrix_names)

    # Print the header
    print(f"Algorithm: {target}")
    print(f"Runs: {runs}")
    print(f"Warmups: {warmups}")
    print(f"Pause: {pause}")
    print("Unit: us")
    print(
        f"{'Matrix':{matrix_column_width}s}"
        f" | {'Mean':^{FORMAT_WIDTH}s} (+- {'Std':^{FORMAT_WIDTH}s})"
        f" | {'GFLOPS':^{FORMAT_WIDTH}s}"
    )

    stats = []
    for i in range(len(matrix_names)):
        # Run the benchmark
        result = context.run(
            f"{BIN} {filenames[i]} {filenames[i]}"
            f" benchmark --runs={runs} --warmups={warmups} --pause={pause}",
            hide=True,
            warn=True,
        )

        # Parse the output
        times = []
        flops = []
        for line in result.stdout.splitlines():
            if not line.startswith("Run"):
                continue
            match = RUN_PATTERN.fullmatch(line)
            if not match:
                print("Wrong output format")
                print(result.stdout)
                print(result.stderr)
                return
            times.append(float(match.group("time")))
            flops.append(float(match.group("flops")))
        if len(times) == 0:
            # Algorithm failed to run for the given matrix
            for i in range(runs):
                times.append(0)
                flops.append(0)
        assert len(times) == runs
        assert len(flops) == runs
        times = np.array(times)
        flops = np.array(flops)

        # Save the results if requested
        if output is not None:
            for i in range(runs):
                stats.append(
                    {
                        "algorithm": target,
                        "matrix": matrix_names[i],
                        "time_ns": times[i],
                        "flops": flops[i],
                    }
                )

        # Convert the time to microseconds and compute the GFLOPS
        times = times / 1000
        gflops = flops / 1e9

        # Print the results
        mean = np.mean(times)
        std = np.std(times)
        print(
            f"{matrix_names[i]:{matrix_column_width}s}"
            f" | {mean:{FORMAT_WIDTH}.0f} (+- {std:{FORMAT_WIDTH}.0f})"
            f" | {gflops.mean():{FORMAT_WIDTH}.2f}"
        )

    # Save the results to a CSV file if requested
    if output is not None:
        stats = pd.DataFrame(stats)
        stats.to_csv(output, index=False)
        print(f"Results saved to {output}")


@invoke.task
def run_fitcluster(
    _: invoke.Context,
    results_dir: Path = Path("results/fitcluster"),
    postfix: str = "",
):
    SBATCH_TEMPLATE = """
#!/bin/bash

#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/{target}.%j.out
#SBATCH --error=logs/{target}.%j.err

inv benchmark --target={target} --output={output}
""".strip()
    results_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        output = results_dir / f"{target}-{postfix}.csv"
        script = SBATCH_TEMPLATE.format(target=target, output=output)
        subprocess.run("sbatch", input=script, text=True, check=True)


@invoke.task
def run_fitstar(
    context: invoke.Context,
    results_dir: Path = Path("results/fitstar"),
    postfix: str = "",
):
    results_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        output = results_dir / f"{target}-{postfix}.csv"
        context.run(f"inv benchmark --target={target} --output={output}")

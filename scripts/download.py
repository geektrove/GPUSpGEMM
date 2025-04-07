#!/usr/bin/env python3

import logging
import multiprocessing
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlretrieve
from contextlib import suppress

import numpy as np
import scipy.io

BASE_URL = "https://sparse.tamu.edu/mat"
MATRICES = [
    "JGD_Homology/m133-b3",
    "Williams/mac_econ_fwd500",
    "Pajek/patents_main",
    "Williams/webbase-1M",
    "Williams/mc2depi",
    "Hamm/scircuit",
    "GHS_indef/mario002",
    "vanHeukelum/cage12",
    "QLi/majorbasis",
    "Um/offshore",
    "Um/2cubes_sphere",
    "FEMLAB/poisson3Da",
    "Oberwolfach/filter3D",
    # "FreeFieldTechnologies/mono_500Hz",
    # "QCD/conf5_4-8x8-05",
    "Williams/cant",
    "Williams/consph",
    "DNVS/shipsec1",
    "Bova/rma10",
    "DIMACS10/delaunay_n24",
    "vanHeukelum/cage15",
    "Gleich/wb-edu",
    "Williams/cop20k_A",
    "GHS_psdef/hood",
    "Boeing/pwtk",
    "Williams/pdb1HYS",
]
URLS = [f"{BASE_URL}/{name}.mat" for name in MATRICES]

logger = logging.getLogger(__name__)


def download_matrix(url: str, output_dir: Path, force: bool) -> None:
    mat_name = Path(url).stem
    csr_filename = output_dir / f"{mat_name}.csr"
    if not force and csr_filename.exists():
        logger.info(f"CSR file {csr_filename} already exists, skipping download")
        return

    mat_filename, _ = urlretrieve(url)
    logger.info(f"Downloaded {url} to {mat_filename}")

    mat = scipy.io.loadmat(mat_filename)
    csr = mat["Problem"]["A"][0][0].tocsr()
    with suppress(Exception):
        zeros = mat["Problem"]["Zeros"][0][0].tocsr()
        csr += zeros
    logger.info(f"Converted {mat_name} to CSR format")

    with csr_filename.open("wb") as fp:
        header = np.array([csr.nnz, csr.shape[0], csr.shape[1]], dtype=np.int32)
        header.tofile(fp)
        csr.indptr.tofile(fp)
        csr.indices.tofile(fp)
        csr.data.tofile(fp)
    logger.info(f"Saved CSR matrix to {csr_filename}")

    Path(mat_filename).unlink()
    logger.info(f"Deleted temporary file {mat_filename}")


def main():
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser(description="Download and convert matrices to CSR format")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/"),
        help="Directory to save matrices",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=multiprocessing.cpu_count() * 2,
        help="Number of threads to use for downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    threads: int = args.threads
    force: bool = args.force

    output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(download_matrix, url, output_dir, force) for url in URLS
        ]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error downloading {future}: {e}")
    logger.info("Download complete")


if __name__ == "__main__":
    main()

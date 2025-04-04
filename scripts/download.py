#!/usr/bin/env python3

import logging
import multiprocessing
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import scipy.io

URLS = [
    "https://sparse.tamu.edu/mat/Williams/webbase-1M.mat",
    "https://sparse.tamu.edu/mat/Williams/pdb1HYS.mat",
    "https://sparse.tamu.edu/mat/Williams/consph.mat",
    "https://sparse.tamu.edu/mat/Williams/cant.mat",
    "https://sparse.tamu.edu/mat/Williams/mac_econ_fwd500.mat",
    "https://sparse.tamu.edu/mat/Williams/mc2depi.mat",
    "https://sparse.tamu.edu/mat/Williams/cop20k_A.mat",
]

logger = logging.getLogger(__name__)


def download_matrix(url: str, output_dir: Path) -> None:
    mat_name = Path(url).stem

    mat_filename, _ = urlretrieve(url)
    logger.info(f"Downloaded {url} to {mat_filename}")

    mat = scipy.io.loadmat(mat_filename)
    csr = mat["Problem"]["A"][0][0].tocsr()
    logger.info(f"Converted {mat_name} to CSR format")

    csr_filename = output_dir / f"{mat_name}.csr"
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
    args = parser.parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    threads: int = args.threads

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(download_matrix, url, output_dir) for url in URLS]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error downloading {future}: {e}")
    logger.info("Download complete")


if __name__ == "__main__":
    main()

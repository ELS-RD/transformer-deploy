import logging
from contextlib import contextmanager
from time import time
from typing import List
import numpy as np


def print_timings(name: str, timings: List[float]):
    mean_time = 1e3 * np.mean(timings)
    std_time = 1e3 * np.std(timings)
    min_time = 1e3 * np.min(timings)
    max_time = 1e3 * np.max(timings)
    median, percent_95_time, percent_99_time = 1e3 * np.percentile(timings, [50, 95, 99])
    logging.info(
        f"timing [{name}]: "
        f"mean={mean_time:.2f}ms, "
        f"sd={std_time:.2f}ms, "
        f"min={min_time:.2f}ms, "
        f"max={max_time:.2f}ms, "
        f"median={median:.2f}ms, "
        f"95p={percent_95_time:.2f}ms, "
        f"99p={percent_99_time:.2f}ms"
    )


def setup_logging():
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)


@contextmanager
def track_infer_time(buffer: [int]):
    start = time()
    yield
    end = time()
    buffer.append(end - start)

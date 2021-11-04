import logging
from contextlib import contextmanager
import time
from typing import List
import numpy as np


def print_timings(name: str, timings: List[float]):
    mean_time = np.mean(timings) / 1e6
    std_time = np.std(timings) / 1e6
    min_time = np.min(timings) / 1e6
    max_time = np.max(timings) / 1e6
    median, percent_95_time, percent_99_time = np.percentile(timings, [50, 95, 99]) / 1e6
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
    start = time.perf_counter_ns()
    yield
    end = time.perf_counter_ns()
    buffer.append(end - start)

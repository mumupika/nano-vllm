from math import nan
from time import perf_counter_ns

class Observer():
    # start and end.
    ttft_start = 0
    ttft_end = 0
    tpot_start = 0
    tpot_end = 0
    # interval.
    ttft = 0
    tpot = 0
    
    @classmethod
    def reset_ttft(cls):
        cls.ttft_start = 0
        cls.ttft_end = 0
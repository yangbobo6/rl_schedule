from typing import Tuple, Dict

class PhysicalQubit:
    """代表一个物理量子比特及其所有属性"""

    def __init__(self, q_id: Tuple[int, int], t1: float, t2: float, f_1q: float, f_ro: float):
        self.id = q_id
        # 静态属性
        self.t1 = t1
        self.t2 = t2
        self.fidelity_1q = f_1q
        self.fidelity_readout = f_ro
        self.connectivity = {}  # 存储与邻居的连接信息
        # 动态属性
        self.booking_schedule = []  # 存储 (start_time, end_time)

    def add_link(self, neighbor_id: Tuple[int, int], f_2q: float, crosstalk: float):
        self.connectivity[neighbor_id] = {'fidelity_2q': f_2q, 'crosstalk_coeff': crosstalk}

    def get_next_available_time(self) -> float:
        if not self.booking_schedule:
            return 0.0
        return self.booking_schedule[-1][1]  # 假设预定是按时间排序的

    def reset(self):
        self.booking_schedule = []

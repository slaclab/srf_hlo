import dataclasses
import unittest
from typing import Optional, List

from lcls_tools.common.controls.pyepics.utils import PV
from lcls_tools.superconducting.sc_linac import Cavity, Linac, Machine
from lcls_tools.superconducting.sc_linac_utils import LINAC_CM_MAP
from scipy.optimize import minimize


class HLOCavity(Cavity):
    def __init__(self, cavity_num, rack_object):
        super().__init__(cavity_num, rack_object)
        self.q0_pv: str = self.pv_addr("Q0")
        self._q0_pv_obj: Optional[PV] = None

    @property
    def q0(self):
        if not self._q0_pv_obj:
            self._q0_pv_obj = PV(self.q0_pv)
        return self._q0_pv_obj.get()

    def heat(self, amplitude: float) -> float:
        r_over_q = 1012
        return ((amplitude * 1e6) ** 2) / (r_over_q * self.q0)

    @property
    def current_heat(self):
        return self.heat(self.acon)

    @dataclasses.dataclass
    class Bounds:
        lower: float
        upper: float

        @property
        def tuple(self):
            return self.lower, self.upper

    @property
    def bounds(self):
        if not self.is_online:
            return self.Bounds(lower=0, upper=0)
        else:
            return self.Bounds(lower=0, upper=self.ades_max)


class HLOLinac(Linac):
    def __init__(
        self,
        linac_section,
        beamline_vacuum_infixes,
        insulating_vacuum_cryomodules,
        machine,
    ):
        super().__init__(
            linac_section,
            beamline_vacuum_infixes,
            insulating_vacuum_cryomodules,
            machine,
        )

        self.cavities: List[HLOCavity] = []
        for cm_obj in self.cryomodules.values():
            for cavity_obj in cm_obj.cavities.values():
                self.cavities.append(cavity_obj)

    def current_heat(self):
        heat = 0
        for cavity in self.cavities:
            heat += cavity.current_heat
        return heat

    def cost(self, amplitudes: List[float]):
        cost = 0
        for cavity, amplitude in zip(self.cavities, amplitudes):
            cost += cavity.heat(amplitude)
        return cost

    def solution(self, desired_mv: float):
        def constraint(amplitudes: List[float]):
            return sum(amplitudes) - desired_mv

        return minimize(
            fun=self.cost,
            x0=[desired_mv / len(self.cavities) for _ in self.cavities],
            constraints={"type": "eq", "fun": constraint},
            bounds=((cavity.bounds.tuple for cavity in self.cavities)),
        )


HLO_MACHINE = Machine(cavity_class=HLOCavity, linac_class=HLOLinac)


class TestSolution(unittest.TestCase):
    def test_linac(self, idx):
        linac: HLOLinac = HLO_MACHINE.linacs[idx]
        print(f"\033[96mL{idx} current heat: {linac.current_heat()}\033[0m")
        solution = linac.solution(PV(f"ACCL:L{idx}B:1:AACTMEANSUM").get())
        print(solution)
        self.assertEqual(len(solution.x), len(LINAC_CM_MAP[idx]) * 8)
        self.assertTrue(solution.fun <= linac.current_heat())

        for cavity_obj, proposed_amp in zip(linac.cavities, solution.x):
            if round(cavity_obj.acon, 2) != round(proposed_amp, 2):
                print(
                    f"\033[96m{cavity_obj} currently at {cavity_obj.acon}, proposing {proposed_amp}\033[0m"
                )

    def test_l0(self):
        self.test_linac(0)

    def test_l1(self):
        self.test_linac(1)

    def test_l2(self):
        self.test_linac(2)

    def test_l3(self):
        self.test_linac(3)


if __name__ == "__main__":
    unittest.main()

import dataclasses
import unittest
from typing import Optional, List

from lcls_tools.common.controls.pyepics.utils import PV
from lcls_tools.superconducting.sc_linac import Cavity, Linac, Machine
from lcls_tools.superconducting.sc_linac_utils import L2B, L1B, L1BHL
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
    def test_l1(self):
        l1: HLOLinac = HLO_MACHINE.linacs[1]
        print(f"L1 current heat: {l1.current_heat()}")
        solution = l1.solution(PV("ACCL:L2B:1:AACTMEANSUM").get())
        print(solution)
        self.assertEqual(len(solution.x), (len(L1B) + len(L1BHL)) * 8)
        self.assertTrue(solution.fun <= l1.current_heat())

        for cavity_obj, proposed_amp in zip(l1.cavities, solution.x):
            if cavity_obj.acon != proposed_amp:
                print(
                    f"{cavity_obj} currently at {cavity_obj.acon}, proposing {proposed_amp}"
                )

    def test_l2(self):
        l2: HLOLinac = HLO_MACHINE.linacs[2]
        print(f"L2 current heat: {l2.current_heat()}")
        solution = l2.solution(PV("ACCL:L2B:1:AACTMEANSUM").get())
        print(solution)
        self.assertEqual(len(solution.x), len(L2B) * 8)
        self.assertTrue(solution.fun <= l2.current_heat())

        for cavity_obj, proposed_amp in zip(l2.cavities, solution.x):
            if cavity_obj.acon != proposed_amp:
                print(
                    f"{cavity_obj} currently at {cavity_obj.acon}, proposing {proposed_amp}"
                )


if __name__ == "__main__":
    unittest.main()

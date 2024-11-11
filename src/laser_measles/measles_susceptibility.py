import numba as nb
import numpy as np
from matplotlib import pyplot as plt


class Susceptibility:
    def __init__(self, model, verbose: bool = False):
        self.__name__ = "susceptibility"
        self.model = model

        return

    def __call__(self, model, tick):
        return

    @staticmethod
    def on_birth(model, _tick, istart, iend):
        # newborns are _not_ susceptible
        set_susceptibility(istart, iend, model.population.susceptibility, 0)

        return

    def plot(self) -> None:
        fig = plt.figure(figsize=(12, 9), dpi=128)
        fig.suptitle("Susceptibility Distribution By Age")
        age_bins = (self.model.params.nticks - self.model.population.dob[0 : self.model.population.count]) // 365
        sus_counts = np.bincount(age_bins, weights=self.model.population.susceptibility[0 : self.model.population.count].astype(np.uint32))
        age_counts = np.bincount(age_bins)
        plt.bar(range(len(age_counts)), age_counts)
        plt.bar(range(len(sus_counts)), sus_counts, alpha=0.5)

        mgr = plt.get_current_fig_manager()
        mgr.full_screen_toggle()

        plt.show()

        return


def setup_susceptibility(model, verbose: bool = False) -> None:
    model.population.add_scalar_property("susceptibility", dtype=np.uint8, default=1)
    initialize_susceptibility(model.population.count, model.population.dob, model.population.susceptibility)

    return Susceptibility(model, verbose)


@nb.njit((nb.uint32, nb.int32[:], nb.uint8[:]), parallel=True, cache=True)
def initialize_susceptibility(count, dob, susceptibility) -> None:
    five_years_ago = -5 * 365
    for i in nb.prange(count):
        # 5 y.o. and older are _not_ susceptible (dobs are negative)
        susceptibility[i] = 0 if dob[i] < five_years_ago else 1

    return


@nb.njit((nb.uint32, nb.uint32, nb.uint8[:], nb.uint8), parallel=True, cache=True)
def set_susceptibility(istart, iend, susceptibility, value) -> None:
    for i in nb.prange(istart, iend):
        susceptibility[i] = value

    return

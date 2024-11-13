import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class MaternalAntibodies:
    def __init__(self, model, verbose: bool = False) -> None:
        self.__name__ = "maternal_antibodies"
        self.model = model

        # TODO - initialize existing agents with maternal antibodies
        model.population.add_scalar_property("ma_timer", np.uint8)  # Use uint8 for timer since 6 months ~ 180 days < 2^8

        return

    def __call__(self, model, tick) -> None:
        nb_update_ma_timers(model.population.count, model.population.ma_timer, model.population.susceptibility)
        return

    @staticmethod
    def on_birth(model, _tick, istart, iend) -> None:
        model.population.susceptibility[istart:iend] = 0  # newborns are _not_ susceptible due to maternal antibodies
        model.population.ma_timer[istart:iend] = int(6 * 365 / 12)  # 6 months in days
        return

    def plot(self, fig: Figure = None) -> None:
        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        cinfants = ((self.model.params.nticks - self.model.population.dob[0 : self.model.population.count]) < 365).sum()
        cwith = (self.model.population.ma_timer[0 : self.model.population.count] > 0).sum()
        cwithout = cinfants - cwith

        fig.suptitle(f"Maternal Antibodies for Infants (< 1 year)\n{cinfants:,} Infants")
        plt.pie([cwithout, cwith], labels=[f"Infants w/out Antibodies {cwithout:,}", f"Infants w/Maternal Antibodies {cwith:,}"])

        return


@nb.njit((nb.uint32, nb.uint8[:], nb.uint8[:]), parallel=True, cache=True)
def nb_update_ma_timers(count, ma_timers, susceptibility):
    for i in nb.prange(count):
        timer = ma_timers[i]
        if timer > 0:
            timer -= 1
            ma_timers[i] = timer
            if timer == 0:
                susceptibility[i] = 1

    return

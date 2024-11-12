import numba as nb
import numpy as np
from matplotlib import pyplot as plt


class Infection:
    def __init__(self, model, verbose: bool = False) -> None:
        self.__name__ = "infection"
        self.model = model

        model.population.add_scalar_property("itimer", dtype=np.uint8, default=0)
        Infection.nb_set_itimers(0, model.population.count, model.population.itimer, 0)

        return

    def __call__(self, model, tick) -> None:
        Infection.nb_infection_update(model.population.count, model.population.itimer)
        return

    @staticmethod
    @nb.njit((nb.uint32, nb.uint8[:]), parallel=True, cache=True)
    def nb_infection_update(count, itimers):
        for i in nb.prange(count):
            itimer = itimers[i]
            if itimer > 0:
                itimer -= 1
                itimers[i] = itimer

        return

    @staticmethod
    def on_birth(model, _tick, istart, iend) -> None:
        Infection.nb_set_itimers(istart, iend, model.population.itimer, 0)
        return

    @staticmethod
    @nb.njit((nb.uint32, nb.uint32, nb.uint8[:], nb.uint8), parallel=True, cache=True)
    def nb_set_itimers(istart, iend, itimers, value) -> None:
        for i in nb.prange(istart, iend):
            itimers[i] = value

        return

    def plot(self) -> None:
        fig = plt.figure(figsize=(12, 9), dpi=128)
        fig.suptitle("Infections By Age")

        ages_in_years = (self.model.params.nticks - self.model.population.dob[0 : self.model.population.count]) // 365
        age_counts = np.bincount(ages_in_years)
        plt.bar(range(len(age_counts)), age_counts)
        itimers = self.model.population.itimer[0 : self.model.population.count]
        infected = itimers > 0
        infection_counts = np.bincount(ages_in_years[infected])
        plt.bar(range(len(infection_counts)), infection_counts)

        mgr = plt.get_current_fig_manager()
        mgr.full_screen_toggle()

        plt.show()

        return

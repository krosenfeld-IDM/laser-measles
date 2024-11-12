import numba as nb
import numpy as np
from matplotlib import pyplot as plt


class Incubation:
    def __init__(self, model, verbose: bool = False) -> None:
        self.__name__ = "incubation"
        self.model = model

        model.population.add_scalar_property("etimer", dtype=np.uint8, default=0)
        # TODO - verify itimer property on population since we use it when etimer hits 0
        # model.population.add_scalar_property("itimer", dtype=np.uint8, default=0)

        return

    def __call__(self, model, tick) -> None:
        Incubation.nb_update_exposure_timers(
            model.population.count, model.population.etimer, model.population.itimer, model.params.inf_mean, model.params.inf_std
        )
        return

    @staticmethod
    @nb.njit((nb.uint32, nb.uint8[:], nb.uint8[:], nb.float32, nb.float32), parallel=True, cache=True)
    def nb_update_exposure_timers(count, etimers, itimers, inf_mean, inf_std) -> None:
        for i in nb.prange(count):
            timer = etimers[i]
            if timer > 0:
                timer -= 1
                etimers[i] = timer
                if timer == 0:
                    itimers[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.normal(inf_mean, inf_std))))

        return

    @staticmethod
    def on_birth(model, _tick, istart, iend) -> None:
        # newborns are _not_ incubating
        Incubation.nb_set_etimers(istart, iend, model.population.etimer, 0)

        return

    @staticmethod
    @nb.njit((nb.uint32, nb.uint32, nb.uint8[:], nb.uint8), parallel=True, cache=True)
    def nb_set_etimers(istart, iend, incubation, value) -> None:
        for i in nb.prange(istart, iend):
            incubation[i] = value

        return

    def plot(self) -> None:
        fig = plt.figure(figsize=(12, 9), dpi=128)
        fig.suptitle("Incubation Period Distribution")

        etimers = self.model.population.etimer[0 : self.model.population.count]
        incubating = etimers > 0
        incubation_counts = np.bincount(etimers[incubating])
        plt.bar(range(len(incubation_counts)), incubation_counts)

        mgr = plt.get_current_fig_manager()
        mgr.full_screen_toggle()

        plt.show()

        return

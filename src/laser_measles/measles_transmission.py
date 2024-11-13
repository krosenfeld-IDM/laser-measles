import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class Transmission:
    def __init__(self, model, verbose: bool = False) -> None:
        self.__name__ = "transmission"
        self.model = model

        model.patches.add_vector_property("cases", length=model.params.nticks, dtype=np.uint32)
        model.patches.add_scalar_property("forces", dtype=np.float32)
        model.patches.add_vector_property("incidence", model.params.nticks, dtype=np.uint32)

        return

    def __call__(self, model, tick) -> None:
        patches = model.patches
        population = model.population

        contagion = patches.cases[tick, :]  # we will accumulate current infections into this view into the cases array
        nodeids = population.nodeid[0 : population.count]  # just look at the active agent indices
        itimers = population.itimer[0 : population.count]  # just look at the active agent indices
        np.add.at(contagion, nodeids[itimers > 0], 1)  # increment by the number of active agents with non-zero itimer

        network = patches.network
        transfer = (contagion * network).round().astype(np.uint32)
        contagion += transfer.sum(axis=1)  # increment by incoming "migration"
        contagion -= transfer.sum(axis=0)  # decrement by outgoing "migration"

        forces = patches.forces
        beta_effective = model.params.beta + model.params.seasonality_factor * np.sin(
            2 * np.pi * (tick - model.params.seasonality_phase) / 365
        )
        np.multiply(contagion, beta_effective, out=forces)
        np.divide(forces, model.patches.populations[tick, :], out=forces)  # per agent force of infection

        Transmission.nb_transmission_update(
            population.susceptibility,
            population.nodeid,
            forces,
            population.etimer,
            population.count,
            model.params.exp_shape,
            model.params.exp_scale,
            model.patches.incidence[tick, :],
        )

        return

    @staticmethod
    @nb.njit(
        (nb.uint8[:], nb.uint16[:], nb.float32[:], nb.uint8[:], nb.uint32, nb.float32, nb.float32, nb.uint32[:]),
        parallel=True,
        nogil=True,
        cache=True,
    )
    def nb_transmission_update(susceptibilities, nodeids, forces, etimers, count, exp_shape, exp_scale, incidence):
        for i in nb.prange(count):
            susceptibility = susceptibilities[i]
            if susceptibility > 0:
                nodeid = nodeids[i]
                force = susceptibility * forces[nodeid]  # force of infection attenuated by personal susceptibility
                if (force > 0) and (np.random.random_sample() < force):  # draw random number < force means infection
                    susceptibilities[i] = 0.0  # no longer susceptible
                    # set exposure timer for newly infected individuals to a draw from a gamma distribution, must be at least 1 day
                    etimers[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.gamma(exp_shape, exp_scale))))

                    incidence[nodeid] += 1

        return

    def plot(self, fig: Figure = None) -> None:
        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        fig.suptitle("Cases and Incidence for Two Largest Patches")

        itwo, ione = np.argsort(self.model.patches.populations[-1, :])[-2:]

        fig.add_subplot(2, 2, 1)
        plt.title(f"Cases - Node {ione}")  # ({self.names[ione]})")
        plt.plot(self.model.patches.cases[:, ione])

        fig.add_subplot(2, 2, 2)
        plt.title(f"Incidence - Node {ione}")  # ({self.names[ione]})")
        plt.plot(self.model.patches.incidence[:, ione])

        fig.add_subplot(2, 2, 3)
        plt.title(f"Cases - Node {itwo}")  # ({self.names[itwo]})")
        plt.plot(self.model.patches.cases[:, itwo])

        fig.add_subplot(2, 2, 4)
        plt.title(f"Incidence - Node {itwo}")  # ({self.names[itwo]})")
        plt.plot(self.model.patches.incidence[:, itwo])

        return

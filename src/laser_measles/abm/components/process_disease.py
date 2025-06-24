from matplotlib.figure import Figure
import numba as nb
import numpy as np
from pydantic import BaseModel, Field
from laser_measles.base import BaseComponent

@nb.njit((nb.uint32, nb.uint16[:], nb.uint16[:], nb.uint8[:], nb.float32, nb.float32, nb.uint32[:], nb.uint16[:]), parallel=True, cache=True)
def nb_gamma_update(count, etimers, itimers, state, mean, shape, flow, nodeid):  # pragma: no cover
    """Numba compiled function to check and update exposed timers for the population in parallel."""
    max_node_id = np.max(nodeid) + 1
    thread_flow = np.zeros((nb.config.NUMBA_DEFAULT_NUM_THREADS, max_node_id), dtype=np.uint32)

    for i in nb.prange(count):
        etimer = etimers[i]
        if etimer > 0:
            etimer -= 1
            # if we have decremented etimer from >0 to <=0, set infectious timer.
            if etimer <= 0:
                scale = mean / shape
                itimers[i] = np.maximum(np.uint16(1), np.uint16(np.ceil(np.random.gamma(shape, scale))))
                thread_flow[nb.get_thread_id(), nodeid[i]] += 1
                state[i] = 2
            etimers[i] = etimer

    flow[:] += thread_flow.sum(axis=0)

    return

@nb.njit((nb.uint32, nb.uint16[:], nb.uint8[:], nb.uint8), parallel=True, cache=True)
def nb_state_update(count, itimers, state, new_state):  # pragma: no cover
    """Numba compiled function to check and update infection timers for the population in parallel."""
    for i in nb.prange(count):
        itimer = itimers[i]
        if itimer > 0:
            itimer -= 1
            if itimer == 0:
                state[i] = new_state
            itimers[i] = itimer

    return

class DiseaseParams(BaseModel):
    inf_mean: float = Field(default=8.0, description="Mean infectious period")
    inf_sigma: float = Field(default=2.0, description="Shape of the infectious period")

    @property
    def inf_shape(self) -> float:
        return self.inf_sigma ** 2 / self.inf_mean
    
    @property
    def inf_scale(self) -> float:
        return self.inf_mean / self.inf_shape

class DiseaseProcess(BaseComponent):
    def __init__(self, model, verbose: bool = False, params: DiseaseParams | None = None):
        super().__init__(model, verbose)
        self.params = params if params is not None else DiseaseParams()

    def __call__(self, model, tick: int) -> None:

        # Update the infectious timers S=0, E=1, I=2, R=3
        nb_state_update(model.population.count, model.population.itimer, model.population.state, np.uint8(3)) # TODO, capture the state in an ENUM?

        # Update the exposure timers for the population in the model, 
        # move to infectious which follows a gamma distribution
        inf_flow = np.zeros(len(model.patches), dtype=np.uint32)
        nb_gamma_update(model.population.count, model.population.etimer, model.population.itimer, model.population.state,
                        self.params.inf_mean, self.params.inf_shape, inf_flow, model.population.nodeid)

        return
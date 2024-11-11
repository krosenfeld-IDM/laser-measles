import numba as nb
import numpy as np
from matplotlib import pyplot as plt


class RoutineImmunization:
    def __init__(self, model, verbose: bool = False):
        self.__name__ = "routine_immunization"
        self.model = model

        # Coverage by patch
        model.patches.add_scalar_property("ri_coverage", dtype=np.float32)
        # TODO - make this more robust
        # Coverage ranges around the specified parameter
        model.patches.ri_coverage[:] = model.prng.poisson(model.params.ri_coverage * 100, model.patches.count) / 100
        # Agents get an MCV status - 0 for unvaccinated _or ineffective vaccination_, 1 for effective MCV1, 2 for effective MCV2
        model.population.add_scalar_property("mcv", dtype=np.uint8)
        model.population.add_scalar_property("ri_timer", dtype=np.uint16)  # Use uint16 for timer since 15 months = ~450 > 2^8

        # TODO - initialize existing agents with MCV status and ri_timer

        return

    def __call__(self, model, tick):
        nb_update_ri_timers(model.population.count, model.population.ri_timer, model.population.susceptibility)
        return

    @staticmethod
    def on_birth(model, _tick, istart, iend):
        set_mcv_status(model, istart, iend)
        set_mcv_timers(model, istart, iend)

        return

    def plot(self):
        fig = plt.figure(figsize=(12, 9), dpi=128)

        population = self.model.population
        indices = population.dob[0 : population.count] > 0
        cindividuals = indices.sum()
        mcv = population.mcv[0 : population.count]  # just active agents
        cunvaccinated = (mcv[indices] == GET_NONE).sum()
        cmcv1 = (mcv[indices] == GET_MCV1).sum()
        cmcv2 = (mcv[indices] == GET_MCV2).sum()

        assert (
            cindividuals == cunvaccinated + cmcv1 + cmcv2
        ), f"Mismatch in MCV status counts:\n{cindividuals=:,} != {cunvaccinated=:,} + {cmcv1=:,} + {cmcv2=:,}"

        fig.suptitle(f"Routine Immunization\n{cindividuals:,} individuals")
        pct_none = 100 * cunvaccinated / cindividuals
        pct_mcv1 = 100 * cmcv1 / cindividuals
        pct_mcv2 = 100 * cmcv2 / cindividuals
        plt.pie(
            [cunvaccinated, cmcv1, cmcv2],
            labels=[
                f"Unvaccinated {cunvaccinated:,} ({pct_none:.1f}%)",
                f"MCV1 {cmcv1:,} ({pct_mcv1:.1f}%)",
                f"MCV2 {cmcv2:,} ({pct_mcv2:.1f}%)",
            ],
        )

        mgr = plt.get_current_fig_manager()
        mgr.full_screen_toggle()

        plt.show()

        return


def setup_routine_immunization(model, verbose: bool = False):
    routine_immunization = RoutineImmunization(model, verbose)

    return routine_immunization


@nb.njit((nb.uint32, nb.uint16[:], nb.uint8[:]), parallel=True)
def nb_update_ri_timers(count, ri_timers, susceptibility):
    for i in nb.prange(count):
        timer = ri_timers[i]
        if timer > 0:
            timer -= 1
            ri_timers[i] = timer
            if timer == 0:
                # When timer expires, vaccinated agents become immune
                susceptibility[i] = 0

    return


GET_MCV1 = 1
GET_MCV2 = 2
GET_NONE = 0


def set_mcv_status(model, istart, iend):
    mcv1_cutoff = model.patches.ri_coverage * model.params.probability_mcv1_take  # probability of (MCV1 vaccination) _and_ (MCV1 take)
    mcv2_cutoff = (
        mcv1_cutoff + model.patches.ri_coverage * (1.0 - model.params.probability_mcv1_take) * model.params.probability_mcv2_take
    )  # probability of (MCV1 vaccination) _and_ (not MCV1 take) and (MCV2 take)

    draws = model.prng.random(size=(iend - istart))
    nodeids = model.population.nodeid[istart:iend]
    get_mcv1 = draws <= mcv1_cutoff[nodeids]
    get_mcv2 = (draws > mcv1_cutoff[nodeids]) & (draws <= mcv2_cutoff[nodeids])
    # get_none = (draws > mcv2_cutoff[nodeids]) # "get_none" is the default
    mcv = model.population.mcv[istart:iend]
    mcv[get_mcv1] = GET_MCV1
    mcv[get_mcv2] = GET_MCV2

    return


def set_mcv_timers(model, istart, iend):
    count = iend - istart
    ri_timer_values_mcv1 = model.prng.integers(model.params.mcv1_start, model.params.mcv1_end, count).astype(
        model.population.ri_timer.dtype
    )
    ri_timer_values_mcv2 = model.prng.integers(model.params.mcv2_start, model.params.mcv2_end, count).astype(
        model.population.ri_timer.dtype
    )

    mcv = model.population.mcv[istart:iend]

    mask_mcv1 = mcv == GET_MCV1
    mask_mcv2 = mcv == GET_MCV2
    # mask_none = mcv == GET_NONE  # for validation
    # if mask_mcv1.sum() == 0:
    #     raise ValueError("Didn't find anyone with mcv set to GET_MCV1")
    # if mask_mcv2.sum() == 0:
    #     raise ValueError("Didn't find anyone with mcv set to GET_MCV2")
    # if mask_none.sum() == 0:
    #     raise ValueError("Didn't find anyone with mcv set to GET_NONE")

    timers = model.population.ri_timer[istart:iend]
    timers[mask_mcv1] = ri_timer_values_mcv1[mask_mcv1]
    timers[mask_mcv2] = ri_timer_values_mcv2[mask_mcv2]

    return

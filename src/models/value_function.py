import numpy as np
import matplotlib.pyplot as plt

def generate_policy(probs, max_period):
    num_choices = 2

    max_streak = max_period
    num_states = max_streak + 1

    ## Get array of choices and possible states
    max_streaks = np.array(range(0, num_states))
    streaks = np.array(range(0, num_states))
    choices = np.array(range(0, num_choices))
    probs_plus_1 = np.concatenate((probs , np.array([1])))

    ## Calculate matrices
    choices_mat, streaks_mat, max_streaks_mat, probs_mat = np.meshgrid(choices, streaks, max_streaks, probs)

    max_less_than_current = (max_streaks_mat < streaks_mat)

    ##### Current Streak Updating
    ## If choice is to skip, then states are just the same as what they were
    streaks_stay_mat = streaks_mat[:, 0, :, :]

    ## If choice is to take the risk, then potential state update is current streak increases by 1
    streaks_win_mat = np.minimum(streaks_mat[:, 1, :, :] + 1, max_period)

    ## The risk though is current streak goes to zero
    streaks_lose_mat = np.zeros((num_states, num_states, num_probs), dtype='int')


    ###### Max streak Updating
    ## If choice is to skip, then states are just the same as what they were
    max_of_max_current = np.maximum(max_streaks_mat[:, 0, :, :], streaks_mat[:, 0, :, :])
    max_streaks_stay_mat = max_of_max_current

    ## If choice is to take the risk, then potential state update is increased if current streak == max streak
    max_of_max_winstreak = np.maximum(max_streaks_mat[:, 1, :, :], streaks_mat[:, 1, :, :] + 1)
    max_streaks_win_mat = np.minimum(max_of_max_winstreak , max_period)

    max_streaks_lose_mat = max_of_max_current

    #### Probs updating
    probs_new_mat = np.tile(range(0, num_probs), (num_states, num_states, 1))

    Opts = {}
    V_funcs = {}
    G_funcs = {}
    Cutoffs = {}

    V_funcs['V' + str(max_period)] = max_of_max_current

    for period in range(max_period-1, -1, -1):
        next = period + 1
        next_V = V_funcs['V' + str(next)]

        Exp = np.zeros((num_states, num_choices, num_states, num_probs))

        Exp_V_stay = np.mean(next_V[streaks_stay_mat, max_streaks_stay_mat, probs_new_mat], axis=2)
        Exp_V_win = np.mean(next_V[streaks_win_mat, max_streaks_win_mat, probs_new_mat], axis=2)
        Exp_V_lose = np.mean(next_V[streaks_lose_mat, max_streaks_lose_mat, probs_new_mat], axis=2)

        Exp[:, 0, :, :] = np.tile(Exp_V_stay, (num_probs, 1, 1)).transpose([1, 2, 0])
        Exp[:, 1, :, :] = (
            np.einsum("ij,k->ijk", Exp_V_win, probs_mat[0, 1, 0, :]) +
            np.einsum("ij,k->ijk", Exp_V_lose, (1 - probs_mat[0, 1, 0, :]))
        )

        G_funcs['G' + str(period)] = np.array(np.argmax(Exp, axis=1), dtype=float)
        G_funcs['G' + str(period)][next:, :, :] = np.nan
        G_funcs['G' + str(period)][:, next:, :] = np.nan
        V_funcs['V' + str(period)] = np.array(np.amax(Exp, axis=1), dtype=float)
        V_funcs['V' + str(period)][next:, :] = np.nan
        V_funcs['V' + str(period)][:, next:, :] = np.nan

    return V_funcs, G_funcs

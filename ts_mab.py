import requests
import pandas as pd
from datetime import datetime, timedelta
import math
import random


class MAB:
    def __init__(self, id_lst, open_lst, send_lst):

        if len(id_lst) != len(open_lst) or len(id_lst) != len(send_lst):
            raise AssertionError("Length of id_lst, open_lst and send_lst should be equal")

        if len(id_lst) == 0 or len(open_lst) == 0 or len(send_lst) == 0:
            raise AssertionError("Length of id_lst, open_lst and send_lst should be greater than 0")

        self.id_lst = id_lst
        self.arm_lst = []
        self.open_lst = open_lst
        self.send_lst = send_lst
        for i in range(len(id_lst)):
            alpha = math.pow(math.floor(open_lst[i] / 2), 0.7)
            beta = math.pow(max(math.floor((send_lst[i] - open_lst[i]) / 2), 0), 0.7)
            self.arm_lst.append(Arm(id_lst[i], alpha, beta))

    def get_alloc(self, n_samples):

        if type(n_samples) != int:
            raise AssertionError("Number of samples should be an integer")

        if n_samples <= 100:
            raise AssertionError("Number of samples should be greater than 100")

        sample_dict = {}
        for arm in self.arm_lst:
            cur_id = arm.id
            sample_lst = arm.sample_from_arm(n_samples)
            sample_dict[cur_id] = sample_lst

        successes_dict = {}
        for arm in self.arm_lst:
            successes_dict[arm.id] = 0

        for i in range(n_samples):
            max_val = -999
            max_id = -1
            for arm in self.arm_lst:
                cur_id = arm.id
                cur_val = sample_dict[cur_id][i]
                if cur_val > max_val:
                    max_val = cur_val
                    max_id = cur_id
            successes_dict[max_id] += 1

        successes_lst = []
        for id in successes_dict:
            successes_dict[id] /= n_samples
            successes_lst.append(successes_dict[id])

        return successes_lst


def sample_from_gamma(shape, scale):
    sumall = 0
    for i in range(math.floor(shape)):
        sumall -= math.log(1-random.random())
    return sumall * scale

def sample_from_beta(alpha, beta):
    gamma_alpha = sample_from_gamma(alpha, 1)
    gamma_beta = sample_from_gamma(beta, 1)
    return gamma_alpha/(gamma_beta + gamma_alpha)
    # return np.random.beta(alpha, beta)

class Arm:
    def __init__(self, id, alpha, beta):
        self.id = id
        self.alpha = alpha
        self.beta = beta

    def sample_from_arm(self, n_samples):

        sample_lst = []
        for i in range(n_samples):
            sample_lst.append(sample_from_beta(self.alpha, self.beta))

        return sample_lst


def normalize_array(array):
    total = sum(array)
    scaling_factor = 100 / total
    normalized_array = []

    for element in array:
        scaled_element = round(element * scaling_factor)
        rounded_element = round(scaled_element / 5) * 5
        normalized_array.append(rounded_element)
    diff = sum(normalized_array) - 100
    diff_lst = [array[i] - normalized_array[i] for i in range(len(array))]
    last_used_index = set()
    if diff > 0:
        while diff > 0:
            max_diff_elements = [i for i, val in enumerate(diff_lst) if val == max(diff_lst)]
            max_index = random.choice(max_diff_elements)
            last_used_index.add(normalized_array[max_index])
            while (normalized_array[max_index] == 0 and max_diff_elements) or normalized_array[
                max_index] not in last_used_index:
                diff_lst.pop(max_index)
                max_diff_elements = [i for i, val in enumerate(diff_lst) if abs(val) == max(abs(v) for v in diff_lst)]
                if max_diff_elements:
                    max_index = random.choice(max_diff_elements)
                    last_used_index.add(normalized_array[max_index])
            normalized_array[max_index] -= 5
            diff -= 5
            diff_lst = [array[i] - normalized_array[i] for i in range(len(array))]
    elif diff < 0:
        while diff < 0:
            min_diff_elements = [i for i, val in enumerate(diff_lst) if val == min(diff_lst)]
            min_index = random.choice(min_diff_elements)
            normalized_array[min_index] += 5
            diff += 5
            diff_lst = [array[i] - normalized_array[i] for i in range(len(array))]
    return normalized_array


def mab(ids_arr, sends_arr, opens_arr, n_sample=1000):
    # dist_rate_dict, mab_share = assign_uniformed_distribution(ids_arr, sends_arr)
    # for key in dist_rate_dict
    bandit = MAB(ids_arr, opens_arr, sends_arr)
    dist_rate = bandit.get_alloc(n_sample)
    return dist_rate


def run_ts_mab(sends_arr, opens_arr):
    dist_dict = {}
    id_dict = {}
    mab_id_arr = []
    mab_sends_arr = []
    mab_opens_arr = []
    new_id = 0
    for i in range(len(sends_arr)):
        if i >= 5:
            dist_dict[i] = 0
            continue
        if sends_arr[i] < 100:
            dist_dict[i] = 100/min(len(sends_arr), 5)
        else:
            id_dict[new_id] = i
            mab_id_arr.append(new_id)
            mab_sends_arr.append(sends_arr[i])
            mab_opens_arr.append(opens_arr[i])
            new_id += 1

    if len(mab_id_arr) > 0:
        mab_dist_rate = mab(mab_id_arr, mab_sends_arr, mab_opens_arr)
        mab_dist_sum = 100 - sum(dist_dict.values())
        # combine with uniform dist
        for j in mab_id_arr:
            dist_dict[id_dict[j]] = mab_dist_rate[j] * mab_dist_sum

    comb_dist = []
    # print(dist_dict)
    for i in range(len(sends_arr)):
        comb_dist.append(dist_dict[i])
    adj_dist_rate = normalize_array(comb_dist)
    if sum(adj_dist_rate) != 100:
        print("*******Erro*******")
    return adj_dist_rate
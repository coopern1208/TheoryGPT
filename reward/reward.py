from grammar.state import GrammarState
import math
import grammar.vocab as vocab   
import torch

def log_reward(value: int, bonus: int = 5) -> float:
    reward = - math.log(1 + abs(value))
    return reward + bonus if value == 0 else reward

def square_threshold_reward(value: int, threshold: int = 2) -> float:
    if value <= 1e-10: value = -10 # to avoid log(0)
    else: value = math.log10(value)
    reward = -abs(value-threshold)
    return reward if value <= threshold else 0
    
def compute_scalar_mass(phi2_coeff, phi4_coeff, acquire_vev: bool = False):
    if acquire_vev: return (phi2_coeff**2 * float(phi4_coeff[6:]))**0.5
    else: return float(phi2_coeff[5:])

def check_length(model: dict, alpha = 5, max_length = 512, max_free_param_num = 35):
    reward = 0
    length = len(model["sequence"])
    if length > 300: 
        reward += -alpha * (length - 300) / (max_length - 300)
    if len(model["params"]) > 10:
        reward += -alpha * (len(model["params"]) - 10) / (max_free_param_num - 10)
    return reward

def check_ptcl_mass(model: dict, threshold: int = 100):
    multiplets = model["multiplets"]
    interactions = model["interactions"]
    free_params = model["params"]

    reward = 0
    light_charged_exotics = {}
    massless_exotics = {}
    # -------- Fermion Mass --------
    for mplt_id, mplt in multiplets.items():
        if mplt["type"] == "CSCALAR":
            for interaction_id, interaction in interactions.items():
                if interaction["type"] == "TERM_PHI2" and interaction["mplt_list"][0] == mplt_id:
                    phi2_coeff = free_params[interaction["param_list"][0]]
                elif interaction["type"] == "TERM_PHI4" and interaction["mplt_list"][0] == mplt_id:
                    phi4_coeff = free_params[interaction["param_list"][0]]
            mass = compute_scalar_mass(phi2_coeff, phi4_coeff, acquire_vev = mplt["vev_id"])
            mplt["mplt_mass_list"]["mass_0"] = mass
            if mass < threshold: reward += square_threshold_reward(mass, math.log10(threshold))

        elif mplt["type"] == "FERMION":
            for charge, ptcls in mplt["mplt_mass_list"].items():
                if not ptcls:
                    mplt["mplt_mass_list"][charge] = ["mass_0"] * int(mplt["gen"][4:])
                if not (charge == "charge_0" and mplt['rep_list'][0] == "singlet"):
                    for ptcl in ptcls:
                        if ptcl in free_params:
                            mass = float(free_params[ptcl][5:])
                            if mass < threshold:
                                light_charged_exotics[ptcl] = mass
                                reward += square_threshold_reward(mass, math.log10(threshold))
                        elif ptcl == "mass_0":
                            massless_exotics[mplt_id] = int(mplt["gen"][4:])
                            reward += square_threshold_reward(0, math.log10(threshold))
    return reward/10

# check anomalies reward
def check_anomalies(model: dict):
    anomalies = model["anomalies"]
    reward = 0
    for key, value in anomalies.items():
        reward += log_reward(value)
    return (reward - 25)/10


def all_rewards(model: dict):    
    if model["too_long"]: 
        return {"length_reward": check_length(model), 
                "anomalies_reward": 0, 
                "light_exotics_reward": 0, 
                "total_reward": check_length(model)} 
    else: 
        length_reward = check_length(model)
        anomalies_reward = check_anomalies(model)
        light_exotics_reward = check_ptcl_mass(model)
        total_reward = length_reward + anomalies_reward + light_exotics_reward

        return {"length_reward": length_reward, 
                "anomalies_reward": anomalies_reward, 
                "light_exotics_reward": light_exotics_reward, 
                "total_reward": total_reward}


if __name__ == "__main__":
    print(compute_scalar_mass(246, "param_0.30", acquire_vev = True))

    
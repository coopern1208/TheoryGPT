from config import config
from grammar.vocab import HYPERCHARGES

# get allowed charge 
def get_allowed_charge(rep_list: list[str]) -> list[str]:
    _, SU2_rep, U1_rep =  rep_list
    Y = int(U1_rep.split("_")[1])/6 
    SU2_T3_map = {"singlet": [0],
                 "fnd": [1/2, -1/2],
                 "adj": [1, 0, -1]
                 }
    T3 = SU2_T3_map.get(SU2_rep, [])
    Q = [round((t3 + Y) * 6) for t3 in T3]

    if any(q > config.max_charge or q < -config.max_charge for q in Q): return []
    else: return [f"charge_{q}" for q in Q]

# get SU3 representation
def get_SU3_rep(particle_type: str, chirality: str, particles: dict) -> list[str]:
    candidate_reps = []
    particle_dict = {}
    if particle_type == "FERMION":
        particle_dict = particles["FERMION"][chirality]
    else:
        particle_dict = particles[particle_type]
    for color, charge_dict in particle_dict.items():
        for charge, p in charge_dict.items():
            if p.get("num_ptcls", 0) > 0: 
                candidate_reps.append("fnd" if color == "COLOR" else "singlet")
    if candidate_reps: return list(set(candidate_reps))
    else: return ["singlet", "fnd"]

# get SU2 representation
def get_SU2_rep(chirality) -> list[str]:
    if chirality == "LEFT": return ["fnd"]
    elif chirality == "RIGHT": return ["singlet"]
    else: return ["singlet", "fnd"]

# get hypercharge
# def get_allowed_hypercharge(Q_list, SU2_rep):
#     Q_list = [int(Q.split("_")[1])/6 for Q in Q_list]
#     T3_map = {
#         "singlet": [0],
#         "fnd": [1/2, -1/2],
#         "adj": [1, 0, -1]}
#     allowed_Y_list = []
#     for T3 in T3_map[SU2_rep]:
#         Y_list = [f"hypercharge_{int(round((Q - T3)*6))}" for Q in Q_list]
#         for Y in Y_list:
#             rep_list = ["SU3_rep", SU2_rep, Y]
#             allowed_charge = get_allowed_charge(rep_list)
#             if allowed_charge: allowed_Y_list.append(Y)
#     return list(set(allowed_Y_list))

# get target charge
def get_target_charge(multiplet_type, chirality, particles, rep_list):
    target_charges = []
    color = "NO_COLOR" if rep_list[0] == "singlet" else "COLOR"

    if multiplet_type == "FERMION": 
        particle_dict = particles["FERMION"][chirality][color]
    else: 
        particle_dict = particles[multiplet_type][color]    

    target_charges.extend(
        int(charge[7:])
        for charge, p in particle_dict.items()
        if p.get("num_ptcls", 0) > 0
    )
    return target_charges

def get_hypercharge(multiplet_type, chirality, rep_list, particles, multiplets):
    existing_rep_lists = [(value["chirality"], value["rep_list"]) for value in multiplets.values() if "rep_list" in value and value["type"] == multiplet_type]
    charge_list = get_target_charge(multiplet_type, chirality, particles, rep_list)
    SU2_rep = rep_list[1]
    SU3_rep = rep_list[0]
    T3_map = {"singlet": [0], "fnd": [1/2, -1/2], "adj": [1, 0, -1]}
    allowed_Y_list = []
    if not charge_list: 
        for Y in HYPERCHARGES:
            rep_list = [SU3_rep, SU2_rep, Y]
            allowed_charge = get_allowed_charge(rep_list)
            if allowed_charge and Y not in allowed_Y_list: # and rep_list not in existing_rep_lists: 
                if multiplet_type == "FERMION" and (chirality, rep_list) not in existing_rep_lists: 
                    allowed_Y_list.append(Y)
                elif multiplet_type == "CSCALAR" or multiplet_type == "RSCALAR":
                    allowed_Y_list.append(Y)
    else:
        for charge in charge_list:
            for T3 in T3_map[SU2_rep]:
                Y = f"hypercharge_{int(round((charge/6 - T3)*6))}"
                rep_list = [SU3_rep, SU2_rep, Y]
                allowed_charge = get_allowed_charge(rep_list)
                if allowed_charge and Y not in allowed_Y_list: # and rep_list not in existing_rep_lists: 
                    if multiplet_type == "FERMION" and (chirality, rep_list) not in existing_rep_lists: 
                        allowed_Y_list.append(Y)
                    elif multiplet_type == "CSCALAR" or multiplet_type == "RSCALAR":
                        allowed_Y_list.append(Y)
    return allowed_Y_list
    



    
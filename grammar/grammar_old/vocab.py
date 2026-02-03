from config import config

# Gauge Group Vocab
GROUP_TYPES = ["GAUGE_U", "GAUGE_SU"]
GROUP_IDS = [f"g_{i+1}" for i in range(config.max_gauge_groups)]
GROUP_RANKS = [f"rank_{i+1}" for i in range(config.max_group_rank)]

# VEV Vocab
VEV_TOKENS = ["VEV", "END_VEV"]
VEV_IDS = [f"v_{i+1}" for i in range(config.max_vevs)]
SM_VEV = ["SM_VEV"]
VEC_TOKENS = ["VEC", "END_VEC"]

# Particle Vocab
PARTICLE_TYPES = ["PTCL_FERMION", "PTCL_CSCALAR", "PTCL_RSCALAR"]
COLORS = ["COLOR", "NO_COLOR"]
CHARGES = ["charge_0"] + [f"charge_{i}" for i in range(-config.max_charge, config.max_charge+1)]
NUM_PTCLS = [f"NUM_{i}" for i in range(1, config.max_gen+1)]
SM_CSCALAR_TAGS = ["SM_Hp", "SM_H0"]
SM_FERMION_TAGS = ["SM_E", "SM_MU", "SM_TAU", "SM_VE", "SM_VM", "SM_VT", "SM_U", "SM_C", "SM_D", "SM_S", "SM_B"]
TAGS = SM_CSCALAR_TAGS + SM_FERMION_TAGS 

SM_TAGS_DICT = {
    'CSCALAR': {'COLOR': {}, 'NO_COLOR': {}},
    'RSCALAR': {'COLOR': {}, 'NO_COLOR': {}},
    'FERMION': {'LEFT': {'COLOR': {'charge_4': ['SM_U', 'SM_C'], 'charge_-2': ['SM_D', 'SM_S', 'SM_B']},
                         'NO_COLOR': {'charge_-6': ['SM_E', 'SM_MU', 'SM_TAU'], 'charge_0': ['SM_VE', 'SM_VM', 'SM_VT']}},
                'RIGHT': {'COLOR': {'charge_4': ['SM_U', 'SM_C'], 'charge_-2': ['SM_D', 'SM_S', 'SM_B']},
                          'NO_COLOR': {'charge_-6': ['SM_E', 'SM_MU', 'SM_TAU']}}}
                }

SM_TAGS_MASS = {"SM_E": "mass_1e-4", 
                "SM_MU": "mass_1e-1", 
                "SM_TAU": "mass_1e0", 
                "SM_VE": "mass_0", 
                "SM_VM": "mass_0", 
                "SM_VT": "mass_0", 
                "SM_U": "mass_1e-3", 
                "SM_C": "mass_1e0", 
                "SM_D": "mass_1e-3", 
                "SM_S": "mass_1e-2", 
                "SM_B": "mass_1e0", 
                "SM_Hp": "mass_1e2", 
                "SM_H0": "mass_1e2"}

# Multiplet Vocab
MULTIPLET_TYPES = ["MPLT_CSCALAR", "MPLT_RSCALAR", "MPLT_FERMION"]
MULTIPLET_IDS = [f"m_{i+1}" for i in range(config.max_multiplets)]
CHIRALITIES = ["NULL", "LEFT", "RIGHT"]
REPRESENTATIONS = ["singlet", "fnd"]
HYPERCHARGES = ["hypercharge_0"] + [f"hypercharge_{i}" for i in range(-config.max_hypercharge, config.max_hypercharge+1)]
GENS = [f"gen_{i}" for i in range(1, config.max_gen+1)]
DIMS = [f"dim_{i}" for i in range(1, config.max_dim+1)]
#Gauge Anomaly Tokens
ANOMALY_COEFF = ["ZERO", "POS_SMALL", "POS_BIG", "NEG_SMALL", "NEG_BIG"]

# Interaction Vocab
INTERACTION_TYPES = ["TERM_YUKAWA", "TERM_PHI4", "TERM_PHI3", "TERM_PHI2"]
INTERACTION_IDS = [f"i_{i+1}" for i in range(config.max_interactions)]
PARAMETERS = [f"param_1e{i}" for i in range(-config.max_value_exp, config.max_value_exp+1)]
MASS= ["mass_0"] + [f"mass_1e{i}" for i in range(-config.max_mass_exp, config.max_mass_exp+1)]

GRAMMAR_TOKENS = [
    "BOS", "EOS", "PAD",

    # Gauge Group Tokens
    "GAUGE_GROUP_BLOCK",
    *GROUP_IDS,
    *GROUP_TYPES,
    *GROUP_RANKS,
    "END_GAUGE",

    # SSB Tokens
    "SSB_BLOCK",
    "VEV",
    *VEV_IDS,
    "SM_VEV",
    "VEC",
    "0", "1",
    "END_VEC",
    "END_VEV",

    # Particle Tokens
    "PARTICLE_BLOCK",
    *PARTICLE_TYPES,
    *CHARGES,
    *TAGS,
    *COLORS,
    *NUM_PTCLS,
    "END_PTCL",
    "END_PARTICLE_BLOCK",

    # Multiplet Tokens
    "MULTIPLET_BLOCK",
    "ACQUIRE",
    *MULTIPLET_TYPES,
    *MULTIPLET_IDS,
    *CHIRALITIES,
    *GENS,
    *DIMS,
    *REPRESENTATIONS,
    *HYPERCHARGES,
    "ANOMALIES", "END_ANOMALIES",
    *ANOMALY_COEFF,
    "REPS", "END_REPS",
    "END_MULTIPLET", 

    # Interaction Tokens
    "INTERACTION_BLOCK",
    *INTERACTION_TYPES,
    *INTERACTION_IDS,
    *PARAMETERS,
    *MASS,
    "MPLTS", "END_MPLT",
    "PARAMS", "END_PARAM",
    "END_INTERACTION"

]

token2id = {token: i for i, token in enumerate[str](GRAMMAR_TOKENS)}
id2token = {i: token for i, token in enumerate(GRAMMAR_TOKENS)}

PAD_TOKEN_ID = token2id["PAD"]
BOS_TOKEN_ID = token2id["BOS"]
EOS_TOKEN_ID = token2id["EOS"]

def encode(sequence: list[str]) -> list[int]:
    return [token2id[token] for token in sequence]

def decode(ids: list[int]) -> list[str]:
    return [id2token[id] for id in ids]

if __name__ == "__main__":
    print(BOS_TOKEN_ID)
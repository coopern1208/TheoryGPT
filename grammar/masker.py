import copy
from fractions import Fraction
from typing import Dict, List, Any, Set, Optional, Tuple

from grammar.state import GrammarState
from grammar.PDG import PDG_IDS
from grammar import utils
import grammar.vocab as vocab
import grammar.groups.representation as rep
import grammar.groups.anomaly as anomaly
import grammar.name_convention as nc
import grammar.interactions.yukawa as term_yukawa 
import grammar.interactions.phi4 as term_phi4
import grammar.interactions.phi3 as term_phi3
import grammar.interactions.phi2 as term_phi2

from config import config

class GrammarMasker:
    def __init__(self):
        self.token2id = vocab.token2id
        self.id2token = vocab.id2token
        self.vocab_size = len(vocab.GRAMMAR_TOKENS)
        self.GROUP_RANK_MAP = {"GAUGE_SU": ["rank_2", "rank_3"],  "GAUGE_U": ["rank_1"]}
        self.SU2_DIM_MAP = {"singlet": "dim_1", "fnd": "dim_2", "adj": "dim_3"}
        self.INTERACTION_MAP = {"TERM_YUKAWA": 3, "TERM_PHI4": 1, "TERM_PHI3": 1, "TERM_PHI2": 1}

    def init_state(self) -> GrammarState:
        return GrammarState()
        
    def step(self, state: GrammarState, token: str) -> GrammarState: 

        # ======================== GAUGE BLOCK ======================== 
        if token == "GAUGE_GROUP_BLOCK": state.current_block = "GAUGE"
        elif token in vocab.GROUP_TYPES: state.group_type = token
        elif token in vocab.GROUP_IDS: state.group_id = token
        elif token in vocab.GROUP_RANKS: state.group_rank = token
        elif token == "END_GAUGE": 
            state.gauge_groups[state.group_id] = {
                "id": state.group_id,
                "type": state.group_type,
                "rank": state.group_rank
            }
            state.group_id = None
            state.group_type = None
            state.group_rank = None

        # ======================== SSB BLOCK ======================== 
        elif token == "SSB_BLOCK": state.current_block = "SSB"
        elif token in vocab.VEV_IDS and state.current_block == "SSB": state.vev_id = token 
        elif token in (*vocab.REPRESENTATIONS, *vocab.HYPERCHARGES): state.rep_list.append(token)
        elif token == "END_REPS" and state.current_block == "SSB": 
            state.charge_list = rep.get_allowed_charge(state.rep_list)
        elif token in vocab.DIMS: state.dim = int(token[4:])
        elif token in ["0", "1"]: state.vev_vector.append(int(token))
        elif token == "END_VEV": 
            state.vevs[state.vev_id] = {
                "id": state.vev_id,
                "vector": state.vev_vector,
                "rep_list": state.rep_list,
                "charge_list": state.charge_list,
                "dim": state.dim,
                "multiplets": None
            }
            state.vev_opts.append(state.vev_id)
            state.vev_id = None
            state.vev_vector = []
            state.rep_list = []
            state.dim = 0


        # ======================== PARTICLE BLOCK ========================
        elif token == "PARTICLE_BLOCK": state.current_block = "PARTICLE"
        elif token in vocab.PARTICLE_TYPES: state.particle_type = token[5:]
        elif token in vocab.CHIRALITIES: state.chirality = token
        elif token in vocab.COLORS and state.color != token:
            # Reset charge options efficiently
            state.charge_opts = {
                "NULL": vocab.CHARGES.copy(),
                "LEFT": vocab.CHARGES.copy(),
                "RIGHT": vocab.CHARGES.copy()
            }
            state.color = token
        elif token in vocab.CHARGES:
            state.charge = token 
            if state.chirality == "NULL":
                if state.particle_type == "CSCALAR":
                    state.charge_opts["NULL"].remove(token)
                elif state.particle_type == "RSCALAR": pass
                else:  # FERMION
                    state.charge_opts["LEFT"].remove(token)
                    state.charge_opts["RIGHT"].remove(token)
            else:
                state.charge_opts[state.chirality].remove(token)
            tag_opts = utils.get_available_tags(state.tags, 
                                                    state.particle_type, 
                                                    state.chirality, 
                                                    state.color, 
                                                    state.charge)
            state.tag_opts = utils.mass_ordering(tag_opts)
        elif token in vocab.NUM_PTCLS: state.num_ptcls = int(token[4:])
        elif token in vocab.TAGS and state.current_block == "PARTICLE": 
            state.tag_list.append(token)
            state.tag_opts.remove(token)
        elif token == "END_PTCL":
            particle_dict = {
                "ptcls": state.tag_list,
                "num_ptcls": state.num_ptcls
            }
            if state.particle_type == "FERMION":
                if state.chirality == "NULL":
                    state.particle_inventory["FERMION"]["LEFT"][state.color][state.charge] = particle_dict.copy()
                    state.particle_inventory["FERMION"]["RIGHT"][state.color][state.charge] = particle_dict.copy()
                else:
                    state.particle_inventory["FERMION"][state.chirality][state.color][state.charge] = particle_dict.copy()
            else: 
                state.particle_inventory[state.particle_type][state.color][state.charge] = particle_dict.copy()

            state.ptcl_counts[state.particle_type] += 1
            state.charge = None
            state.num_ptcls = 0
            state.tag_list = []
            state.color = None
            state.particle_count += 1

        # ======================== MULTIPLET BLOCK ========================
        elif token == "MULTIPLET_BLOCK": 
            state.particles = copy.deepcopy(state.particle_inventory)
            state.current_block = "MULTIPLET"
        elif token in vocab.MULTIPLET_TYPES: state.multiplet_type = token[5:]
        elif token in vocab.MULTIPLET_IDS and state.current_block == "MULTIPLET": state.multiplet_id = token
        elif token in vocab.CHIRALITIES: state.chirality = token
        elif token == "REPS": pass
        # same as the SSB_BLOCK
        #elif token in (*vocab.REPRESENTATIONS, *vocab.HYPERCHARGES): state.rep_list.append(token)
        elif token == "END_REPS" and state.current_block == "MULTIPLET": 
            state.charge_list = rep.get_allowed_charge(state.rep_list)
            state.ptcl_list = {charge: [] for charge in state.charge_list}
            state.mass_list = {charge: [] for charge in state.charge_list}
            state.name_list = {charge: [] for charge in state.charge_list}
            state.width_list = {charge: [] for charge in state.charge_list}
            state.pdgi_list = {charge: [] for charge in state.charge_list}

        elif token in vocab.DIMS: state.dim = token
        elif token in vocab.GENS: 
            state.gen = int(token[4:])
            color = "NO_COLOR" if state.rep_list[0] == "singlet" else "COLOR"
            
            for charge in state.charge_list:
                if state.multiplet_type == "FERMION": particle_dict = state.particle_inventory["FERMION"][state.chirality][color]
                else: particle_dict = state.particle_inventory[state.multiplet_type][color]
                particle = particle_dict.get(charge)
                if particle:
                    state.ptcl_list[charge] = particle.get("ptcls", [])
                    particle["num_ptcls"] -= state.gen

        elif token in vocab.VEV_IDS and state.current_block == "MULTIPLET": 
            state.rep_list = state.vevs[token]["rep_list"]
            state.charge_list = rep.get_allowed_charge(state.rep_list)
            state.dim = state.vevs[token]["dim"]
            state.gen = 1
            state.vevs[token]["multiplets"] = state.multiplet_id
            state.vev_opts.remove(token)
            state.vev_id = token
            
            # fixed for Higgs doublet
            state.ptcl_list = {"charge_6": ["Hp"], "charge_0": ["H0"]}
            state.mass_list = {"charge_6": [], "charge_0": []}
            state.name_list = {"charge_6": [], "charge_0": []}
            state.width_list = {"charge_6": [], "charge_0": []}
            state.pdgi_list = {"charge_6": [251], "charge_0": [250]}

        elif token == "END_MULTIPLET": 
            if state.rep_list == ["singlet", "fnd", "hypercharge_3"] and state.multiplet_type == "CSCALAR": mplt_name = "H"
            elif state.rep_list == ["singlet", "fnd", "hypercharge_-3"] and state.multiplet_type == "FERMION": mplt_name = "l"
            elif state.rep_list == ["singlet", "singlet", "hypercharge_-6"] and state.multiplet_type == "FERMION": mplt_name = "e"
            elif state.rep_list == ["fnd", "fnd", "hypercharge_1"] and state.multiplet_type == "FERMION": mplt_name = "q"
            elif state.rep_list == ["fnd", "singlet", "hypercharge_4"] and state.multiplet_type == "FERMION": mplt_name = "u"
            elif state.rep_list == ["fnd", "singlet", "hypercharge_-2"] and state.multiplet_type == "FERMION": mplt_name = "d"
            else:
                if state.multiplet_type == 'FERMION' and state.rep_list[0] != 'singlet':
                    prefix = "Q"
                elif state.multiplet_type == 'FERMION' and state.rep_list[0] == 'singlet':
                    prefix = "L"
                elif state.multiplet_type in ['CSCALAR', 'RSCALAR']:
                    prefix = "S"
                suffix = nc.num2abc(state.X_num+1)
                mplt_name = f"{prefix}{suffix}"
                state.X_num += 1
            
            state.multiplets[state.multiplet_id] = {
                "id": state.multiplet_id,
                "name": mplt_name,
                "type": state.multiplet_type,
                "chirality": state.chirality,
                "rep_list": state.rep_list,
                "gen": state.gen,
                "dim": state.dim,
                "charges": state.charge_list,
                "ptcl_list": state.ptcl_list,
                "mass_list": state.mass_list,
                "name_list": state.name_list,
                "width_list": state.width_list,
                "pdgi_list": state.pdgi_list,
                "vev_id": state.vev_id
            }
            state.multiplet_id = None
            state.mplt_counts[state.multiplet_type] += 1
            state.chirality = None
            state.charge_list = []
            state.rep_list = []
            state.ptcl_list = {}
            state.gen = 0
            state.dim = 0
            state.vev_id = None

        # ======================= INTERACTION BLOCK =======================
        elif token == "INTERACTION_BLOCK": 
            state.current_block = "INTERACTION"
            # Initialize allowed multiplets for each interaction type
            state.allowed_mplts["TERM_PHI2"] = term_phi2.allowed_mplts(state.multiplets)
            state.allowed_mplts["TERM_PHI3"] = term_phi3.allowed_mplts(state.multiplets)
            state.allowed_mplts["TERM_PHI4"] = term_phi4.allowed_mplts(state.multiplets)
            state.allowed_mplts["TERM_YUKAWA"] = term_yukawa.allowed_mplts(state.multiplets)
        elif token in vocab.INTERACTION_TYPES: state.interaction_type = token
        elif token in vocab.INTERACTION_IDS: state.interaction_id = token
        elif token in [*vocab.TAGS, *vocab.VEV_IDS, *vocab.MASS, *vocab.PARAMETERS, "PARAM"] and state.current_block == "INTERACTION": 
            state.param_list.append(token)
            if token in state.param_opts: state.param_opts.remove(token)            
            if state.interaction_type == "TERM_YUKAWA":
                if state.multiplets[state.mplt_list[0]]["vev_id"]:
                    for idx in [1, 2]:
                        mplt = state.multiplets[state.mplt_list[idx]]
                        if token.startswith("SM_"):
                            mplt["name_list"][state.charge_eigenstate].append(PDG_IDS[token]['name'])
                            mplt["mass_list"][state.charge_eigenstate].append(PDG_IDS[token]['mass'])
                            mplt["width_list"][state.charge_eigenstate].append(PDG_IDS[token]['width'])
                            mplt["pdgi_list"][state.charge_eigenstate].append(PDG_IDS[token]['pdgid'])
                        else: 
                            mplt["name_list"][state.charge_eigenstate].append(f"X{nc.num2abc(state.X_ptcl + 1 - 10000)}")
                            mplt["mass_list"][state.charge_eigenstate].append(float(token[5:]))
                            mplt["width_list"][state.charge_eigenstate].append("Automatic")
                            mplt["pdgi_list"][state.charge_eigenstate].append(state.X_ptcl + 1)
                    if not token.startswith("SM_"): state.X_ptcl += 1

        elif token in vocab.MULTIPLET_IDS and state.current_block == "INTERACTION": 
            state.mplt_list.append(token)
        elif token == "END_MPLT":
            if state.interaction_type == "TERM_YUKAWA":
                scalar = state.multiplets[state.mplt_list[0]]
                mplt_left = state.multiplets[state.mplt_list[1]]
                mplt_right = state.multiplets[state.mplt_list[2]]
                state.charge_eigenstate = list[Any](set(mplt_left["charges"]) & set(mplt_right["charges"]))[0]
                state.param_opts = utils.mass_ordering(mplt_left["ptcl_list"][state.charge_eigenstate])
                state.num_params = mplt_left["gen"]
                mplt_right["gen"] = state.num_params
                state.allowed_mplts["TERM_YUKAWA"].remove(state.mplt_list)
                state.LagHC = term_yukawa.get_Lag(scalar, mplt_left, mplt_right)
            elif state.interaction_type == "TERM_PHI4":
                state.num_params = 1
                state.allowed_mplts["TERM_PHI4"].remove(state.mplt_list)
                state.LagNoHC = term_phi4.get_Lag(state.multiplets[state.mplt_list[0]])
            elif state.interaction_type == "TERM_PHI3":
                state.num_params = 1
                state.allowed_mplts["TERM_PHI3"].remove(state.mplt_list)
                state.LagNoHC = term_phi3.get_Lag(state.multiplets[state.mplt_list[0]])
            elif state.interaction_type == "TERM_PHI2":
                state.num_params = 1
                if state.multiplets[state.mplt_list[0]]["vev_id"]:
                    state.param_opts.append(state.multiplets[state.mplt_list[0]]["vev_id"])
                state.allowed_mplts["TERM_PHI2"].remove(state.mplt_list)
                state.LagNoHC = term_phi2.get_Lag(state.multiplets[state.mplt_list[0]])

        elif token == "END_INTERACTION":

            state.interactions[state.interaction_id] = {
                "id": state.interaction_id,
                "type": state.interaction_type,
                "param_list": state.param_list,
                "mplt_list": state.mplt_list,
                "LagHC": state.LagHC,
                "LagNoHC": state.LagNoHC
            }
            state.interaction_id = None
            state.interaction_type = None
            state.param_opts = []
            state.param_list = []
            state.mplt_list = []
            state.charge_eigenstate = None
            state.num_params = 0
            state.LagHC = None
            state.LagNoHC = None
    
        # ======================== ANOMALY BLOCK ========================
        elif token == "ANOMALY_BLOCK": state.current_block = "ANOMALY"
        elif token in vocab.ANOMALY_COEFF:
            anomaly_funcs = [
                    ("U1_SU3_anomaly", anomaly.U1_SU3_anomaly),
                    ("U1_SU2_anomaly", anomaly.U1_SU2_anomaly),
                    ("U1_anomaly", anomaly.U1_anomaly),
                    ("grav_anomaly", anomaly.grav_anomaly),
                    ("witten_anomaly", anomaly.witten_anomaly),
            ]
            idx = len(state.anomalies)
            if idx < 5:
                key, func = anomaly_funcs[idx]
                coeff = func(state.multiplets)
                state.anomalies[key] = coeff

        # ================== UPDATE LAST TOKEN AND LENGTH ==================
        elif token == "EOS": pass

        state.last_token= token
        state.length += 1
        return state



    def get_valid_tokens(self, state: GrammarState, token: str) -> list[str]:

        if token == "BOS": return ['GAUGE_GROUP_BLOCK']
        elif token == "EOS": return []  # Terminal state, no more valid tokens
        elif token in ["THEORY_TOO_LONG", "TOO_MANY_INTERACTIONS", "TOO_MANY_PARAMS"]: return ["EOS"]
        elif state.length == config.MAX_LENGTH - 1: return ["THEORY_TOO_LONG"]

        # ======================== GAUGE BLOCK ======================== 
        elif token == "GAUGE_GROUP_BLOCK": return vocab.GROUP_TYPES
        elif state.current_block == "GAUGE": 
            if token in vocab.GROUP_TYPES: return [f"g_{len(state.gauge_groups)+1}"]
            elif token.startswith("g_"): return self.GROUP_RANK_MAP[state.group_type]
            elif token in vocab.GROUP_RANKS: return ["END_GAUGE"]
            elif token == "END_GAUGE": 
                if len(state.gauge_groups) < config.max_gauge_groups:
                    return [*vocab.GROUP_TYPES, "SSB_BLOCK"]
                else: 
                    return ["SSB_BLOCK"]

        # ======================== SSB BLOCK (fixed) ======================== 
        elif token == "SSB_BLOCK": return ["VEV"]
        elif state.current_block == "SSB":
            if token == "VEV": return [f"v_{len(state.vevs)+1}"]
            elif token.startswith("v_"): return ["REPS"]
            elif token in ("REPS", *vocab.REPRESENTATIONS, *vocab.HYPERCHARGES): 
                group_key = f"g_{len(state.rep_list)+1}"
                if group_key == "g_1": return vocab.REPRESENTATIONS
                elif group_key == "g_2": return vocab.REPRESENTATIONS
                elif group_key == "g_3": return vocab.HYPERCHARGES
                else: return ["END_REPS"]
            elif token == "END_REPS": return [self.SU2_DIM_MAP[state.rep_list[1]]]
            elif token in vocab.DIMS: return ["SM_VEV"]
            elif token == "SM_VEV": return ["VEC"]
            elif token in ["VEC", "0", "1"]: 
                if len(state.vev_vector) < state.dim: return ["0", "1"] 
                else: return ["END_VEC"]
            elif token == "END_VEC": return ["END_VEV"]
            elif token == "END_VEV": return ["PARTICLE_BLOCK"]

        # ======================== PARTICLE BLOCK ========================
        elif token == "PARTICLE_BLOCK": return utils.get_ptcl_type(state.ptcl_counts)
        elif token in vocab.PARTICLE_TYPES: return vocab.COLORS
        elif state.current_block == "PARTICLE": 
            # --- Complex Scalar ---
            if state.particle_type == "CSCALAR": 
                if token in vocab.COLORS: return ["NULL"]
                elif token in vocab.CHIRALITIES: return state.charge_opts["NULL"]
                elif token in vocab.CHARGES: 
                    return utils.get_ptcl_num(state.ptcl_counts, state.particle_type)
                elif token in vocab.NUM_PTCLS: 
                    if state.tag_opts: return [state.tag_opts[0]]
                    else: return ["END_PTCL"]
                elif token in vocab.TAGS: 
                    if state.tag_opts and len(state.tag_list) < state.num_ptcls: 
                        return [state.tag_opts[0]]
                    else: return ["END_PTCL"]
                elif token == "END_PTCL": 
                    if state.ptcl_counts["CSCALAR"] >= config.max_ptcl_count["CSCALAR"]: 
                        return ["PTCL_RSCALAR", "PTCL_FERMION"]
                    else: return ["PTCL_CSCALAR", "PTCL_RSCALAR", "PTCL_FERMION"]
            # --- Real Scalar ---
            elif state.particle_type == "RSCALAR": 
                if token in vocab.COLORS: return ["NULL"]
                elif token in vocab.CHIRALITIES: return ["charge_0"]
                elif token in vocab.CHARGES: 
                    return utils.get_ptcl_num(state.ptcl_counts, state.particle_type)
                elif token in vocab.NUM_PTCLS: 
                    if state.tag_opts: return [state.tag_opts[0]]
                    else: return ["END_PTCL"]
                elif token in vocab.TAGS: 
                    if state.tag_opts and len(state.tag_list) < state.num_ptcls: 
                        return [state.tag_opts[0]]
                    else: return ["END_PTCL"]
                elif token == "END_PTCL": 
                    if state.ptcl_counts["RSCALAR"] >= config.max_ptcl_count["RSCALAR"]: 
                        return ["PTCL_FERMION"]
                    else: return ["PTCL_RSCALAR", "PTCL_FERMION"]
            # --- Fermion ---
            elif state.particle_type == "FERMION": 
                if token in vocab.COLORS: return vocab.CHIRALITIES
                elif token in vocab.CHIRALITIES: 
                    if token == "NULL": 
                        return list(set[Any](state.charge_opts["LEFT"]) & set[Any](state.charge_opts["RIGHT"]))
                    else: 
                        return state.charge_opts[token]
                elif token in vocab.CHARGES: 
                    return utils.get_ptcl_num(state.ptcl_counts, state.particle_type)
                elif token in vocab.NUM_PTCLS: 
                    if state.tag_opts: return [state.tag_opts[0]]
                    else: return ["END_PTCL"]
                elif token in vocab.TAGS: 
                    if state.tag_opts and len(state.tag_list) < state.num_ptcls: 
                        return [state.tag_opts[0]]
                    else: return ["END_PTCL"]
                elif token == "END_PTCL": 
                    if state.ptcl_counts["FERMION"] >= config.max_ptcl_count["FERMION"]: 
                        return ["MULTIPLET_BLOCK"]
                    elif state.ptcl_counts["FERMION"] <= config.max_ptcl_count["FERMION"]/4:
                        return ["PTCL_FERMION"]
                    else: return ["PTCL_FERMION", "MULTIPLET_BLOCK"]
        

        # ======================== MULTIPLET BLOCK ========================
        elif token == "MULTIPLET_BLOCK": 
            if state.vev_opts: return ["MPLT_CSCALAR"]
            return utils.get_mplt_type(state.vevs, state.particle_inventory, state.multiplet_type, state.mplt_counts)
        elif token in vocab.MULTIPLET_TYPES: return [f"m_{len(state.multiplets)+1}"]
        elif state.current_block == "MULTIPLET":

            # --- multiplet acquires vev ---
            if token.startswith("m_") and state.vev_opts: return ["NULL"]
            if token == "NULL" and state.vev_opts: return ["ACQUIRE"]
            elif token == "ACQUIRE": return [state.vev_opts[0]]
            elif token in vocab.VEV_IDS: return ["END_MULTIPLET"]

            # --- multiplet does not acquire vev ---
            elif token.startswith("m_") and not state.vev_opts: 
                if state.multiplet_type in ["CSCALAR", "RSCALAR"]: return ["NULL"]
                elif state.multiplet_type == "FERMION": 
                    return utils.get_fermion_chirality(state.particle_inventory)
            elif token in vocab.CHIRALITIES: return ["REPS"]
            elif token in ("REPS", *vocab.REPRESENTATIONS, *vocab.HYPERCHARGES) or token.startswith("hypercharge_"):
                group_key = f"g_{len(state.rep_list)+1}"
                if group_key == "g_1": 
                    return rep.get_SU3_rep(state.multiplet_type, state.chirality, state.particle_inventory)
                elif group_key == "g_2":
                    return rep.get_SU2_rep(state.chirality)
                elif group_key == "g_3": 
                    return rep.get_hypercharge(state.multiplet_type, state.chirality, state.rep_list, state.particle_inventory, state.multiplets)
                else: return ["END_REPS"]
            elif token == "END_REPS": return [self.SU2_DIM_MAP[state.rep_list[1]]]
            elif token in vocab.DIMS: 
                color = "NO_COLOR" if state.rep_list[0] == "singlet" else "COLOR"
                return utils.get_preferred_gen(state.chirality, color, state.charge_list, state.particle_inventory)
            elif token in vocab.GENS: return ["END_MULTIPLET"]
            elif token == "END_MULTIPLET": 
                if len(state.multiplets) == config.max_multiplets: return ["INTERACTION_BLOCK"]
                else: return utils.get_mplt_type(state.vevs, state.particle_inventory, state.multiplet_type, state.mplt_counts)

        # ======================== INTERACTION BLOCK ========================
        elif token == "INTERACTION_BLOCK": 
            for interaction_type, allowed in state.allowed_mplts.items():
                if allowed: return [interaction_type]
            return ["EOS"]
        elif token in vocab.INTERACTION_TYPES: return [f"i_{len(state.interactions)+1}"]
        elif state.current_block == "INTERACTION":
            if token.startswith("i_"): return ["MPLTS"]
            elif token in ["MPLTS", *vocab.MULTIPLET_IDS]:
                if len(state.mplt_list) == self.INTERACTION_MAP[state.interaction_type]:
                    return ["END_MPLT"]
                elif state.interaction_type in state.allowed_mplts:
                    return utils.get_next_mplt(
                        state.allowed_mplts[state.interaction_type], state.mplt_list
                    )
            elif token == "END_MPLT": return ["PARAMS"]
            elif token in ["PARAMS", *vocab.TAGS, *vocab.VEV_IDS, *vocab.MASS, *vocab.PARAMETERS]: 
                if len(state.param_list) == state.num_params: 
                    return ["END_PARAM"]
                elif state.param_opts: return [state.param_opts[0]]
                elif state.interaction_type == "TERM_YUKAWA": 
                    return term_yukawa.get_valid_param(state.last_token)
                elif state.interaction_type == "TERM_PHI4": 
                    return vocab.PARAMETERS
                elif state.interaction_type == "TERM_PHI3": 
                    return vocab.MASS
                elif state.interaction_type == "TERM_PHI2": 
                    vev_id = state.multiplets[state.mplt_list[0]]["vev_id"]
                    if vev_id: return [vev_id]
                    else: return vocab.MASS
            elif token == "END_PARAM": return ["END_INTERACTION"]
            elif token == "END_INTERACTION":
                if len(state.interactions) == config.max_interactions: 
                    print("TOO_MANY_INTERACTIONS")
                    return ["TOO_MANY_INTERACTIONS"]
                for interaction_type, allowed in state.allowed_mplts.items():
                    if allowed: return [interaction_type]
                return ["ANOMALY_BLOCK"]

        # ======================== ANOMALY BLOCK ========================
        elif token in ["ANOMALY_BLOCK", *vocab.ANOMALY_COEFF]: 
            anomaly_funcs = [
                    ("U1_SU3_anomaly", anomaly.U1_SU3_anomaly),
                    ("U1_SU2_anomaly", anomaly.U1_SU2_anomaly),
                    ("U1_anomaly", anomaly.U1_anomaly),
                    ("grav_anomaly", anomaly.grav_anomaly),
                    ("witten_anomaly", anomaly.witten_anomaly),
            ]
            idx = len(state.anomalies)
            if idx < 5: # fixed for SU(3)xSU(2)xU(1)
                key, func = anomaly_funcs[idx]
                coeff = func(state.multiplets)
                return [anomaly.anomaly_range(coeff)]
            else: return ["EOS"]


    def post_init(self, state: GrammarState):
        """ Post Initialize the theory dict for model file generation """
        for _, multiplet in state.multiplets.items():
            elements = []
            for charge in multiplet["charges"]:
                # ----- fill in the physical particles -----
                ptcl_list = multiplet["ptcl_list"][charge]
                name_list = multiplet["name_list"][charge]
                if not name_list and not ptcl_list:
                    multiplet["name_list"][charge] = [f"X{nc.num2abc(state.X_ptcl + i + 1- 10000)}" for i in range(1, multiplet["gen"] + 1)]
                    multiplet["mass_list"][charge] = [0.0] * multiplet["gen"]
                    multiplet["width_list"][charge] = ["Automatic"] * multiplet["gen"]
                    multiplet["pdgi_list"][charge] = [state.X_ptcl + i for i in range(1, multiplet["gen"] + 1)]
                    state.X_ptcl += multiplet["gen"]

                elif not name_list and ptcl_list:
                    multiplet["name_list"][charge] = [PDG_IDS[ptcl]["name"] for ptcl in ptcl_list]
                    multiplet["mass_list"][charge] = [0.0] * multiplet["gen"]
                    multiplet["width_list"][charge] = ["Automatic"] * multiplet["gen"]
                    multiplet["pdgi_list"][charge] = [PDG_IDS[ptcl]["pdgid"] for ptcl in ptcl_list]
                    if len(ptcl_list) < multiplet["gen"]:
                        for i in range(multiplet["gen"] - len(ptcl_list)):
                            multiplet["name_list"][charge].append(f"X{nc.num2abc(state.X_ptcl + i + 1- 10000)}")
                            multiplet["pdgi_list"][charge].append(state.X_ptcl + i + 1)
                            state.X_ptcl += 1

                # ----- fill in the unphysical particles -----
                chiral_idx = 1 if multiplet["chirality"] in {"LEFT", "NULL"} else -1
                rep_map = {
                    0: lambda rep: '1' if rep == "singlet" else str(chiral_idx * 3) if rep == "fnd" else "",
                    1: lambda rep: '1' if rep == "singlet" else str(chiral_idx * 2) if rep == "fnd" else "",
                    2: lambda rep: str(chiral_idx * Fraction(int(rep.split("_")[1]), 6)) if rep.startswith("hypercharge_") else "",
                }
                rep_str = [rep_map[idx](rep) for idx, rep in enumerate(multiplet["rep_list"]) if idx in rep_map and rep_map[idx](rep)]
                rep_str = ", ".join(reversed(rep_str))
                
                element_name = multiplet["name_list"][charge][0]
                if element_name == "ve": element_name = 'v'
                if multiplet["chirality"] == "LEFT": elements.append(f"{element_name}L")
                elif multiplet["chirality"] == "RIGHT": elements.append(f"conj[{element_name}R]")
                else: elements.append(element_name)
            if multiplet["dim"] > 1: elements = f"{{{', '.join(elements)}}}"
            else: elements = elements[0]
            multiplet["multiplet_def"] = f"{multiplet['name']}, {multiplet['gen']}, {elements}, {rep_str}"
        
        for _, interaction in state.interactions.items():
            if interaction["type"] in ("TERM_PHI2", "TERM_PHI3", "TERM_PHI4", "TERM_YUKAWA"):
                term_map = {
                    "TERM_PHI2": term_phi2,
                    "TERM_PHI3": term_phi3,
                    "TERM_PHI4": term_phi4,
                    "TERM_YUKAWA": term_yukawa,
                }
                result = term_map[interaction["type"]].get_param(state.multiplets, interaction, state.LesHouches_idx)
                state.ExtParam.update(result["ext_param"])
                state.IntParam.update(result["int_param"])
                state.LesHouches_idx = result["LesHouches_idx"]
                state.matching_conditions.extend(result["matching_conditions"])
                state.tadpole_params.extend(result["tadpole_params"])

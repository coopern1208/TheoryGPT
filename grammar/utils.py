from config import config
import grammar.vocab as vocab


def all_ptcls_included(particles: dict) -> bool:
    for particle_type, content in particles.items():
        if particle_type in ["RSCALAR", "CSCALAR"]:
            for color, charge_dict in content.items():
                for charge, p in charge_dict.items():
                    if p.get("num_ptcls", 0) > 0: 
                        return False
        elif particle_type == "FERMION":
            for chirality, color_dict in content.items():
                for color, charge_dict in color_dict.items():
                    for charge, p in charge_dict.items():
                        if p.get("num_ptcls", 0) > 0: 
                            return False
    return True

def all_fermions_included(particles: dict) -> bool:
    for chirality, color_dict in particles["FERMION"].items():
        for color, charge_dict in color_dict.items():
            for charge, p in charge_dict.items():
                if p.get("num_ptcls", 0) > 0: 
                    return False
    return True

def all_real_scalars_included(particles: dict) -> bool:
    for particle_type, content in particles.items():
        if particle_type == "RSCALAR":
            for color, charge_dict in content.items():
                for charge, p in charge_dict.items():
                    if p.get("num_ptcls", 0) == 0: 
                        return False
    return True

def all_complex_scalars_included(particles: dict) -> bool:
    for particle_type, content in particles.items():
        if particle_type == "CSCALAR":
            for color, charge_dict in content.items():
                for charge, p in charge_dict.items():
                    if p.get("num_ptcls", 0) == 0: 
                        return False
    return True

def ptcls_type_options(particles: dict) -> list[str]:
    options = []
    for particle_type, content in particles.items():
        if particle_type in ["RSCALAR", "CSCALAR"]:
            for color, charge_dict in content.items():
                for charge, p in charge_dict.items():
                    if p.get("num_ptcls", 0) > 0: 
                        options.append(particle_type)
        elif particle_type == "FERMION":
            for chirality, color_dict in content.items():
                for color, charge_dict in color_dict.items():
                    for charge, p in charge_dict.items():
                        if p.get("num_ptcls", 0) > 0: 
                            options.append(particle_type)
    return list(set(options))

def get_mplt_type(vevs: dict, particles: dict, current_mplt_type: str, mplt_counts: dict) -> list[str]:
    # mplt_type_ordering = ["MPLT_CSCALAR", "MPLT_RSCALAR", "MPLT_FERMION"]
    # if mplt_counts[current_mplt_type] >= config.max_mplt_count[current_mplt_type]:
    #     next_mplt_type = []
    # elif current_mplt_type:
    #     #next_mplt_type = [f"MPLT_{current_mplt_type}"] 
    #     next_mplt_type = []
    # else: 
    #     next_mplt_type = []
        
    for vev_id, vev in vevs.items():
        if vev["multiplets"] is None:
            if vev["rep_list"][2] != "hypercharge_0": # complex representation
                return [f"MPLT_CSCALAR"]
            else: # real representation
                return [f"MPLT_RSCALAR", f"MPLT_CSCALAR"]

    for particle_type, content in particles.items():
        if particle_type in ["RSCALAR", "CSCALAR"]:
            for _, charge_dict in content.items():
                for _, p in charge_dict.items():
                    if p.get("num_ptcls", 0) > 0: 
                        return [f"MPLT_{particle_type}"]
        elif particle_type == "FERMION":
            for _, color_dict in content.items():
                for _, charge_dict in color_dict.items():
                    for _, p in charge_dict.items():
                        if p.get("num_ptcls", 0) > 0: 
                            return [f"MPLT_FERMION"]
    return ["INTERACTION_BLOCK"]

def get_fermion_chirality(particles: dict) -> list[str]:
    candidate_chiralities = []
    for chirality, color_dict in particles["FERMION"].items():
        for color, charge_dict in color_dict.items():
            for charge, p in charge_dict.items():
                if p.get("num_ptcls", 0) > 0: 
                    candidate_chiralities.append(chirality)
    if candidate_chiralities: return list(set(candidate_chiralities))
    else: return ["LEFT", "RIGHT"]

def get_available_tags(tags_dict, particle_type, chirality, color, charge):
    if particle_type == 'CSCALAR':
        return tags_dict['CSCALAR'][color].get(charge, [])
    elif particle_type == 'RSCALAR':
        return tags_dict['RSCALAR'][color].get(charge, [])
    elif particle_type == 'FERMION':
        if chirality == 'LEFT':
            return tags_dict['FERMION']['LEFT'][color].get(charge, [])
        elif chirality == 'RIGHT':
            return tags_dict['FERMION']['RIGHT'][color].get(charge, [])
        elif chirality == 'NULL':
            left_tags = tags_dict['FERMION']['LEFT'][color].get(charge, [])
            right_tags = tags_dict['FERMION']['RIGHT'][color].get(charge, [])
            return list(set(left_tags + right_tags))
    else: return []

def mass_ordering(particle_tags):
    particle_ordering = ["SM_VE", "SM_VM", "SM_VT", "SM_E", "SM_U", "SM_D", "SM_MU", "SM_S", "SM_C", "SM_TAU", "SM_B"] + vocab.PARAM_IDS
    return sorted(particle_tags, key=lambda x: particle_ordering.index(x))

def get_next_mplt(mplt_list, query):
    valid_list = [mplts for mplts in mplt_list if mplts[:len(query)] == query]
    idx = len(query) 
    next_mplt = [candidate[idx] for candidate in valid_list]
    return next_mplt

def get_preferred_gen(chirality, color, charge_list, particles) -> list[str]:
    if chirality == "NULL": return ["gen_1"]
    preferred_gen = 1
    for charge in charge_list:
        num_particles = particles["FERMION"][chirality][color].get(charge, {}).get("num_ptcls", 0)
        if num_particles > preferred_gen:
            preferred_gen = num_particles
    preferred_gen_tokens = [f"gen_{i}" for i in range(preferred_gen, config.max_gen+1)]
    return preferred_gen_tokens 

def get_ptcl_num(ptcl_count: dict, particle_type: str):
    avail = config.max_ptcl_count[particle_type] - ptcl_count[particle_type]
    if avail < 1: return ["NUM_1"]
    return [f"NUM_{i}" for i in range(1, min(avail, 4) + 1)]

def get_ptcl_type(ptcl_count: dict): 
    ptcl_type_opts = []
    if ptcl_count["CSCALAR"] < config.max_ptcl_count["CSCALAR"]: ptcl_type_opts.append("PTCL_CSCALAR")
    if ptcl_count["RSCALAR"] < config.max_ptcl_count["RSCALAR"]: ptcl_type_opts.append("PTCL_RSCALAR")
    if ptcl_count["FERMION"] < config.max_ptcl_count["FERMION"]: ptcl_type_opts.append("PTCL_FERMION")
    return ptcl_type_opts

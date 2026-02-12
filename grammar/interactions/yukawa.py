from grammar.utils import mass_ordering
import grammar.vocab as vocab

def allowed_mplts(multiplets: dict) -> list[str]:
    left_fermions = [key for key, value in multiplets.items() if value["chirality"] == "LEFT" and value["type"] == "FERMION"]
    right_fermions = [key for key, value in multiplets.items() if value["chirality"] == "RIGHT" and value["type"] == "FERMION"]
    scalars = [key for key, value in multiplets.items() if value["chirality"] == "NULL" and value["type"] == "CSCALAR"]

    def check_dim():
        return multiplets[scalar]["dim"] == multiplets[lf]["dim"]

    def check_charges():
        return multiplets[rf]["charges"][0] in multiplets[lf]["charges"]

    def check_U1Y():
        Y_scalar = int(multiplets[scalar]["rep_list"][2].split("_")[1])
        Y_left = int(multiplets[lf]["rep_list"][2].split("_")[1])   
        Y_right = int(multiplets[rf]["rep_list"][2].split("_")[1])
        if Y_scalar - Y_left + Y_right == 0: return True
        elif -Y_scalar - Y_left + Y_right == 0: return True
        else: return False

    def check_color():
        color_left = multiplets[lf]["rep_list"][0]
        color_right = multiplets[rf]["rep_list"][0]
        return color_left == color_right
    
    valid_yukawas = []
    for scalar in scalars:
        for lf in left_fermions:
            for rf in right_fermions:
                if check_charges() and check_U1Y() and check_color() and check_dim():
                    valid_yukawas.append([scalar, lf, rf])

    return valid_yukawas


def get_valid_param(last_token):
    if last_token in vocab.MASS: 
        mass_idx = vocab.MASS.index(last_token)
        return vocab.MASS[mass_idx:]
    elif last_token in vocab.TAGS:
        last_mass = vocab.SM_TAGS_MASS.get(last_token, "mass_0")
        mass_idx = vocab.MASS.index(last_mass)
        return vocab.MASS[mass_idx:]
    else: return vocab.MASS

def get_Lag(scalar, left_fermion, right_fermion):
    yukawa_name = f"Y{right_fermion['name']}"
    Y_scalar = int(scalar["rep_list"][2].split("_")[1])
    Y_left = int(left_fermion["rep_list"][2].split("_")[1])   
    Y_right = int(right_fermion["rep_list"][2].split("_")[1])
    if - Y_scalar - Y_left + Y_right == 0:
        return f" - {yukawa_name} {right_fermion['name']}.{left_fermion['name']}.{scalar['name']}"
    elif Y_scalar - Y_left + Y_right == 0:
        return f" - {yukawa_name} conj[{scalar['name']}].{right_fermion['name']}.{left_fermion['name']}"

def get_param(multiplets, interaction, LesHouches_idx):
    left_fermion = multiplets[interaction['mplt_list'][1]]
    right_fermion = multiplets[interaction['mplt_list'][2]]
    yukawa_name = f"Y{right_fermion['name']}"
    ext_param = {}
    int_param = {}
    matching_conditions = []
    tadpole_params = []

    # ---- External Parameters ----
    charge_eigenstate = list(set(left_fermion["charges"]) & set(right_fermion["charges"]))[0]
    particle_name = left_fermion["name_list"][charge_eigenstate]
    for idx, param in enumerate(interaction['param_list']):
        math_idx = idx + 1
        if param == "SM_E": matching_conditions.append(f"{yukawa_name}[{math_idx}, {math_idx}], YeSM[1,1]")
        elif param == "SM_MU": matching_conditions.append(f"{yukawa_name}[{math_idx}, {math_idx}], YeSM[2,2]")
        elif param == "SM_TAU": matching_conditions.append(f"{yukawa_name}[{math_idx}, {math_idx}], YeSM[3,3]")
        elif param == "SM_U": matching_conditions.append(f"{yukawa_name}[{math_idx}, {math_idx}], YuSM[1,1]")
        elif param == "SM_C": matching_conditions.append(f"{yukawa_name}[{math_idx}, {math_idx}], YuSM[2,2]")
        elif param == "SM_D": matching_conditions.append(f"{yukawa_name}[{math_idx}, {math_idx}], YdSM[1,1]")
        elif param == "SM_S": matching_conditions.append(f"{yukawa_name}[{math_idx}, {math_idx}], YdSM[2,2]")
        elif param == "SM_B": matching_conditions.append(f"{yukawa_name}[{math_idx}, {math_idx}], YdSM[3,3]")
        elif param in ["SM_VE", "SM_VM", "SM_VT"]: continue
        else: 
            param_name = f"M{particle_name[idx]}"
            ext_param[param_name] = {
                'Description': f"{particle_name[idx]} Mass",
                'OutputName': param_name,
                'Block': "YUKAWA",
                'Dependence': None,
                'DependenceNum': None,
                'DependenceSPheno': None,
                'DependenceOptional': None,
                'Real': False,
                'Value': float(param.split("_")[-1]),
                'LesHouches': f"M{particle_name[idx]}",
                'LaTeX': f"M_{{{particle_name[idx]}}}"
            }
            if any(param.startswith("SM_") for param in interaction['param_list']):
                matching_conditions.append(f"{yukawa_name}[{math_idx}, {math_idx}], Sqrt[2]/vSM*M{particle_name[idx]}")

            LesHouches_idx += 1

            

    # ---- Internal Parameters ----
    name = right_fermion['name']
    descriptions = {
        'e': ("Lepton-Yukawa-Coupling", "Left-Lepton-Mixing-Matrix", "Right-Lepton-Mixing-Matrix"),
        'u': ("Up-Yukawa-Coupling", "Left-Up-Mixing-Matrix", "Right-Up-Mixing-Matrix"),
        'd': ("Down-Yukawa-Coupling", "Left-Down-Mixing-Matrix", "Right-Down-Mixing-Matrix"),
    }
    yukawa_description, left_description, right_description = descriptions.get(
        name, (f"{name}-Yukawa-Coupling", f"Left-{name}-Mixing-Matrix", f"Right-{name}-Mixing-Matrix")
    )

    int_param[f"Y{name}"] = {
        'Description': yukawa_description,
        'OutputName': f"Y{right_fermion['name']}",
        'Block': "YUKAWA",
        'Dependence': None,
        'DependenceNum': None,
        'DependenceSPheno': None,
        'DependenceOptional': None,
        'Real': False,
        'LesHouches': f"Y{right_fermion['name']}",
        'LaTeX': f"Y^{{{right_fermion['name']}}}_R"
    }
    int_param[f"V{name}"] = {
        'Description': left_description,
        'OutputName': f"Z{left_fermion['name'].upper()}L",
        'Block': "YUKAWA",
        'Dependence': None,
        'DependenceNum': None,
        'DependenceSPheno': None,
        'DependenceOptional': None,
        'Real': False,
        'LesHouches': f"U{left_fermion['name'].upper()}LMIX",
        'LaTeX': f"U^{{{left_fermion['name'].upper()}}}_L"
    }
    int_param[f"U{name}"] = {
        'Description': right_description,
        'OutputName': f"Z{right_fermion['name'].upper()}",
        'Block': "YUKAWA",
        'Dependence': None,
        'DependenceNum': None,
        'DependenceSPheno': None,
        'DependenceOptional': None,
        'Real': False,
        'LesHouches': f"U{right_fermion['name'].upper()}RMIX",
        'LaTeX': f"U^{{{right_fermion['name'].upper()}}}_R"
    }

    return {"ext_param": ext_param, 
            "int_param": int_param, 
            "LesHouches_idx": LesHouches_idx,
            "matching_conditions": matching_conditions,
            "tadpole_params": tadpole_params
            }

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
    for lf in left_fermions:
        for rf in right_fermions:
            for scalar in scalars:
                if check_charges() and check_U1Y() and check_color() and check_dim():
                    valid_yukawas.append([scalar, lf, rf])
    return valid_yukawas

def get_param(last_token):
    if last_token in vocab.MASS: 
        mass_idx = vocab.MASS.index(last_token)
        return vocab.MASS[mass_idx:]
    elif last_token in vocab.TAGS:
        last_mass = vocab.SM_TAGS_MASS.get(last_token, "mass_0")
        mass_idx = vocab.MASS.index(last_mass)
        return vocab.MASS[mass_idx:]
    else: return vocab.MASS
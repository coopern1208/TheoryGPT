

def allowed_mplts(multiplets: dict) -> list[str]:
    left_fermions = [key for key, value in multiplets.items() if value["chirality"] == "LEFT" and value["type"] == "FERMION"]
    right_fermions = [key for key, value in multiplets.items() if value["chirality"] == "RIGHT" and value["type"] == "FERMION"]

    def check_rep():
        return multiplets[lf]["rep_list"] == multiplets[rf]["rep_list"]
    
    valid_vectorlikes = []
    for lf in left_fermions:
        for rf in right_fermions:
            if check_rep():
                valid_vectorlikes.append([lf, rf])
    return valid_vectorlikes
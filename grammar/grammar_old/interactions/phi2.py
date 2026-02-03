def allowed_mplts(multiplets: dict) -> list[str]:
    allowed_list = []
    for key, value in multiplets.items():
        if value["type"] in ["CSCALAR", "RSCALAR"]:
            allowed_list.append([key])
    return allowed_list

def print_term():
    print("1/2 m^2 phi^2")


def allowed_mplts(multiplets: dict) -> list[str]:
    majorana_candidates = [key for key, value in multiplets.items() if value["type"] == "FERMION" and value["rep_list"][2] == "hypercharge_0"]
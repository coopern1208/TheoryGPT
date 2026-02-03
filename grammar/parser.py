from grammar import vocab

def read_model_txt(file):
    seq = []
    with open(file, "r") as f:
        sm_txt = f.read()
        for line in sm_txt.split("\n"):
            if not line.startswith("#"):
                line = line.split("#")[0].strip()
                if not line:
                    continue
                seq.extend(line.split())
    # Only add BOS/EOS if they don't already exist
    if not seq or seq[0] != "BOS":
        seq.insert(0, "BOS")
    if not seq or seq[-1] != "EOS":
        seq.append("EOS")
    return seq


def render_sequence(sequence: list[str]) -> str:
    new_seq = []
    for token in sequence:
        if any(x in token for x in ["BLOCK"]):
            new_seq.append(f"\n{token:12s}\n")
        elif any(x in token for x in ["END_GAUGE", "END_PTCL", "END_MULTIPLET", "END_INTERACTION", "END_ANOMALIES", "END_VEV", *vocab.PARAMETERS, *vocab.MASS]):
            new_seq.append(f"{token:8s}\n")
        elif any(x in token for x in ["FERMION", "SCALAR", "hypercharge", "TERM", "ANOMALIES", "VEV"]):
            new_seq.append(f"{token:14s}")
        elif any(x in token for x in ["BOS", "EOS"]): 
            pass
        else:
            new_seq.append(f"{token:10s}")
        
    return " ".join(new_seq)

def print_sequence(sequence: list[str]) -> None:
    """Print a formatted sequence of tokens"""
    formatted = render_sequence(sequence)
    print(formatted)

def render_model(model: dict):
    output = []
    output.append(" =============================== Gauge Groups =============================== \n")
    for group_id,gauge_group in model["gauge_groups"].items():
        output.append(f"{gauge_group}")
    output.append("\n =============================== Vevs =============================== \n")
    for vev_id,vev in model["vevs"].items():
        output.append(f"{vev}")
    output.append("\n =============================== Particles =============================== \n")
    for particle_type,particle in model["particles"].items():
        output.append(f"----------  {particle_type:5s}----------")
        for key, value in particle.items():
            output.append(f"    {key:15s}:")
            for color, color_value in value.items():
                output.append(f"        {color:15s}:")
                for charge, charge_value in color_value.items():
                    output.append(f"            {charge:15s}: {charge_value}")
    output.append("\n =============================== Multiplets =============================== \n")
    for multiplet_id,multiplet in model["multiplets"].items():
        output.append(f"----------  {multiplet_id:5s}----------")
        for key, value in multiplet.items():
            if key in ["id", "charges"]: continue
            output.append(f"    {key:15s}: {value}")

    output.append("\n =============================== Interactions =============================== \n")
    for interaction_id,interaction in model["interactions"].items():
        output.append(f"{interaction}")
    output.append("\n =============================== Anomalies =============================== \n")
    for anomaly_id,anomaly in model["anomalies"].items():
        output.append(f"{anomaly_id:15s}: {anomaly}")
    output.append("\n =============================== Params =============================== \n")
    for param_id,param in model["params"].items():
        output.append(f"{param_id:4s}: {param}")
    return "\n".join(output)
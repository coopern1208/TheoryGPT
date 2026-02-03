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


def print_sequence(sequence: list[str]) -> str:
    new_seq = []
    for token in sequence:
        if any(x in token for x in ["BLOCK"]):
            new_seq.append(f"\n{token:12s}\n")
        elif any(x in token for x in ["END_GAUGE", "END_PTCL", "END_MULTIPLET", "END_INTERACTION", "END_ANOMALIES", "END_VEV"]):
            new_seq.append(f"{token:8s}\n")
        elif any(x in token for x in ["FERMION", "SCALAR", "hypercharge", "TERM", "ANOMALIES", "VEV"]):
            new_seq.append(f"{token:14s}")
        elif any(x in token for x in ["BOS", "EOS"]): 
            pass
        else:
            new_seq.append(f"{token:10s}")
        
    print(" ".join(new_seq))

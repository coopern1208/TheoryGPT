def num2abc(number):
    result = ""
    while number > 0:
        remainder = (number - 1) % 26
        result = chr(remainder + 97) + result  # 97 is ASCII for 'a'
        number = (number - 1) // 26
    
    return result

def particle_name(rep_list: list[str], type: str, charge: int, vev_id: str) -> str:
    color = True if rep_list[0] != 'singlet' else False
    if type == 'FERMION':
        if color:
            if charge == 4: return 'u', 'Up Quark'
            elif charge == -2: return 'd', 'Down Quark'
            else:
                if charge > 0: return f'q{charge}p', f'Quark ({charge})'
                else: return f'q{abs(charge)}m', f'Quark ({charge})'
        else:
            if charge == 0: return 'v', 'Neutrino'
            elif charge == -6: return 'e', "Lepton"
            else:
                if charge > 0: return f'l{charge}p', f'Charged Lepton ({charge})'
                else: return f'l{abs(charge)}m', f'Charged Lepton ({charge})'

    elif type in ['CSCALAR', 'RSCALAR']:
        if charge == 0 and vev_id: return 'H0', 'Higgs Boson'
        elif charge == 6 and vev_id: return 'Hp', 'Higgs Boson'
        else:
            if charge > 0: return f'H{charge}p', f'Higgs Boson ({charge})'
            else: return f'H{abs(charge)}m', f'Higgs Boson ({charge})'

# def mplt_name(rep_list: list[str], type: str, mplt_id: str, X_num: int) -> str:
#     if rep_list == ["singlet", "fnd", "hypercharge_3"] and type == "CSCALAR": return "H"
#     elif rep_list == ["singlet", "fnd", "hypercharge_-3"] and type == "FERMION": return "l"
#     elif rep_list == ["singlet", "singlet", "hypercharge_-6"] and type == "FERMION": return "e"
#     elif rep_list == ["fnd", "fnd", "hypercharge_1"] and type == "FERMION": return "q"
#     elif rep_list == ["fnd", "singlet", "hypercharge_4"] and type == "FERMION": return "u"
#     elif rep_list == ["fnd", "singlet", "hypercharge_-2"] and type == "FERMION": return "d"
#     else:
#         if type == 'FERMION' and rep_list[0] != 'singlet':
#             prefix = "Q"
#         elif type == 'FERMION' and rep_list[0] == 'singlet':
#             prefix = "L"
#         elif type in ['CSCALAR', 'RSCALAR']:
#             prefix = "X"
#         suffix = num2abc(X_num+1)
#         print(suffix)

#         X_num += 1
#         return f"{prefix}{suffix}"


def exotic_particle(X_ptcl: int):    
    particle_names = "X" + num2abc(X_ptcl + 1)
    return particle_names

if __name__ == "__main__":
    for i in range(100):
        print(exotic_particle(i))
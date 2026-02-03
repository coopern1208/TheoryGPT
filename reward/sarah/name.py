def num2words(num):
    return {
        1: 'One',
        2: 'Two',
        3: 'Three',
        4: 'Four',
        5: 'Five',
        6: 'Six',
        7: 'Seven',
        8: 'Eight',
        9: 'Nine',
        10: 'Ten'
    }[num]


def fermion(charge, color):
    if color:
        if charge == 0: 
            return 'v', 'Neutrino'
        elif charge == -3:
            return 'e', "Lepton"
        else:
            if charge > 0:
                return f'l{charge}p', f'Charged Lepton ({charge})'
            else:
                return f'l{abs(charge)}m', f'Charged Lepton ({charge})'
    else:
        if charge == 2:
            return 'u', 'Up Quark'
        elif charge == -1:
            return 'd', 'Down Quark'
        else:
            if charge > 0:
                return f'q{charge}p', f'Quark ({charge})'
            else:
                return f'q{abs(charge)}m', f'Quark ({charge})'

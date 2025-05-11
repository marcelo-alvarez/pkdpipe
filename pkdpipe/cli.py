def parsecommandline(cparams,description):
    import argparse

    parser   = argparse.ArgumentParser(description=description)

    for param in cparams:
        pdval = cparams[param]['val']
        ptype = cparams[param]['type']
        pdesc = cparams[param]['desc']
        if ptype is bool:
            action='store_true'
            if pdval:
                print("storing false")
                action='store_false' 
            parser.add_argument('--'+param, action=action,help=f'{pdesc}')
        else:
            parser.add_argument('--'+param, default=pdval, help=f'{pdesc} [{pdval}]', type=ptype)

    return vars(parser.parse_args())

def create():
    from pkdpipe.simulation import Simulation
    Simulation(parse=True).create()

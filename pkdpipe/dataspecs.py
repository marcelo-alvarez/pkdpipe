
properties = {
    'xvp' : {'dtype' : ['f4','f4','f4','f4','f4','f4'], 
              'vars' : [ 'x', 'y', 'z','vx','vy','vz']},

    'xvh' : {'dtype' : ['f4','f4','f4','f4','f4','f4',   'i4'], 
              'vars' : [ 'x', 'y', 'z','vx','vy','vz','npart']},

    'xp'  : {'dtype' : ['f4','f4','f4'], 
              'vars' : [ 'x', 'y', 'z']}
}

fileformats = {
    # tps format
    #      all - 0,...,8 (36 bytes)
    #     mass - 0       (1 float)
    #    x,y,z - 1,2,3   (3 floats)
    # vx,vy,vz - 4,5,6   (3 floats)
    #      eps - 7       (1 float)
    #      phi - 8       (1 float)   
     'tps' : {'dtype'  : [('mass','>f4'),('x','>f4'),('y','>f4'),('z','>f4'),
                        ('vx','>f4'),('vy','>f4'),('vz','>f4'),('eps','>f4'),('phi','>f4')],
             'ext'    : None,
             'name'   : 'tipsy',
             'sliced' : False,
             'offset' : [-0.5,-0.5,-0.5],
             'hsize'  : 32,
             'dsize'  : 36},
    # lcp format
    #      all - 0,...,9 (40 bytes)
    #     mass - 0,1     (1 double)
    #    x,y,z - 2,3,4   (3 floats)
    # vx,vy,vz - 5,6,7   (3 floats)
    #      eps - 8       (1 float)
    #      phi - 9       (1 float)   
    'lcp' : {'dtype'  : [('mass','d'),("x",'f4'),("y",'f4'),("z",'f4'),
                        ("vx",'f4'),("vy",'f4'),("vz",'f4'),("eps",'f4'),("phi",'f4')],
             'ext'    : 'lcp',
             'name'   : 'lightcone',
             'sliced' : True,
             'offset' : [0,0,0],
             'hsize'  : 0,
             'dsize'  : 40},
    # fof format
    #      all - 0,...,33 (132 bytes)
    #    x,y,z - 0,1,2    (3 floats)
    #      pot - 3        (1 float)
    #     dum1 - 4-8      (12 bytes)
    # vx,vy,vz - 7,8,9    (3 floats)
    #     dum2 - 4-8      (84 bytes)
    #    npart - 31       (1 int)
    #     dum3 - 32,33    (8 bytes)
    'fof' : {'dtype'  : [ ('x','f4'), ('y','f4'), ('z','f4'),('pot','f4'),('dum1',('f4',3)),
                        ('vx','f4'),('vy','f4'),('vz','f4'),('dum2',('f4',21)),
                        ('npart','i4'),('dum3',('f4',2))],
             'ext'    : 'fofstats',
             'name'   : 'fof',
             'sliced' : False,
             'offset' : [-0.5,-0.5,-0.5],
             'hsize'  : 0,
             'dsize'  : 132}
}
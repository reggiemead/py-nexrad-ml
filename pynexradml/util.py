from __future__ import division

import numpy as np
import math

debug = False

def LOG(message):
    if debug:
        print message

def resample_scan(scan, dim='800', mtype='r'):
    if mtype == 'r':
        data = scan.refs
    elif mtype == 'v':
        data = scan.vels
    else:
        data = scan.sws

    a = map(lambda x: x if x < (0x20000 - 1) else 0, data.flatten())
    a = np.array(a)
    a.shape = data.shape

    b = np.zeros(dim * dim)
    b.shape = (dim, dim)
    for x in range(dim):
        for y in range(dim):
            x_t = x - (dim / 2)
            y_t = y - (dim / 2)
            if x_t != 0:
                theta = math.degrees(math.atan(y_t / x_t))
                if x_t < 0:
                    theta += 180
                elif y_t < 0:
                    theta += 360
                
            elif y_t >= 0:
                theta = 90
            else:
                theta = 270
            theta_t = ((theta * -1) + 90) % 360
            r = math.sqrt(x_t**2 + y_t**2)
            mindiff = map(lambda x: abs(theta_t - x), scan.azimuth)
            idx = mindiff.index(min(mindiff))
            rng = math.floor(r * math.cos(math.radians(mindiff[idx])))
            if mtype != 'r':
                rng = rng / 4
            if rng >= 0 and rng < len(a[idx]):
                b[x,y] = a[idx, rng]
            else:
                b[x,y] = 0
    return b

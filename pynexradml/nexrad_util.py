from __future__ import division

import numpy as np
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def display_sweep(imgs, color_maps):
    f = plt.figure()
    for n in range(len(imgs)):
        f.add_subplot(1, len(imgs), n + 1)
        plt.imshow(imgs[n], interpolation='bilinear', cmap=color_maps[n], origin="lower")
    plt.show()

def display_scan_images(imgs, map1=cm.jet, map2=cm.gray):
    for i in range(len(imgs)):
        if i == 0:
            im = plt.imshow(imgs[0], interpolation='bilinear', cmap=map2, origin='lower')
        else:
            im = plt.imshow(imgs[i], interpolation='bilinear', cmap=map1, alpha=.5, origin='lower')
    plt.show()

def normalize_image(img):
    result = np.zeros(img.shape[0] * img.shape[1] * 3)
    result.shape = (img.shape[0], img.shape[1], 3)

    dmin = min(map(lambda x: min(x), img))
    dmax = min(map(lambda x: min(x), img))

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
                result[x,y] = (img[x, y] - dmin) / (dmax - dmin)
    return result

def convert_to_red_green(img):
    result = np.zeros(img.shape[0] * img.shape[1] * 3)
    result.shape = (img.shape[0], img.shape[1], 3)

    dmin = min(map(lambda x: min(x), img))
    dmax = min(map(lambda x: min(x), img))

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            pixel = img[x, y]
            if pixel < 0:
                result[x,y] = [0.0, pixel / dmax, 0.0]
            else:
                result[x,y] = [abs(pixel / dmin), 0.0, 0.0]
    return result

def resample_scan(scan, dim=800, mtype='r'):
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
            theta_t = theta
            r = math.sqrt(x_t**2 + y_t**2)
            (idx, mindiff) = 0, 360
            for i in range(len(scan.azimuth)):
                diff = abs(theta_t - scan.azimuth[i])
                if diff < mindiff:
                    (idx, mindiff) = i, diff
            rng = r * math.cos(math.radians(mindiff))

            if mtype != 'r':
                rng = rng * 4
            rng = math.floor(rng)

            if rng >= 0 and rng < len(a[idx]):
                b[x,y] = a[idx, rng]
            else:
                b[x,y] = 0
    return b

def resample_sweep_polar(sweep, rscan_id=0, dscan_id=1, beam_width=1):
    GATES = 1840
    GATE_SIZE = 250

    rscan = sweep.scans[rscan_id]
    dscan = sweep.scans[dscan_id]

    rdata = rscan.refs
    vdata = dscan.vels
    sdata = dscan.sws

    r_vec = map(lambda x: x if x < (0x20000 - 1) else 0, rdata.flatten())
    r_vec = np.array(r_vec)
    r_vec.shape = rdata.shape

    v_vec = map(lambda x: x if x < (0x20000 - 1) else 0, vdata.flatten())
    v_vec = np.array(v_vec)
    v_vec.shape = vdata.shape

    s_vec = map(lambda x: x if x < (0x20000 - 1) else 0, sdata.flatten())
    s_vec = np.array(s_vec)
    s_vec.shape = sdata.shape

    result = np.zeros(360 * GATES * 3)
    result.shape = (360, GATES, 3)

    refFactor = GATE_SIZE / rscan.rGateSize
    dopFactor = GATE_SIZE / dscan.dGateSize

    for theta in xrange(360):
        r_az = 0
        mindiff = 360.0
        for i in range(len(rscan.azimuth)):
            diff = abs(theta - rscan.azimuth[i])
            if diff < mindiff:
                (r_az, mindiff) = i, diff
        d_az = 0
        mindiff = 360.0
        for i in range(len(dscan.azimuth)):
            diff = abs(theta - dscan.azimuth[i])
            if diff < mindiff:
                (d_az, mindiff) = i, diff
        for r in xrange(GATES):
            rangeIndex = r * refFactor
            result[theta, r, 0] = r_vec[r_az, rangeIndex] if rangeIndex < len(r_vec[r_az]) else 0
            rangeIndex = r * dopFactor
            result[theta, r, 1] = v_vec[d_az, rangeIndex] if rangeIndex < len(v_vec[d_az]) else 0
            result[theta, r, 2] = s_vec[d_az, rangeIndex] if rangeIndex < len(s_vec[d_az]) else 0
    return result

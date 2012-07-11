import numpy as np
cimport cython

from math import degrees
from math import radians

cdef extern from "math.h":
    double sqrt(double x)
    double atan(double x)
    double cos(double x)
    double fabs(double x)

@cython.boundscheck(False)
@cython.wraparound(False)
def resample_scan(scan, int dim=800, mtype='r'):
    cdef int x_t, y_t, irng, idx
    cdef double theta, rng, diff, r, mindiff
    cdef int dim_t = <int>(dim / 2)

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
            x_t = x - dim_t
            y_t = y - dim_t
            if x_t != 0:
                theta = degrees(atan(<double>y_t / <double>x_t))
                if x_t < 0:
                    theta += 180.0
                elif y_t < 0:
                    theta += 360.0
            elif y_t >= 0:
                theta = 90.0
            else:
                theta = 270.0
            r = sqrt(x_t**2 + y_t**2)
            idx = 0
            mindiff = 360.0
            for i in range(len(scan.azimuth)):
                diff = fabs(theta - scan.azimuth[i])
                if diff < mindiff:
                    (idx, mindiff) = i, diff
            rng = r * cos(radians(mindiff))

            if mtype != 'r':
                rng = rng * 4.0
            irng = (<int>rng)

            if rng >= 0 and rng < len(a[idx]):
                b[x,y] = a[idx, rng]
            else:
                b[x,y] = 0
    return b

@cython.boundscheck(False)
@cython.wraparound(False)
def resample_sweep_polar(sweep, rscan_id=0, dscan_id=1, beam_width=1):
    cdef int GATES = 1840
    cdef int GATE_SIZE = 250
    cdef int r_az, d_az, rangeIndex
    cdef double refFactor, dopFactor, mindiff, diff

    rscan = sweep.scans[rscan_id]
    dscan = sweep.scans[dscan_id]

    rdata = rscan.refs
    vdata = dscan.vels
    sdata = dscan.sws

    r_vec = rdata.flatten()
    r_vec = np.array(r_vec)
    r_vec.shape = rdata.shape

    v_vec = vdata.flatten()
    v_vec = np.array(v_vec)
    v_vec.shape = vdata.shape

    s_vec = sdata.flatten()
    s_vec = np.array(s_vec)
    s_vec.shape = sdata.shape

    result = np.zeros(360 * GATES * 3)
    result.shape = (360, GATES, 3)

    refFactor = <double>GATE_SIZE / <double>rscan.rGateSize
    dopFactor = <double>GATE_SIZE / <double>dscan.dGateSize
    for theta in xrange(360):
        r_az = 0
        mindiff = 360.0
        for i in range(len(rscan.azimuth)):
            diff = fabs(theta - rscan.azimuth[i])
            if diff < mindiff:
                (r_az, mindiff) = i, diff
        d_az = 0
        mindiff = 360.0
        for i in range(len(dscan.azimuth)):
            diff = fabs(theta - dscan.azimuth[i])
            if diff < mindiff:
                (d_az, mindiff) = i, diff
        for r in xrange(GATES):
            rangeIndex = r * refFactor
            result[theta, r, 0] = r_vec[r_az, rangeIndex] if rangeIndex < len(r_vec[r_az]) else 0x20000
            rangeIndex = r * dopFactor
            result[theta, r, 1] = v_vec[d_az, rangeIndex] if rangeIndex < len(v_vec[d_az]) else 0x20000
            result[theta, r, 2] = s_vec[d_az, rangeIndex] if rangeIndex < len(s_vec[d_az]) else 0x20000
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def resample_sweep(sweep, rscan_id=0, dscan_id=1, int dim=800):
    cdef int irng, idx
    cdef double x_t, y_t, theta, rng, diff, r, mindiff
    cdef double dim_t = (<double>dim / 2.0)

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

    result = np.zeros(dim * dim * 3)
    result.shape = (dim, dim, 3)

    for x in range(dim):
        for y in range(dim):
            x_t = <double>x - dim_t
            y_t = <double>y - dim_t
            if x_t != 0:
                theta = degrees(atan(y_t / x_t))
                if x_t < 0:
                    theta += 180
                elif y_t < 0:
                    theta += 360
            elif y_t >= 0:
                theta = 90
            else:
                theta = 270
            r = sqrt(x_t**2 + y_t**2)

            #Calculate Ref data
            idx = 0
            mindiff = 360.0
            for i in range(len(rscan.azimuth)):
                diff = fabs(theta - rscan.azimuth[i])
                if diff < mindiff:
                    (idx, mindiff) = i, diff
            rng = r * cos(radians(mindiff))
            irng = (<int>rng)
            if irng >= 0 and irng < len(r_vec[idx]):
                result[x,y,0] = r_vec[idx, irng]
            else:
                result[x,y,0] = 0

            #Calculate Vel/SW data
            idx = 0
            mindiff = 360.0
            for i in range(len(dscan.azimuth)):
                diff = fabs(theta - dscan.azimuth[i])
                if diff < mindiff:
                    (idx, mindiff) = i, diff
            rng = r * cos(radians(mindiff))
            irng = <int>rng * 4
            if irng >= 0 and irng < len(v_vec[idx]) and irng < len(s_vec[idx]):
                result[x,y,1] = v_vec[idx, irng]
                result[x,y,2] = s_vec[idx, irng]
            else:
                result[x,y,1] = 0
                result[x,y,2] = 0
    return result

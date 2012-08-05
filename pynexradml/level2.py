from __future__ import division

import numpy as np
import struct
import zcomp
import gzip
import math
import util

from collections import namedtuple

BADVAL = 0x20000
RFVAL = BADVAL - 1

def refConversion(x):
    if x > 1:
        return ((x - 2) / 2) - 32
    elif x == 1:
        return RFVAL
    else:
        return BADVAL

def dopConversion(x):
    if x > 1:
        return ((x - 2) / 2) - 63.5
    elif x == 1:
        return RFVAL
    else:
        return BADVAL

def bufferData(f, num):
    if num <= 0:
        return None
    data = f.read(num)
    while len(data) < num:
        nextData = f.read(num - len(data))
        if len(nextData) == 0:
                raise Level2Error("Data Error, expected %d but data ends at %d." % (num, len(data)))
        data += nextData
    return data

class Level2Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Packet9(object):
    def __init__(self, f):
        fields = "ctm, msgSize, msgChannel, msgType, idSeq, msgDate, msgTime, numSeg, segNum, " \
                 "rayTime, rayDate, unamRng, azm, rayNum, rayStatus, elev, elevNum, reflRng, " \
                 "dopRng, reflSize, dopSize, numRefl, numDop, secNum, sysCal, reflPtr, velPtr, " \
                 "spcPtr, velRes, volCpat, refPtrp, velPtrp, spcPtrp, nyqVel, atmAtt, minDif";
        self.__dict__.update(dict(zip(fields.split(', '), struct.unpack(">12sh2B2hi2hi14hf5h8x6h34x", bufferData(f, 128)))))
        self.data = struct.unpack("2300B", bufferData(f, 2300))
        self.fts = bufferData(f, 4)

    def getVCP(self):
        return self.volCpat
    def getUnamRange(self):
        return self.unamRng
    def getRefGateSize(self):
        return self.reflSize
    def getDopGateSize(self):
        return self.dopSize
    def getRefRange(self):
        return self.reflRng
    def getDopRange(self):
        return self.dopRng
    def getNumRef(self):
        return self.numRefl
    def getNumDop(self):
        return self.numDop
    def getAzimuth(self):
        return self.azm / 8 * (180 / 4096)
    def getElevation(self):
        return self.elev / 8 * (180 / 4096)
    def getRefData(self):
        return map(refConversion, self.data[self.reflPtr - 100 : self.reflPtr - 100 + self.numRefl])
    def getVelData(self):
        return map(dopConversion, self.data[self.velPtr - 100 : self.velPtr - 100 + self.numDop])
    def getSwData(self):
        return map(dopConversion, self.data[self.spcPtr - 100 : self.spcPtr - 100 + self.numDop])
    def isRadialData(self):
        return self.msgType == 1

class Packet10(object):
    def __init__(self, f):
        self.stream = f
        fields = "ctm, msgSize, msgChannel, msgType, idSeq, msgDate, msgTime, numSeg, segNum, " \
                 "identifier, rayTime, rayDate, azmNum, azmAngle, compression, radialLength, " \
                 "azmRes, rayStatus, elevNum, cutNum, elevAngle, spotBlanking, azmIdxMode, dbCount, volPtr, elevPtr, radialPtr"
        header = self.readFields(fields, ">12sh2B2hi2h4si2hfBxh4Bf2Bh3i", 72)
        self.__dict__.update(header)

        if self.msgType != 31:
            bufferData(f, 2360)
            return

        util.LOG(self.__dict__)

        mPtrs = struct.unpack(">6i", bufferData(f, 24))
        dataPtr = 0x44
        if self.dbCount > 0:
            if self.volPtr > dataPtr:
                bufferData(f, self.volPtr - dataPtr)
            self.volume = self.readFields("type, name, size, majorVersion, minorVersion, latitude, longitude, siteHeight, feedhornHeight, calibration, hTx, vTx, zdrCalibration, initialDp, vcp", ">B3sh2B2f2h5fh2x", 44)
            dataPtr = self.volPtr + 44
            util.LOG(self.volume)
        if self.dbCount > 1:
            if self.elevPtr > dataPtr:
                bufferData(f, self.elevPtr - dataPtr)
            self.elevation = self.readFields("type, name, size, attenuation, calibration", ">B3s2hf", 12)
            dataPtr = self.elevPtr + 12
        if self.dbCount > 2:
            if self.radialPtr > dataPtr:
                bufferData(f, self.radialPtr - dataPtr)
            self.radial = self.readFields("type, name, size, unamRange, hNoise, vNoise, nyq", ">B3s2h2fh2x", 20)
            dataPtr = self.radialPtr + 20

        self.moments = {}

        for i in range(self.dbCount - 3):
            if mPtrs[i] > dataPtr:
                bufferData(f, mPtrs[i] - dataPtr)
            moment = self.readFields("type, name, gates, range, gateSize, tover, snr, cFlags, wordSize, scale, offset", ">B3s4x5h2B2f", 28)
            dataSize = int(moment['gates'] * (moment['wordSize'] / 8))
            rawdata= struct.unpack(">%dB" % dataSize, bufferData(f, dataSize))
            data = []
            for j in range(len(rawdata)):
                if rawdata[j] == 0:
                    data.append(BADVAL)
                elif rawdata[j] == 1:
                    data.append(RFVAL)
                else:
                    data.append((rawdata[j] - moment['offset']) / moment['scale'])
            moment['data'] = data
            self.moments[moment['name']] = moment
            dataPtr = mPtrs[i] + 28 + dataSize

        endPtr = (self.msgSize * 2) - 0x10
        if dataPtr < endPtr:
            bufferData(f, endPtr - dataPtr)

    def readFields(self, fields, struct_org, size):
        return dict(zip(fields.split(", "), struct.unpack(struct_org, bufferData(self.stream, size))))
    def getVCP(self):
        return self.volume['vcp']
    def getUnamRange(self):
        return self.radial['unamRange']
    def getRefGateSize(self):
        return self.moments['REF']['gateSize']
    def getDopGateSize(self):
        return self.moments['VEL']['gateSize']
    def getRefRange(self):
        return self.moments['REF']['range']
    def getDopRange(self):
        return self.moments['VEL']['range']
    def getNumRef(self):
        if not 'REF' in self.moments:
            return 0
        return self.moments['REF']['gates']
    def getNumDop(self):
        if not 'VEL' in self.moments:
            return 0
        return self.moments['VEL']['gates']
    def getAzimuth(self):
        return self.azmAngle
    def getElevation(self):
        return self.elevAngle
    def getRefData(self):
        return self.moments['REF']['data']
    def getVelData(self):
        return self.moments['VEL']['data']
    def getSwData(self):
        return self.moments['SW ']['data']
    def isRadialData(self):
        return self.msgType == 31

class Level2Scan(object):
    def __init__(self, packets):
        self.vcp = packets[0].getVCP()
        self.unam = packets[0].getUnamRange() * 100 #Range in meters

        self.isRScan = (packets[0].getNumRef() > 0)
        self.isDScan = (packets[0].getNumDop() > 0)

        if self.isRScan:
            self.rGateSize = packets[0].getRefGateSize()
            self.rStartRange = packets[0].getRefRange()
        if self.isDScan:
            self.dGateSize = packets[0].getDopGateSize()
            self.dStartRange = packets[0].getDopRange()
        self.refs = []
        self.vels = []
        self.sws = []

        self.azimuth = map(lambda x: x.getAzimuth() % 360, packets)
        self.elev = map(lambda x: x.getElevation(), packets)

        if self.isRScan:
            self.refs = np.zeros(packets[0].getNumRef() * len(packets))
            self.refs.shape = (len(packets), -1)

        if self.isDScan:
            self.vels = np.zeros(packets[0].getNumDop() * len(packets))
            self.vels.shape = (len(packets), -1)
            self.sws = np.zeros(packets[0].getNumDop() * len(packets))
            self.sws.shape = (len(packets), -1)

        idx = 0
        for packet in packets:
            if self.isRScan:
                self.refs[idx, :] = packet.getRefData()
            if self.isDScan:
                self.vels[idx, :] = packet.getVelData()
                self.sws[idx, :] = packet.getSwData()
            idx += 1

class Sweep(object):
    def __init__(self, path, angle = 0.5):
        if path.endswith(".Z"):
            self.build = 9
        elif path.endswith(".gz"):
            self.build = 10

        f = self.openStream(path)
        self.readHeader(f)

        self.vcp_scan_angles = {11 : [.5, .5, 1.5, 1.5, 2.4, 3.4, 4.3, 5.3, 6.2, 7.5, 8.7, 10.0, 12.0, 14.0, 16.7, 19.5],
                                12 : [.5, .5, .9, .9, 1.3, 1.3, 1.8, 2.4, 3.1, 4.0, 5.1, 6.4, 8.0, 10.0, 12.5, 15.6, 19.5],
                                21 : [.5, .5, 1.5, 1.5, 2.4, 3.4, 4.3, 6.0, 9.9, 14.6, 19.5],
                                31 : [.5, .5, 1.5, 1.5, 2.5, 3.5, 4.5],
                                32 : [.5, .5, 1.5, 1.5, 2.5, 3.5, 4.5],
                                121 : [.5, .5, .5, .5, 1.5, 1.5, 1.5, 1.5, 2.4, 2.4, 2.4, 3.4, 3.4, 3.4, 4.3, 4.3, 6.0, 9.9, 14.6, 19.5],
                                211 : [.5, .5, 1.5, 1.5, 2.4, 3.4, 4.3, 5.3, 6.2, 7.5, 8.7, 10.0, 12.0, 14.0, 16.7, 19.5],
                                212 : [.5, .5, .9, .9, 1.3, 1.3, 1.8, 2.4, 3.1, 4.0, 5.1, 6.4, 8.0, 10.0, 12.5, 15.6, 19.5],
                                221 : [.5, .5, 1.5, 1.5, 2.4, 3.4, 4.3, 6.0, 9.9, 14.6, 19.5],}

        self.scans = []
        if angle > 0:
            self.loadScansForSweep(f, angle)

    def openStream(self, path):
        if self.build == 9:
            return zcomp.ZFileDecompressor(path);
        else:
            return gzip.open(path, "rb")

    def readHeader(self, f):
        if self.build == 9:
            Header = namedtuple('Header', 'name, ext, date, time')
            self.header = Header._make(struct.unpack(">9s3s2i4x", f.read(24)))
            if self.header.name != "ARCHIVE2." and self.header.name != "AR2V0001.":
                raise Level2Error("Error: This is not an Archive 2 File.")
        else:
            Header = namedtuple('Header', 'name, date, time, call')
            self.header = Header._make(struct.unpack(">12s2xhi4s", f.read(24)))

    def loadScansForSweep(self, f, angle):
        scan = self.readNextScan(f)
        vcp_angles = self.vcp_scan_angles[scan.vcp]
        scanIndexes = filter(lambda y: y >= 0, map(lambda x: x if vcp_angles[x] == angle else -1, range(len(vcp_angles))))
        if 0 in scanIndexes:
            self.scans.append(scan)

        for i in range(1, scanIndexes[-1] + 1):
            scan = self.readNextScan(f)
            if i in scanIndexes:
                self.scans.append(scan)

    def readNextScan(self, f):
        packets = []
        packet = self.readPacket(f)
        #search for beginning of scan
        while (packet.isRadialData() != True) or (packet.rayStatus != 0 and packet.rayStatus != 3):
            packet = self.readPacket(f)
        #search for end of scan
        while  (packet.isRadialData() != True) or (packet.rayStatus != 2 and packet.rayStatus != 4):
            if packet.isRadialData():
                packets.append(packet)
            packet = self.readPacket(f)
        packets.append(packet)
        return Level2Scan(packets)

    def readPacket(self, f):
        if self.build == 9:
            return Packet9(f)
        return Packet10(f)

from __future__ import division

import numpy as np
import struct
import zcomp
import gzip
import math

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
        self.__dict__.update(dict(zip(fields, struct.unpack(">12sh2B2hi2hi14hf5h8x6h34x", bufferData(f, 128)))))
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
        return self.azm
    def getElevation(self):
        return self.elev
    def getRefData(self):
        return map(refConversion, self.data[self.reflPtr - 100 : self.reflPtr - 100 + self.numRefl])
    def getVelData(self):
        return map(dopConversion, self.data[self.velPtr - 100 : self.velPtr - 100 + self.numDop])
    def getSwData(self):
        return map(dopConversion, self.data[self.spcPtr - 100 : self.spcPtr - 100 + self.numDop])

class PacketB10(object):
    def __init__(self, f):
        fields = "ctm, msgSize, msgChannel, msgType, idSeq, msgDate, msgTime, numSeg, segNum, " \
                 "header, identifier, rayTime, rayDate, azmNum, azmAngle, compression, radialLength, " \
                 "azmRes, status, elevNum, cutNum, elevAngle, spotBlanking, azmIdxMode, dbCount, volPtr, elevPtr, radialPtr"

        data = struct.unpack(">12sh2B2hi2h4si2hfBxh4Bf2Bh3i", bufferData(f, 72))
        self.__dict__.update(dict(zip(fields, data)))
        self.moments = {}

    def getVCP(self):
        pass
    def getUnamRange(self):
        pass
    def getRefGateSize(self):
        pass
    def getDopGateSize(self):
        pass
    def getRefRange(self):
        pass
    def getDopRange(self):
        pass
    def getNumRef(self):
        pass
    def getNumDop(self):
        pass
    def getAzimuth(self):
        pass
    def getElevation(self):
        pass
    def getRefData(self):
        pass
    def getVelData(self):
        pass
    def getSwData(self):
        pass


class Level2Scan(object):
    def __init__(self, packets):
        self.vcp = packets[0].getVCP()
        self.unam = packets[0].getUnamRange() * 100 #Range in meters
        self.rGateSize = packets[0].getRefGateSize()
        self.rStartRange = packets[0].getRefRange()
        self.dGateSize = packets[0].getDopGateSize()
        self.dStartRange = packets[0].getDopRange()
        self.refs = []
        self.vels = []
        self.sws = []

        self.azimuth = map(lambda x: (x.getAzimuth() / 8 * (180 / 4096)) % 360, packets)
        self.elev = map(lambda x: x.getElevation() / 8 * (180 / 4096), packets)

        self.isRScan = (packets[0].getNumRef() > 0)
        self.isDScan = (packets[0].getNumDop() > 0)

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

    def _refConversion(self, x):
        if x > 1:
            return ((x - 2) / 2) - 32
        elif x == 1:
            return RFVAL
        else:
            return BADVAL

    def _dopConversion(self, x):
        if x > 1:
            return ((x - 2) / 2) - 63.5
        elif x == 1:
            return RFVAL
        else:
            return BADVAL

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
        while packet.msgType != 1 or (packet.rayStatus != 0 and packet.rayStatus != 3):
            packet = self.readPacket(f)
        #search for end of scan
        while packet.msgType != 1 or (packet.rayStatus != 2 and packet.rayStatus != 4):
            if packet.msgType == 1:
                packets.append(packet)
            packet = self.readPacket(f)
        packets.append(packet)
        return Level2Scan(packets)

    def readPacket(self, f):
        if self.build == 9:
            return Packet9(f, self)
        else:
            return Packet10(f, self)

    """
    def readBuild9Packet(self, f):
        PacketHeader = namedtuple('PacketHeader', 'ctm, msgSize, msgChannel, msgType, idSeq, msgDate, msgTime, numSeg, segNum')
        pheader = PacketHeader._make(struct.unpack(">12sh2B2hi2h", self.bufferData(f, 28)))
        
        if pheader.msgType == 1:
            RayPacket = namedtuple('RayPacket', 'header, rayTime, rayDate, unamRng, azm, rayNum, rayStatus, elev, elevNum, reflRng, dopRng, reflSize, dopSize, numRefl, numDop, secNum, sysCal, reflPtr, velPtr, spcPtr, velRes, volCpat, refPtrp, velPtrp, spcPtrp, nyqVel, atmAtt, minDif, data, fts')
            return RayPacket._make((pheader,) + struct.unpack(">i14hf5h8x6h34x", self.bufferData(f, 100)) + (self.bufferData(f, 2300), self.bufferData(f, 4)))
        else:
            SkipPacket = namedtuple('SkipPacket', 'header')
            self.bufferData(f, 2404)
            return SkipPacket._make((pheader,))
    """

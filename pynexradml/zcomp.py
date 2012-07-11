import sys
import struct
import numpy as np

LZW_MAGIC = 0x1f9d
MAX_BITS = 16
INIT_BITS = 9
HDR_MAXBITS = 0x1f
HDR_EXTENDED = 0x20
HDR_FREE = 0x40
HDR_BLOCK_MODE = 0x80

TBL_CLEAR = 0x100
TBL_FIRST = TBL_CLEAR + 1

EXTRA = 64

class CompressionError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class ZFileDecompressor(object):
    def __init__(self, path):
        self.data = np.array(np.zeros(10000), dtype='B')
        self.bit_pos = 0
        self.end = 0
        self.got = 0
        self.eof = False
        self.debug = False
        self.buffer_full = False

        self.f = open(path, "rb")
        self.parse_header()

    def __del__(self):
        self.f.close()

    def log(self, msg):
        if self.debug:
            print msg

    def resetbuf(self):
        pos = self.bit_pos >> 3
        self.data[0:self.end-pos] = self.data[pos:self.end]
        self.end -= pos
        self.bit_pos = 0
        
    def fill(self):
        amt = len(self.data) - 1 - self.end
        binstring = self.f.read(amt)
        self.got = len(binstring)
        if self.got > 0:
            bindata = struct.unpack("%dB" % (self.got), binstring)
            self.data[self.end: self.end + self.got] = bindata[0:self.got]
            self.end += self.got

    def buffer_data(self):
        if self.end < EXTRA:
            self.fill()
        bit_in = (self.end - self.end % self.n_bits) << 3 if self.got > 0 else (self.end << 3) - (self.n_bits - 1)
        
        while self.bit_pos < bit_in:
            #check for code-width expansion
            if self.free_ent > self.maxcode:
                n_bytes = self.n_bits << 3
                byte_pos = (self.bit_pos - 1) + n_bytes
                self.bit_pos = byte_pos - (byte_pos % n_bytes)

                self.n_bits += 1
                self.maxcode = self.maxmaxcode if self.n_bits == self.maxbits else (1 << self.n_bits) - 1

                self.bitmask = (1 << self.n_bits) - 1
                self.resetbuf()
                return
            
            #read next code
            pos = self.bit_pos >> 3
            code = ((self.data[pos] | self.data[pos+1] << 8 | self.data[pos+2] << 16) >> (self.bit_pos & 0x7)) & self.bitmask
            self.log("Code = %d" % (code))
            if code < 256:
                self.log("Chr = %c" % chr(code))
            self.bit_pos += self.n_bits

            #check for first iteration
            if self.oldcode == -1:
                if code > 255:
                    raise CompressionError("Corrupt input: %d  > 255" % (code))
                self.oldcode = code
                self.finchar = np.uint8(code)
                self.buf[self.off] = self.finchar
                self.off += 1
                self.num -= 1
                continue

            #check for CLEAR code
            if (code == TBL_CLEAR) and self.block_mode:
                self.log("CLEAR Code")
                self.tab_prefix[0:256] = np.zeros(256)
                self.free_ent = TBL_FIRST - 1
                n_bytes = self.n_bits << 3;
                self.bit_pos = (self.bit_pos - 1) + n_bytes - ((self.bit_pos - 1 + n_bytes) % n_bytes)
                self.n_bits = INIT_BITS
                self.maxcode = (1 << self.n_bits) - 1
                self.bitmask = self.maxcode
                self.resetbuf()
                return

            incode = code
            self.stackp = len(self.stack)

            #handle KwK case
            if code >= self.free_ent:
                if code > self.free_ent:
                    raise CompressionError("Corrupt input: code=%d, free_ent=%d" % (code, self.free_ent))
                self.log("KwK")
                self.stackp -= 1
                self.stack[self.stackp] = self.finchar
                code = self.oldcode

            #generate output in reverse order
            while code >= 256:
                self.stackp -= 1
                self.stack[self.stackp] = self.tab_suffix[code]
                code = self.tab_prefix[code]
            self.finchar = self.tab_suffix[code]
            self.buf[self.off] = self.finchar
            self.off += 1
            self.num -= 1

            #put them out in forward order
            s_size = len(self.stack) - self.stackp
            amt = min(self.num, s_size)
            if amt > 0:
                self.log("Print Stack = [%s]" % (','.join(map(str, self.stack[self.stackp:self.stackp+amt]))))
                self.buf[self.off:self.off+amt] = self.stack[self.stackp:self.stackp+amt]
                self.off += amt
                self.num -= amt
                self.stackp += amt

            #generate new entry in table
            if self.free_ent < self.maxmaxcode:
                self.tab_prefix[self.free_ent] = self.oldcode
                self.tab_suffix[self.free_ent] = self.finchar
                self.free_ent += 1

            self.oldcode = incode

            if self.num == 0:
                self.buffer_full = True
                return
            
        self.resetbuf()

    def read(self, num):
        self.buf = np.array(np.zeros(num), dtype='B')
        self.off = 0
        self.num = num

        if self.eof:
            return self.buf

        #empty stack if data left over
        s_size = len(self.stack) - self.stackp
        if s_size > 0:
            amt = min(self.num, s_size)
            self.buf[0:amt] = self.stack[self.stackp : self.stackp + amt]
            self.off += amt
            self.num -= amt
            self.stackp += amt

        if self.num == 0:
            return self.buf
        
        self.buffer_data()
        if not self.buffer_full:
            while self.got > 0 and not self.buffer_full:
                self.buffer_data()

        if self.got == 0:
            self.eof = True
        return struct.pack("%dB" % (self.off), *self.buf[0:self.off])

    def parse_header(self):
        (magic,) = struct.unpack(">H", self.f.read(2))
        if magic != LZW_MAGIC:
            raise CompressionError("Error: File not in compress format")
        (header,) = struct.unpack("B", self.f.read(1))
        self.block_mode = (header & HDR_BLOCK_MODE) > 0
        self.maxbits = header & HDR_MAXBITS
        if self.maxbits > MAX_BITS:
            raise CompressionError("Stream compress with %d bits, but can only handle %d bits" % (self.maxbits, MAX_BITS))
        if (header & HDR_EXTENDED) > 0:
            raise CompressionError("Header extension bit set")
        if (header & HDR_FREE) > 0:
            raise CompressionError("Header bit 6 set")
        
        self.maxmaxcode = 1 << self.maxbits
        self.n_bits = INIT_BITS
        self.maxcode = (1 << self.n_bits) - 1
        self.bitmask = self.maxcode
        self.oldcode = -1
        self.finchar = 0
        if self.block_mode:
            self.free_ent = TBL_FIRST
        else:
            self.free_ent = 256

        self.tab_prefix = np.array(np.zeros(1 << self.maxbits), dtype='i')
        self.tab_suffix = np.array(np.zeros(1 << self.maxbits), dtype='B')
        self.stack = np.array(np.zeros(1 << self.maxbits), dtype='B')
        self.stackp = (1 << self.maxbits)

        for i in range(0, 256):
            self.tab_suffix[i] = np.uint8(i)

#------------------------------------------------------------------------
# Fixed point quantization class
# FixedPoint(N,F) is a signed 2's-complement type with N total bits 
# and F fractional bits.
# Use covert method to quantize a single number or an ndarray
#------------------------------------------------------------------------

import numpy as np

class FixedPoint():
    
    # N = total bits
    # F = fractional bits
    def __init__(self, N=32, F=0):
        assert F <= N
        assert N <= 32
        
        self.N = N
        self.F = F
        self.I = N-F

    def convert_one(self, Val):
        res = 0.0
        x = np.power(2, np.float32(self.I-1))
        
        if Val < 0:
            res -= x
            Val += x

        for i in range(self.N-1):
            tmp = np.power(2, np.float32(self.I-2-i))
            if Val >= tmp:
                res += tmp
                Val -= tmp

        # np.float3232 needed for theano
        return np.float32(res)

    def convert(self, x):
        if type(x) == np.ndarray:
            shp = x.shape
            res = [self.convert_one(xi) for xi in x.flatten()]
            return np.array(res).reshape(shp)
        else:
            return self.convert_one(x)


    def bits_one(self, Val):
        res = 0.0

        if Val < 0:
            res = np.power(10, np.float32(self.I-1))
            Val += np.power(2, np.float32(self.I-1))

        for i in range(self.N-1):
            tmp = np.power(2, np.float32(self.I-2-i))
            if Val >= tmp:
                res += np.power(10, np.float32(self.I-2-i))
                Val -= tmp

        return res

# Test suite
if __name__ == "__main__":
    fs = np.random.random(21)*128 - 64
    fs[0] = 0.0
    err = 0.0

    fixed = FixedPoint(16,8)

    for f in fs:
        ff = fixed.convert(f)
        print f, "->", ff, "==", fixed.bits(f)
        err += abs(ff-f)

    print "Mean Error =", err/len(fs)
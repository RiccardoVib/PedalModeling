#H = mag_smoothing(B, n)
import numpy as np
# This is a Matlab function that implements the 1 / n octave smoothing of the input magnitude response B.

def mag_smoothing(B, n):
    stop = int(len(B) / 2 + 1) # We don 't need the other side of spectra
    smooth_factor = 1 / n   #1 / n octave bandwidth
    Q = np.sqrt(2 ** (smooth_factor)) / (2 ** (smooth_factor) - 1)  # Q - factor
    B1 = np.abs(B)  # Only magnitude
    H = B1  # Initialize output vector
    q = int(np.fix(Q) + 1)  # pointer to first frequency to be smoothed

    for i in range(q, len(H)):  # scan each frequency bin

        N = int(np.fix(i / Q))  # amount of frequency bins in the smoothing window
        if N % 2 == 0:  #detect if it is even
            N = N + 1

        fc = (N + 1) / 2  # center frequency
        fh = int(N - fc)

        if i + fh > len(H):   # detect if there is not enough bandwidth
            f = int(N - (i + fh - len(H)))
        else:
            f = int(N)

        H[i] = np.sqrt(np.sum(B1[i - fh:i + fh - (N - f)]** 2) / f) # smoothing
    return H

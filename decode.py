from scipy.io import wavfile
from scipy.signal import butter, lfilter
import numpy as np
import math
import sys

# Constants
MARK_FREQ = 2083.3  # Frequency for bit 1
SPACE_FREQ = 1562.5 # Frequency for bit 0
BAUD_RATE = 520.83  # Bits per second (bitrate)
SAME_BUFFER = 128   # Max number of bytes to decode from message
PREAMBLE_STEPS = 1
ATTENTION_STEPS = 4

def main():
    # Read WAV file
    sample_rate, data = wavfile.read(sys.argv[1])
    print(f"sample_rate = {sample_rate}")
    print(f"shape = {data.shape}")
    length = round(data.shape[0] / sample_rate, 1)
    print(f"length = {length}s")

    # Calculate how many samples represent one bit
    bit_size = int(round((1 / BAUD_RATE) * sample_rate, 0))
    print(f"bit_size = {bit_size}")

    # If stereo, use only the first channel
    if len(data.shape) > 1:
        data = data[:, 0]

    # Normalize waveform to range [-1, 1]
    data = data / np.max(np.abs(data))

    # Filter signal to isolate expected tone range
    data = bandpass_filter(data, sample_rate)

    pos = 0
    while pos < len(data):
        # Find start of message
        pos = find_preamble(data, pos, sample_rate, bit_size)
        if pos < 0:
            break

        # Find message prefix (ZCZC or NNNN)
        pos, prefix = find_prefix(data, pos, sample_rate, bit_size)
        if pos < 0:
            break

        if prefix == "ZCZC":
            # Decode and print SAME message
            pos = print_same(data, pos, sample_rate, bit_size)
        else:
            # End of SAME message
            print(prefix)

def print_same(data, pos, sample_rate, bit_size):
    # Read bits from signal using a sliding window
    bits = []
    while pos < len(data):
        bit, offset = get_bit_sliding_window(data, sample_rate, bit_size, pos)
        if len(bits) > 1 and bit == bits[-1]:
            pos += bit_size  # if this bit is same as previous bit, no adjustment needed
        else:
            pos += bit_size + offset  # adjust using bit offset
        bits.append(bit)
        if len(bits) > SAME_BUFFER * 8:
            break

    # Convert bits to printable ASCII characters
    chars = []
    for i in range(0, len(bits), 8):
        tmp = bits[i:i + 8]
        tmp.reverse()  # transmit order is LSB first
        byte = int(''.join(str(b) for b in tmp), 2)
        if byte & 0x80:
            chars.append(".")  # invalid or non-printable ASCII
        elif 32 <= byte <= 126:
            chars.append(chr(byte))
        else:
            chars.append(".")

    # Print full message or just up to "+" marker plus 22 chars
    plus = chars.index("+") if "+" in chars else None
    if plus and len(chars) > plus + 22:
        print(''.join(chars[0 : plus + 22]))
    else:
        print(''.join(chars))

    return pos

# Bandpass filter to isolate relevant frequency range
def bandpass_filter(signal, fs, lowcut=1400.0, highcut=2200.0, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

# Goertzel algorithm: detect power of a tone within a block
def goertzel(samples, sample_rate, freq):
    N = len(samples)
    k = int(0.5 + (N * freq) / sample_rate)
    w = 2 * np.pi * k / N
    cosine = np.cos(w)
    sine = np.sin(w)
    coeff = 2 * cosine
    q0, q1, q2 = 0.0, 0.0, 0.0
    for sample in samples:
        q0 = coeff * q1 - q2 + sample
        q2, q1 = q1, q0
    magnitude = q1**2 + q2**2 - q1 * q2 * coeff
    return magnitude

# Compare MARK vs SPACE tone magnitude to determine bit value
def detect_bit(block, sample_rate):
    mark = goertzel(block, sample_rate, MARK_FREQ)
    space = goertzel(block, sample_rate, SPACE_FREQ)
    return 1 if mark > space else 0

# Look for 8-bit preamble (0xAB = 10101011) indicating message start
def find_preamble(data, pos, sample_rate, bit_size):
    while pos <= len(data) - bit_size * 8:
        chunk = data[pos:pos + bit_size * 8]
        bits = []
        for p in range(0, len(chunk), bit_size):
            b = detect_bit(chunk[p:p + bit_size], sample_rate)
            bits.append(b)
        if bits == [1, 1, 0, 1, 0, 1, 0, 1]:
            return pos
        pos += int(bit_size / PREAMBLE_STEPS)
    return -1

# Look for ZCZC (start) or NNNN (end) SAME message prefix
def find_prefix(data, pos, sample_rate, bit_size):
    offset = 0
    while pos <= len(data) - bit_size * 8 * 4:
        bits = []
        for i in range(0, 8 * 4):
            b = detect_bit(data[pos + i * bit_size:pos + (i + 1) * bit_size], sample_rate)
            bits.append(b)

        attention = [ 0,1,0,1,1,0,1,0,
                      1,1,0,0,0,0,1,0,
                      0,1,0,1,1,0,1,0,
                      1,1,0,0,0,0,1,0 ]
        if bits == attention:
            return pos, "ZCZC"

        eom = [ 0,1,1,1,0,0,1,0,
                0,1,1,1,0,0,1,0,
                0,1,1,1,0,0,1,0,
                0,1,1,1,0,0,1,0 ]
        if bits == eom:
            return pos, "NNNN"

        pos += int(bit_size / ATTENTION_STEPS)
        offset += 1
        if offset >= 16 * bit_size * ATTENTION_STEPS:
            return pos, None
    return -1, None

# Find local peaks in power array, ensuring minimum spacing between them
def find_peaks(power_array, min_spacing = 80):
    peaks = []
    last_peak = -min_spacing
    for i in range(1, len(power_array) - 1):
        if power_array[i] > power_array[i-1] and power_array[i] > power_array[i+1]:
            if i - last_peak >= min_spacing:
                peaks.append(i)
                last_peak = i
    return peaks

# Use sliding window to detect bit and estimate best alignment offset
def get_bit_sliding_window(data, sample_rate, bit_size, pos):
    start = pos - int(bit_size / 4)
    power_mark = []
    power_space = []
    for i in range(start, start + int(bit_size / 2)):
        block = data[i:i + bit_size]
        mark = goertzel(block, sample_rate, MARK_FREQ)
        space = goertzel(block, sample_rate, SPACE_FREQ)
        power_mark.append(mark)
        power_space.append(space)

    peaks_mark = find_peaks(power_mark)
    peaks_space = find_peaks(power_space)
    max_mark = max(power_mark)
    max_space = max(power_space)

    if max_mark > max_space:
        offset = power_mark.index(max_mark) - int(bit_size / 4)
        return 1, offset
    else:
        offset = power_space.index(max_space) - int(bit_size / 4)
        return 0, offset

if __name__ == "__main__":
    main()

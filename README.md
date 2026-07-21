# EAS-SAME lab 🧪

A Python toolkit for encoding and decoding EAS/SAME (Emergency Alert System/Specific Area Message Encoding) messages in WAV audio files.

## Features

- **Decode**: Extract EAS/SAME messages from WAV files
- **Encode**: Generate EAS/SAME messages and save as WAV files
- Professional argument parser with help and validation
- Support for attention tones and end-of-message markers
- Verbose mode for detailed processing information

## Installation

### System Dependencies

The encoder requires `ffmpeg` for audio file conversion. Install it using your system's package manager:

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Fedora:**
```bash
sudo dnf install ffmpeg
```

**Arch Linux:**
```bash
sudo pacman -S ffmpeg
```

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

### Python Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd eas-same-lab

# Install Python dependencies using uv
uv sync
```

## Usage

### Decoding EAS/SAME Messages

Decode an EAS/SAME message from a WAV file:

```bash
$ uv run decode.py same.wav
sample_rate = 48000
shape = (769156, 2)
length = 16.0s
bit_size = 92
ZCZC-CIV-RWT-000000+0300-832257-XDIF/004-
ZCZC-CIV-RWT-000000+0300-832257-XDIF/004-
ZCZC-CIV-RWT-000000+0300-832257-XDIF/004-
NNNN
NNNN
NNNN
```

#### Options

```bash
$ uv run decode.py --help
usage: decode.py [-h] [-v] [--version] input_file

Decode EAS/SAME (Emergency Alert System/Specific Area Message Encoding)
messages from WAV files.

positional arguments:
  input_file       Path to the WAV file containing the EAS/SAME message

options:
  -h, --help       show this help message and exit
  -v, --verbose    Enable verbose output with detailed processing information
  --version        show program's version number and exit

Examples:
  decode.py alert.wav
  decode.py --verbose emergency.wav
```

#### Example with verbose output

```bash
$ uv run decode.py --verbose test.wav
[INFO] Reading file: test.wav
sample_rate = 24000
shape = (252990,)
length = 10.5s
bit_size = 46
ZCZC-CIV-LOL-123456+7890-123456-XXXX/000-
ZCZC-CIV-LOL-123456+7890-123456-XXXX/000-
ZCZC-CIV-LOL-123456+7890-123456-XXXX/000-
NNNN
NNNN
NNNN
```

### Encoding EAS/SAME Messages

Generate an EAS/SAME message and save it as a WAV file:

```bash
$ uv run encode.py 'ZCZC-CIV-LOL-123456+7890-123456-XXXX/000-' test.wav
header = ZCZC-CIV-LOL-123456+7890-123456-XXXX/000-
output = test.wav
```

#### Options

```bash
$ uv run encode.py --help
usage: encode.py [-h] [-a] [--no-eom] [-v] [--version] header output_file

Encode EAS/SAME (Emergency Alert System/Specific Area Message Encoding)
messages to WAV files.

positional arguments:
  header                EAS/SAME header string (e.g., "ZCZC-CIV-RWT-
                        000000+0300-832257-XDIF/004-")
  output_file           Path to the output WAV file

options:
  -h, --help            show this help message and exit
  -a, --attention-tone  Include attention tone in the generated audio
  --no-eom              Disable end-of-message marker (NNNN)
  -v, --verbose         Enable verbose output
  --version             show program's version number and exit

Examples:
  encode.py 'ZCZC-CIV-RWT-000000+0300-832257-XDIF/004-' output.wav
  encode.py 'ZCZC-CIV-LOL-123456+7890-123456-XXXX/000-' test.wav --attention-tone
  encode.py --help

SAME Header Format:
  ZCZC-ORG-EEE-PSSCCC+TTTT-JJJHHMM-LLLLLLLL-

  Where:
    ZCZC       - Preamble (fixed)
    ORG        - Originator code (e.g., CIV, WXR, EAS)
    EEE        - Event code (e.g., RWT, TOR, EAN)
    PSSCCC     - Location codes (can be repeated, +PSSCCC)
    +TTTT      - Valid time period in minutes
    JJJHHMM    - Julian day and time
    LLLLLLLL   - Station callsign/identifier
```

#### Example with attention tone

```bash
$ uv run encode.py 'ZCZC-CIV-RWT-000000+0300-832257-XDIF/004-' alert.wav --attention-tone
header = ZCZC-CIV-RWT-000000+0300-832257-XDIF/004-
output = alert.wav
```

#### Example with verbose output

```bash
$ uv run encode.py 'ZCZC-CIV-RWT-000000+0300-832257-XDIF/004-' output.wav -v
[INFO] Encoding header: ZCZC-CIV-RWT-000000+0300-832257-XDIF/004-
[INFO] Output file: output.wav
[INFO] Attention tone: False
[INFO] End-of-message: True
header = ZCZC-CIV-RWT-000000+0300-832257-XDIF/004-
output = output.wav
[INFO] Successfully created output.wav
```

## SAME Message Format

The SAME (Specific Area Message Encoding) header follows this format:

```
ZCZC-ORG-EEE-PSSCCC+TTTT-JJJHHMM-LLLLLLLL-
```

Where:
- **ZCZC**: Preamble (fixed identifier)
- **ORG**: Originator code (e.g., CIV, WXR, EAS)
- **EEE**: Event code (e.g., RWT=Required Weekly Test, TOR=Tornado Warning)
- **PSSCCC**: Location codes (can be repeated with +PSSCCC for multiple areas)
- **+TTTT**: Valid time period in minutes
- **JJJHHMM**: Julian day and time (JJJ = day of year, HHMM = time)
- **LLLLLLLL**: Station callsign or identifier

## Technical Details

### Decoder
- Uses Goertzel algorithm for tone detection
- Supports MARK (2083.3 Hz) and SPACE (1562.5 Hz) frequencies
- Baud rate: 520.83 bits per second
- Automatic stereo to mono conversion
- Bandpass filter (1400-2200 Hz) for signal isolation

### Encoder
- Powered by EASGen library
- Generates standard-compliant EAS/SAME audio
- Optional attention tone (853/960 Hz dual-tone)
- Automatic end-of-message (NNNN) marker

## Complete Example Workflow

```bash
# 1. Encode a test message
$ uv run encode.py 'ZCZC-CIV-LOL-123456+7890-123456-XXXX/000-' test.wav

# 2. Verify by decoding
$ uv run decode.py test.wav
sample_rate = 24000
shape = (252990,)
length = 10.5s
bit_size = 46
ZCZC-CIV-LOL-123456+7890-123456-XXXX/000-
ZCZC-CIV-LOL-123456+7890-123456-XXXX/000-
ZCZC-CIV-LOL-123456+7890-123456-XXXX/000-
NNNN
NNNN
NNNN
```

## License

See LICENSE file for details.

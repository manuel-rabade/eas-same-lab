from EASGen import EASGen
import sys
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Encode EAS/SAME (Emergency Alert System/Specific Area Message Encoding) messages to WAV files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s 'ZCZC-CIV-RWT-000000+0300-832257-XDIF/004-' output.wav
  %(prog)s 'ZCZC-CIV-LOL-123456+7890-123456-XXXX/000-' test.wav --attention-tone
  %(prog)s --help

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
        '''
    )
    parser.add_argument(
        'header',
        help='EAS/SAME header string (e.g., "ZCZC-CIV-RWT-000000+0300-832257-XDIF/004-")'
    )
    parser.add_argument(
        'output_file',
        help='Path to the output WAV file'
    )
    parser.add_argument(
        '-a', '--attention-tone',
        action='store_true',
        help='Include attention tone in the generated audio'
    )
    parser.add_argument(
        '--no-eom',
        action='store_true',
        help='Disable end-of-message marker (NNNN)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0'
    )

    args = parser.parse_args()

    if args.verbose:
        print(f"[INFO] Encoding header: {args.header}")
        print(f"[INFO] Output file: {args.output_file}")
        print(f"[INFO] Attention tone: {args.attention_tone}")
        print(f"[INFO] End-of-message: {not args.no_eom}")

    print(f"header = {args.header}")
    print(f"output = {args.output_file}")

    Alert = EASGen.genEAS(
        header=args.header,
        attentionTone=args.attention_tone,
        endOfMessage=not args.no_eom
    )
    EASGen.export_wav(args.output_file, Alert)

    if args.verbose:
        print(f"[INFO] Successfully created {args.output_file}")

if __name__ == "__main__":
    main()

from EASGen import EASGen
import sys

header = sys.argv[1]
output = sys.argv[2]

print(f"header = {header}")
print(f"output = {output}")

Alert = EASGen.genEAS(header=header, attentionTone=False, endOfMessage=True)
EASGen.export_wav(output, Alert)

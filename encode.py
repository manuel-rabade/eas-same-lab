from EASGen import EASGen

header = "ZCZC-CIV-RWT-000000+0300-832257-XDIF/004-"
print(header)

Alert = EASGen.genEAS(header=header, attentionTone=False, endOfMessage=True)
EASGen.export_wav("test-same.wav", Alert)

# EAS-SAME lab 🧪

Decode:

```
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

Encode:

```
$ uv run encode.py 'ZCZC-CIV-LOL-123456+7890-123456-XXXX/000-' test.wav
header = ZCZC-CIV-LOL-123456+7890-123456-XXXX/000-
output = test.wav
```

Verify:

```
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

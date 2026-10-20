# Data files used

Download into `data/raw/` from `https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/<patient>/<file>`.

| Patient | Summary file      | Seizure files                                                        | Seizure-free files |
| ------- | ----------------- | -------------------------------------------------------------------- | ------------------ |
| chb01   | chb01-summary.txt | chb01_03, chb01_04, chb01_15, chb01_16, chb01_18, chb01_21, chb01_26 | chb01_01, chb01_02 |
| chb02   | chb02-summary.txt | chb02_16, chb02_16+, chb02_19                                        | chb02_01, chb02_02 |
| chb03   | chb03-summary.txt | chb03_01, chb03_02, chb03_03, chb03_04, chb03_34, chb03_35, chb03_36 | chb03_05, chb03_06 |
| chb05   | chb05-summary.txt | chb05_06, chb05_13, chb05_16, chb05_17, chb05_22                     | chb05_01, chb05_02 |

All files have the `.edf` extension. 30 recordings, 28.9 hours, 22 seizures.

In URLs, write `chb02_16+.edf` as `chb02_16%2B.edf`.

Example (Windows cmd):

```
for %f in (01 02 03 04 15 16 18 21 26) do curl -O https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/chb01/chb01_%f.edf
```

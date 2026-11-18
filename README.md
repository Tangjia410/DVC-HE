# DVC-HE

Reference implementation of **DVC-HE**, an open-set domain adaptation method for
cross-working-condition fault diagnosis.

DVC-HE combines a shared-backbone dual-expert architecture with an
openness-aware gated fusion, so that a single checkpoint can classify known
fault types and reject unknown ones on an unlabelled target domain.



## Requirements

```
python >= 3.9
torch
timm
numpy
scikit-learn
matplotlib
prettytable
openpyxl        # optional, only for the Excel report in Batch_testing.py
```



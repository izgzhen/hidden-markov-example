Homework 2: HMMs
================

Author: Zhen Zhang

NOTE: the code assumes a `data` folder that contains `twt.train.txt` etc., but
they are not included as part of the submission.


## Usage

```
python models.py <ADD_K_CONST_val> <DISCOUNT_val> <testset-name>
```

For example

```
python models.py 0.5 0.1 dev
```

To turn on error analysis mode:

```
MODE=ERROR_ANALYSIS python models.py 0.5 0.1 dev
```

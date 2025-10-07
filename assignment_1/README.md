# Assignment 1: Multiarm Bandit Problme

## Instructions
* Make sure your working directory is the "assignment_1" directory with the `pwd` command.
* Run the script `testbed.py` with `python testbed.py`.
* You can use the following flags to alter behavior:
    - `-s`, `--stationary`: will run with the stationary testbed
    - `-t`, `--nonstationary`: will run with the non-stationary testbed
    - `-n`, `--normal`: will run the testbed with normal residuals
    - `-l`, `--lognormal`: will run the testbed with lognormal residuals
    - `-e`, `--exponential`: will run the testbed with exponential residuals
* To recreate the results in the report, you can run: `python testbed.py -stnle`

## Dependencies
The only dependencies outside of the Python Standard Library are as follows:
* numpy
* pandas
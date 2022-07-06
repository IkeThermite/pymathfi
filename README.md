# pymathfi

A collection of math finance experiments in python.
The directory layout follows "Application with Internal Packages" from [Python Application Layouts: A Reference](https://realpython.com/python-application-layouts/).

Right now, the entry points to the code are `run_tests.py` and `run_examples.py`.

### To-do:
- **Analytical Formulae:**
    - [x] Black-Scholes Analytical Put and Call Prices
        - [x] Test: Black-Scholes Analytical Put and Call Prices
    - [x] Black-Scholes Analytical Put and Call Deltas
        - [ ] Test: Black-Scholes Analytical Put and Call Deltas
    - [x] CEV Analytical Put and Call Prices
        - [x] Test: CEV Analytical Put and Call Prices
    - [ ] CEV Analytical Put and Call Deltas
        - [ ] Test: CEV Analytical Put and Call Deltas
- **Finite Difference:**
    - **Products:**
        - [x] European Put
        - [x] European Call
    - **Solvers:**
        - [x] 1-dimensional Theta Method
    - **Models:**
        - [x] Black-Scholes
        - [x] CEV
    - **Tests:**
        - [x] 1D Theta Method, Black-Scholes, Put
        - [x] 1D Theta Method, Black-Scholes, Call
        - [x] 1D Theta Method, CEV, Put
        - [x] 1D Theta Method, CEV, Call
    - **Examples:**
        - [ ] 1D Theta Method, Black-Scholes, Put
        - [ ] 1D Theta Method, CEV, Put
- **Monte Carlo:**
    - **Products:**
        - [x] European Put
        - [x] European Call
    - **Solvers:**
        - [ ] Crude Monte Carlo
        - [ ] Crude Monte Carlo with Targeted Accuracy
    - **Models:**
        - [x] Black-Scholes
        - [ ] CEV
    - **Tests:**
        - [ ] Crude Monte Carlo, Black-Scholes Put
        - [ ] Crude Monte Carlo, Black-Scholes Call
    - **Examples:**
        - [x] Path Simulation, Black-Scholes

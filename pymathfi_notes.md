# PyMathFi

[https://realpython.com/python-application-layouts/#application-with-internal-packages](Python Application Layouts: A Reference)

```bash
helloworld/
│
├── helloworld/
│   ├── __init__.py
│   ├── helloworld.py
│   └── helpers.py
│
├── tests/
│   ├── helloworld_tests.py
│   └── helpers_tests.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

What should the ``__init__.py`` look like?
[https://realpython.com/python-modules-packages/#package-initialization](Python Modules and Packages: An Introduction)

```python
print(f'Invoking __init__.py for {__name__}')
```

Note that the above is using *f-strings*, available from Python 3.6. More: [https://realpython.com/python-f-strings/](Python 3's f-Strings: An Improved String Formatting Syntax)

## Anaconda Environments and Emacs
This answer tells me how to activate conda environments from inside Emacs [https://github.com/jorgenschaefer/elpy/issues/285](Issue)

To get a list of current environments and their locations I ran ``conda env list``. This yields
```bash
# conda environments:
#
base                  *  C:\Users\thera\Anaconda3
mlcc                     C:\Users\thera\Anaconda3\envs\mlcc
pelicansite              C:\Users\thera\Anaconda3\envs\pelicansite
torchenv                 C:\Users\thera\Anaconda3\envs\torchenv
```
Now I can try: ``M-x pyvenv-activate`` followed by ``<path>``. Works!
Wrong. The environment activates, but I can't import numpy. This problem doesn't exist in the anaconda prompt so
elpy isn't playing nice with conda.




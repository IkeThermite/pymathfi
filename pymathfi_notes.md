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

## Pulling my old pieces of jumbled code into pymathfi
[https://saintgimp.org/2013/01/22/merging-two-git-repositories-into-one-repository-without-losing-file-history/](This)
seems to be what I want to do.
Okay, doesn't quite do what I want because ``pymathfi`` already had a history.
So I had to add the ``--allow-unrelated-histories`` option to the ``merge`` command.
I got this idea from [https://github.community/t5/How-to-use-Git-and-GitHub/How-to-deal-with-quot-refusing-to-merge-unrelated-histories-quot/m-p/16305#M5000](here).

## Python Modules & Packages
Alright, learning from [https://realpython.com/python-modules-packages/](here):

```bash
pkg/
│
├── __init__.py
├── mod1.py
├── mod2.py
```
This seems fairly straightforward.

## Unit Tests
Learning from [https://realpython.com/python-testing/](here).
However, once I've split my tests into a separate folder, what must the import statement look like?
Apparently, this is well known to all developers, except NOOBS, but luckily one of those asked the question [https://stackoverflow.com/questions/1896918/running-unittest-with-typical-test-directory-structure](here).

So there's no fancy import statement; instead you make use of ``unittest``'s command line tools. Two things to note:
1. Tests must be named ``test_*.py``.
2. Tests must reside in a packge, so you have to make an ``__init__.py``.

Now, from the root project folder, I can run:

```bash
python -m unittest discover
```

You can also set this up in Visual Studio Code using the command palette (and the python plugin). Then the tests can be run from the bar at the bottom of the screen. Pretty neat.
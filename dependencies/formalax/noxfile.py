#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.

import nox

nox.options.sessions = ["tests"]
nox.options.default_venv_backend = "uv|virtualenv"


@nox.session(python=["3.12", "3.13", "3.14"])
def tests(session):
    session.install(".[test]")
    session.run("pytest")


# @nox.session(python=["3.12"])
# def benchmark(session):
#     session.install(".[test]")
#     session.run("python", "-m", "pytest", "--benchmark", "-m", '"benchmark"')

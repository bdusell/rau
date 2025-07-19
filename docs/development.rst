Development
===========

This page is for people who want to set up a development environment for Rau.

Setup
-----

Clone the repo:

.. code-block:: sh

    git clone git@github.com:bdusell/rau.git
    cd rau

Optional: To simplify software installation, set up the pre-made Docker
container defined in this repo. To do this, install
`Docker <https://www.docker.com/get-started>`_
and the
`NVIDIA Container Toolkit <https://www.docker.com/get-started>`_
(for GPU support), then run this script in order to start a shell inside of the
container:

.. code-block:: sh

    bash scripts/docker_shell.bash --build

(If you don't have an NVIDIA GPU, don't install the NVIDIA Container Toolkit,
and run the above command with the flag ``--cpu`` added.)

Install Python dependencies. This can be done by installing the package manager
`Poetry <https://python-poetry.org/docs/#installation>`_
(it's already installed in the Docker container) and running this script:

.. code-block:: sh

    bash scripts/install_python_packages.bash

Start a shell inside the Python virtual environment using Poetry:

.. code-block:: sh

    bash scripts/poetry_shell.bash

You are now ready to use Rau.

Running Unit Tests
------------------

To run all unit tests, run this script (either inside or outside the Poetry
shell):

.. code-block:: sh

    bash scripts/run_tests.bash

You can run individual unit tests with the ``pytest`` command.

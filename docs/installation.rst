=============
Installation
=============

At the command line:

.. code-block:: bash

    pip install laser-measles


You can also install the in-development version with:

.. code-block:: bash

    pip install git+https://github.com/InstituteforDiseaseModeling/laser-measles.git@main

Development
===========

To run all the tests run:

.. code-block:: bash

    tox

And to build the documentation run:

.. code-block:: bash

    tox -e docs

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox

You can check that the bump versioning works by running:

.. code-block:: bash

    uvx bump-my-version bump minor --dry-run -vv
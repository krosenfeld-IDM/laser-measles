=============
API reference
=============

.. currentmodule:: laser_measles

This page lists laser-measles's API.

Core Framework
==============

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :recursive:
   :nosignatures:

   demographics
   generic
   nigeria

Biweekly Model
==============

.. currentmodule:: laser_measles.biweekly

Core Model
----------

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   model
   params
   base

Model Processes
---------------

Components that modify population states and drive model dynamics:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   components.InfectionProcess
   components.VitalDynamicsProcess

Trackers
--------

Components that monitor and record model state for analysis:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   components.StateTracker
   components.FadeOutTracker

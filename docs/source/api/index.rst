Core API
========

This section lists the stable user-facing API. Most workflows use one of the
three model classes, then call ``setup_data()``, a test method, and
``get_formatted_test_results()``. Lower-level extension points and backend
classes are documented separately in :doc:`advanced`.

The pages below split the API by how users usually work with SPLISOSM, so the
left navigation stays compact while still exposing the main public methods.

.. toctree::
   :maxdepth: 2

   core
   helpers
   io

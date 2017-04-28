Examples
==========

Here are some code examples to test out the theme. 

Sub-Heading
-----------
This is a second level heading (``h2``).

Sub-Sub-Heading
~~~~~~~~~~~~~~~
This is a third level heading (``h3``).


Code
----

Here is some  ``inline code text`` and::

    multiline
    code text

It also works with existing Sphinx highlighting:

.. code-block:: html

    <html>
      <body>Hello World</body>
    </html>

.. code-block:: python

    def hello():
        """Greet."""
        return "Hello World"

.. code-block:: javascript

    /**
     * Greet.
     */
    function hello(): {
      return "Hello World";
    }


Admonitions
-----------

See Also
~~~~~~~~

.. seealso:: This is a **seealso**.

.. seealso::

    This is a longer seealso. It might also contain links to our code such as a
    link to :func:`convert_notebooks <klink.__init__.convert_notebooks>` and it may also
    simply contain a normal hyperlink to http://www.google.com.

Note
~~~~
.. note:: This is a **note**.

.. note::

    This is a longer note. It might also contain links to our code such as a
    link to :func:`convert_notebooks <klink.__init__.convert_notebooks>` and it may also
    simply contain a normal hyperlink to http://www.google.com.

Warning
~~~~~~~
.. warning:: This is a **warning**.

.. warning::

    This is a longer warning. It might also contain links to our code such as a
    link to :func:`convert_notebooks <klink.__init__.convert_notebooks>` and it may also
    simply contain a normal hyperlink to http://www.google.com.

Danger
~~~~~~
.. danger:: This is **danger**-ous.

.. danger::

    This is a longer danger. It might also contain links to our code such as a
    link to :func:`convert_notebooks <klink.__init__.convert_notebooks>` and it may also
    simply contain a normal hyperlink to http://www.google.com.

Footnotes
---------
I have footnoted a first item [#f1]_ and second item [#f2]_.

.. rubric:: Footnotes
.. [#f1] My first footnote.
.. [#f2] My second footnote.

Tables
------
Here are some examples of Sphinx
`tables <http://sphinx-doc.org/rest.html#rst-tables>`_. 

Grid
~~~~

+------------------------+------------+----------+----------+
| Header1                | Header2    | Header3  | Header4  |
+========================+============+==========+==========+
| row1, cell1            | cell2      | cell3    | cell4    |
+------------------------+------------+----------+----------+
| row2 ...               | ...        | ...      |          |
+------------------------+------------+----------+----------+
| ...                    | ...        | ...      |          |
+------------------------+------------+----------+----------+

Simple
~~~~~~

=====  =====  =======
H1     H2     H3
=====  =====  =======
cell1  cell2  cell3
...    ...    ...
...    ...    ...
=====  =====  =======

Code Documentation
~~~~~~~~~~~~~~~~~~

An example Python function.

.. py:function:: format_exception(etype, value, tb[, limit=None])

   Format the exception with a traceback.

   :param etype: exception type
   :param value: exception value
   :param tb: traceback object
   :param limit: maximum number of stack frames to show
   :type limit: integer or None
   :rtype: list of strings

An example JavaScript function.

.. js:class:: MyAnimal(name[, age])

   :param string name: The name of the animal
   :param number age: an optional age for the animal

IPython Notebook
----------------

This is what Notebook integration looks like:

.. include:: nb-examples.rst

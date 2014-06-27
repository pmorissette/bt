# Notebook sphinx extensions

## Code snippets in documentation

This packages two useful [Sphinx](http://sphinx-doc.org/) extensions: `notebook`
and `notebook-cell`. These extensions are useful for embedding entire
notebooks or single notebook cells, respectively, inside sphinx documentation.

In the past, it was relatively straightforward to include example scripts inside
of version controlled documentation. For example, one could include code
snippets inside of sphinx documentation using the rst `code-block` directive:

```rst
.. code-block:: python

   for i in range(5):
     print i

```

While this does produce a syntax highlighted python script embedded in a sphinx
document, it does not run the code or provide any facilities for checking whether
the code is correct.

[IPython](http://ipython.org) notebooks offer a powerful environment for
literate programming, with code input, output, and explanatary text embedded
into a single document. It's tempting to include notebooks into documentation
wholesale. However, there are some issues with this approach as
well. Versioning notebooks is difficult - output can change and if the notebook
output contains large amounts of data, the diffs can easily grow quickly,
producing an inconveniently large repository. Versioning evluated notebooks also
offers no guarantee that the code in the notebook is still functional.

## Using Sphinx Extensions to Automate Notebook Running

The extension included in this package make it easy to include unevaluated
notebooks or short python code snippets inside of documentation. The extension
make use of [runipy](https://github.com/paulgb/runipy) to script the evaluation
of notebooks and
[nbconvert](http://ipython.org/ipython-doc/rel-1.1.0/interactive/nbconvert.html)
to conver the resulting evaluated notebooks into HTML suitable for embedding in
a Sphinx document.

## Dependencies

This extension has two required dependencies:

* runipy
* IPython

Note that all IPython dependencies (even the optional ones) must be
installed. In particular, [pandoc](http://johnmacfarlane.net/pandoc/) and
[node.js](http://nodejs.org/) must be available since these are used by
nbconvert.

## Examples

Suppose I want to include a notebook named `example.ipynb` inline in my
documentation. To do so, add the following to any sphinx ReStructuredText
document:

```rst

.. notebook:: example.ipynb

```

During preprocessing, sphinx will evaluate the notebok, convert it to html, and
embed it into the document in the place where the `notebook` directive was
used.

If a full notebook does not make sense or if you would like to more tightly link
a script to the source of your documentation, you can use `notebook-cell` to
embed a single-cell mini notebook:

```rst

.. notebook-cell::

   for i in range(5):
     print i

```

This will convert the code snippet into a notebook, evaluate the notebook, and
then embed the result in the document. Note that notebook-cell does not
currently accept a user namespace, so all imports necessary for the code to run
must be included in the source.

## Known issues

These extensions use a version of the 'full' HTML output from the nbconvert HTML
output. This includes the full notebook CSS which will probably conflict with
your documentation theme. There are some monkeypatches to reduce the impact of
the notebook CSS on the document, but it is like that the monkeypatching is
fragile.

Images are embedded directly in the document HTML, just as they are in an
IPython notebook. This can easily create multi-megabytes pages that some web
browsers have trouble with.

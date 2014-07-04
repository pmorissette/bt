klink - A Simple & Clean Sphinx Theme
=====================================

Klink is a **simple** and **clean** theme for creating `Sphinx docs
<http://sphinx-doc.org/>`__. It is heavily inspired by the beautiful `jrnl theme
<https://github.com/maebert/jrnl>`__. It also supports embedding `IPython
Notebooks <http://ipython.org/notebook.html>`__ which can be mighty useful.

Options
-------

Here are the theme options. They should be added to the html_theme_options in
your **conf.py** file.

* **github**
    The github address of the project. The format is name/project
    (pmorissette/klink).
* **logo**
    The logo file. Assumed to be in the _static dir. Default is logo.png. The logo
    should be 150x150.
* **analytics_id**
    Your Google Analytics id (usually starts with UA-...)

IPython Notebook Integration
----------------------------

With the klink helper function :func:`convert_notebooks()
<klink.__init__.convert_notebooks>`, all notebooks will be
converted to .rst so that they can be included in your docs. This includes all
output including images. It’s a very convenient way to create Python docs! 

All you have to do is create notebooks within your source directory (same directory
as your conf.py file). Then, you add a call to klink.convert_notebooks() in your
conf.py. You can also mix in **Mardown** cells or **Raw NBConvert** cells in
your workbook. These will be converted to rst as well. 

.. note::

    If you use the Raw NBConvert type cells, add a blank line at the start. There
    seems to be a bug in the rst conversion and if the cell does not begin with a
    blank line, you may run into some issues. 

Using a Raw NBConvert cell with rst text inside is convenient, especially if you
want to have links to other parts of your Sphinx docs. 

.. danger::

    Do not name your Notebooks with the same name as another reST file in your
    source directory as the file will be **overwritten** when calling convert_notebooks. 

.. include:: intro.rst

.. toctree::
    :hidden: 
    :maxdepth: 2

    Overview <index>
    Installation Guide <install>
    Examples <examples>
    API <klink>

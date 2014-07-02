.. image:: docs/source/_static/logo.png


klink - A Simple & Clean Sphinx Theme
=====================================

Klink is a **simple** and **clean** theme for creating `Sphinx docs
<http://sphinx-doc.org/>`__. It is heavily inspired by the beautiful `jrnl theme
<https://github.com/maebert/jrnl>`__. It also supports embedding `IPython
Notebooks <http://ipython.org/notebook.html>`__ which can be mighty useful.

For a live demo, please visit `our docs <http://pmorissette.github.io/klink/>`__.

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

With the klink helper function **klink.convert_notebooks()**, all notebooks will be
converted to .rst so that they can be included in your docs. This includes all
output including images. It’s a very convenient way to create Python docs! 

All you have to do is create notebooks within your source directory (same directory
as your conf.py file). Then, you add a call to klink.convert_notebooks() in your
conf.py.

Installation
------------

Assuming you have pip installed:

.. code:: sh

    $ pip install klink

That's it.

Usage
-----

In your docs' **conf.py** file, add the following:

.. code:: python

    import klink

    html_theme = 'klink'
    html_theme_path [klink.get_html_theme_path()]
    html_theme_options = {
        'github': 'yourname/yourrepo',
        'analytics_id': 'UA-your-number-here',
        'logo': 'logo.png'
    }

Klink also comes with a useful helper function that allows you to integrate an
IPython Notebook into a .rst file. It basically converts the Notebook to .rst
and copies the static data (images, etc) to your _static dir. 

If you have IPython Notebooks that you would like to integrate, use the
following code to your **conf.py**:

.. code:: python

    klink.convert_notebooks()

Once the conversion is done, you will have a .rst file with the same name as
each one of your notebooks.


*NOTE: Place your notebooks in your docs' source dir.*

Now all you have to do is use the **include** command to insert them into your
docs.


Customization
-------------

Obviously, some of you will want to customize the theme. The easiest way to
achieve this is to clone the repo into your _themes folder (create it if it does
not exist in your docs' source dir). To change the style, I recommend editing
the LESS files themselves. You will also need lessc to convert from less to css.
See the css command in the Makefile for an example. 

You will also need to change your conf.py file. The following settings should
work::

    html_theme = 'klink'
    html_theme_path ['_themes']
    html_theme_options = {
        'github': 'yourname/yourrepo',
        'analytics_id': 'UA-your-number-here',
        'logo': 'logo.png'
    }


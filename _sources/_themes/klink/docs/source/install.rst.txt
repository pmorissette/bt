Installation Guide
==================

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

.. note::

    Place your notebooks in your docs' source dir. **Do not** give them the same
    name as another reST file as this file will be **overwriiten** when you call
    klink.convert_notebooks.

Now all you have to do is use the **include** command to insert them into your
docs.


Customization
-------------

Obviously, some of you will want to customize the theme. The easiest way to
achieve this is to clone the repo into your _themes folder (create it if it does
not exist in your docs' source dir). To change the style, I recommend editing
the LESS files themselves. You will also need lessc to convert from less to css.
See the css command in the Makefile for an example. 

You may also want to explore the option of using **git subtree**. Here is a good
`intro tutorial <http://makingsoftware.wordpress.com/2013/02/16/using-git-subtrees-for-repository-separation/>`__.

You will also need to change your conf.py file. The following settings should
work::

    html_theme = 'klink'
    html_theme_path ['_themes']
    html_theme_options = {
        'github': 'yourname/yourrepo',
        'analytics_id': 'UA-your-number-here',
        'logo': 'logo.png'
    }


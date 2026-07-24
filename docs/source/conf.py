import os
import sys

project = 'MuLaConf'
copyright = '2026, Kostas Katsios'
author = 'Kostas Katsios'
release = '0.3.0'

extensions = [  'sphinx.ext.autodoc',
                'sphinx.ext.napoleon',
                'sphinx_copybutton',
                'myst_parser',
                'sphinx.ext.viewcode',
                'sphinx.ext.intersphinx'
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/k-kostas/MuLaConf",
}

sys.path.insert(0, os.path.abspath('../../src'))

autodoc_member_order = 'bysource'

autodoc_mock_imports = ["torch", "scipy", "numpy", "pandas", "sklearn"]

copybutton_prompt_text = r">>> ?|\.\.\. ?"
copybutton_prompt_is_regexp = True

html_title = 'MuLaConf v. 0.3.0'

html_sidebars = {
    "**": []
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

myst_enable_extensions = [
    "dollarmath"
]

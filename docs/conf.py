# Sphinx configuration for TyGrit documentation

project = "TyGrit"
author = "Hu Tianrun"
copyright = "2025, Hu Tianrun"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

# MyST-Parser
myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
    "html_image",
]
myst_raw_html = True

# Theme
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "TyGrit"

# Autodoc
autodoc_member_order = "bysource"
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "mani_skill",
    "vamp_preview",
    "GraspGen",
    "pointnet2_ops",
    "spconv",
    "ikfast_fetch",
    "pytracik",
    "rospy",
    "actionlib",
    "control_msgs",
    "trajectory_msgs",
    "moveit_msgs",
    "sensor_msgs",
    "geometry_msgs",
    "tf",
    "cv2",
    "trimesh",
    "pyrender",
    "meshcat",
    "h5py",
    "tensordict",
    "diffusers",
    "timm",
    "webdataset",
    "sharedarray",
    "sapien",
]

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Source settings
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
exclude_patterns = ["_build"]

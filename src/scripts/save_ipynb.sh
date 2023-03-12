#!/bin/sh
# To change the permissions of this file so it can be run in terminal, do
# chmod 755 <filename>.sh
# chmod 755 src/scripts/save_ipynb.sh
# to run
# ./src/scripts/save_ipynb.sh

# This will work for any ipynb files inside src/notebooks including any subfolders

# Available export formats are:
# html, python, script, pdf, webpdf, markdown, rst, latex, asciidoc, slides (reveal.js)

# For converting to html, the following arguments can be used
# --embed-images  <- will embed images as base64 urls in the resulting HTML file.
# --template classic  <- Simplified HTML, using the classic jupyter look and feel.
# --template basic    <- Base HTML, rendering with minimal structure and styles.
# --template lab      <- (default) A full static HTML render of the notebook. This looks very similar to the JupyterLab interactive view. The lab template supports the extra --theme option, which defaults to light
jupyter nbconvert --to html src/notebooks/**/*.ipynb

# "--to script" was formerly called "--to python"
# Source: https://stackoverflow.com/a/19779226/2879686
jupyter nbconvert --to script src/notebooks/**/*.ipynb

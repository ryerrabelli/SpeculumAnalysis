#!/bin/sh
# To change the permissions of this file so it can be run in terminal, do
# chmod 755 <filename>.sh
# chmod 755 src/scripts/save_ipynb.sh
# to run
# ./src/scripts/save_ipynb.sh
jupyter nbconvert --to html src/notebooks/*.ipynb
jupyter nbconvert --to python src/notebooks/*.ipynb
#!/bin/sh
# Remove all .h5-files

OUTPUT_DIR="./outputs/"

# move to sim/DREAM/ if not already here
cd "${0%/*}"

# remove
rm -r $OUTPUT_DIR*.h5
rm *.h5

#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for i in $(seq -f "%04g" 0 5 200)
    do python detect_single_threaded.py --tracker 2 --video /home/drussel1/data/custom/$i.mp4
done

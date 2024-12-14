#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
echo "running Med-NCA train2.py"
for i in {1..10}
do
    echo "Loop ITERATION $i"
    python /home/squirrel/dev/MDN/Neural-Cellular-Automata/med-nca/train2.py
    echo "Loop complete"
    sleep 10
done

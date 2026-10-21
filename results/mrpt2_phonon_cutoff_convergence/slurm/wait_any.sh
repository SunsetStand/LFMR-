#!/bin/bash
# Block until any of the given Slurm job IDs is no longer queued/running; print which ones ended.
ids="$*"
while true; do
  live=$(squeue -h -j $(echo $ids | tr ' ' ',') -o %i 2>/dev/null | tr '\n' ' ')
  gone=""
  for i in $ids; do [[ " $live " == *" $i "* ]] || gone="$gone $i"; done
  if [ -n "$gone" ]; then echo "ended:$gone"; break; fi
  sleep 60
done

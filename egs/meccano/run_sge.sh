#!/usr/bin/env bash
#$ -wd /home/jdjones/data/output/grid_logs
#$ -V
#$ -N ikea_anu_pipeline
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M jdjones@jhu.edu
#$ -m e
#$ -l ram_free=10G,mem_free=10G

set -ue

conda activate kinemparse
cd /home/jdjones/repo/kinemparse/egs/meccano

./run.sh "$@"

#!/bin/bash
#SBATCH --job-name=stitch_generate
#SBATCH --output=stitch_generate_%j.log
#SBATCH --error=stitch_generate_%j.err
#SBATCH --account=mst115022
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1500G
#SBATCH --time=4:00:00

set -euo pipefail

cd "/work/u1007247/agent_stitch_s_pipeline"
export MODE="${MODE:-generate}"
exec bash run_pipeline.sh

#!/bin/bash
# Template: strong-scaling sweep of viz_mpi.py on a SLURM cluster.
# Adjust partition/account/modules to the site, then:  sbatch run_scaling.slurm
#SBATCH --job-name=viz-scaling
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=viz-scaling-%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME

# module load openmpi python        # site-specific
# source activate physics_305_hpc_env

N=512          # grid points per axis
REPEATS=3
export OMP_NUM_THREADS=1

for p in 1 2 4 8 16 32; do
  for rep in $(seq 0 $((REPEATS - 1))); do
    mpirun -n "$p" python viz_mpi.py --n "$N" --no-images --out "results/slurm_p${p}_r${rep}"
  done
done

# Merge the per-run records into the file the notebook reads (cells D4-D5).
python - <<'EOF'
import glob, json, re
runs = []
for f in sorted(glob.glob("results/slurm_p*_r*/timing.json")):
    r = json.load(open(f))
    r["repeat"] = int(re.search(r"_r(\d+)", f).group(1))
    runs.append(r)
json.dump(runs, open("results/scaling.json", "w"), indent=1)
print(len(runs), "runs written to results/scaling.json")
EOF

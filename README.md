This repository contains all code for the paper "Cross-Domain and Cross-Dataset Adaptations for RF Fingerprinting Using Prototypical Networks" as well as for my Master's thesis.

# Where are the results?
Results are aggregated in the notebook `csc500-notebooks/analysis/analysis.ipynb`.


# Preparing Environment
This project uses submodules as well as python virtual environment.  
```
git submodule update --init --recursive
python3 -m venv ./venv
source venv/bin/activate
source source_this_after_venv.sh
pip install -r requirements.txt
```

# Reproducing Results
All experiments use deterministic algoriths, so they should be reproducible (though I have not tested this between disparate hardware/torch/cuda versions).  
All experiments were run on CUDA Version: 11.4  
  
First follow the steps in "Preparing Environment", then pick any experiment in "csc500-notebooks/experiments" and run "run_all.sh".  
This project uses the python module "Papermill" in order to parameterize jupyter notebooks and execute them programatically. In this way we can keep record of how each experiment was run.

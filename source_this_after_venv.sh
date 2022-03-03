ulimit -n $(ulimit -Hn)
export PYTHONPATH=$(realpath csc500-utils):$PYTHONPATH
export PYTHONPATH=$(realpath csc500-models):$PYTHONPATH
export DATASETS_ROOT_PATH=$PWD/datasets/
export EXPERIMENTS_ROOT_PATH=$PWD/csc500-notebooks/experiments/
export CSC500_ROOT_PATH=$PWD
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ulimit -n $(ulimit -Hn)
export PYTHONPATH=$(realpath csc500-utils):$PYTHONPATH
export PYTHONPATH=$(realpath csc500-models):$PYTHONPATH
export DATASETS_ROOT_PATH=$PWD/datasets/
export CSC500_ROOT_PATH=$PWD
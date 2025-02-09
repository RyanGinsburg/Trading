conda create --name my_env python=3.9
conda activate my_env
python -m pip install --upgrade pip
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 openblas blas libgcc
pip install "numpy<2" "tensorflow<2.11"
python gpu.py
conda deactivate

conda activate my_env
# ML-SemReg
code for `ML-SemReg: Boosting Point Cloud Registration with Multi-Level Semantic Consistency`

# Env
```shell
conda create -n mlsemreg python=3.9 
conda activate mlsemreg
pip install -r requirements.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

# demo

```shell
python -m demo
python -m demo -is_use_baseline
python -m demo -is_vis
python -m demo -is_vis -is_use_baseline
```

# About semantic label

[Ref Pointcept](https://github.com/Pointcept/Pointcept)


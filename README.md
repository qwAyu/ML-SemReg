# ML-SemReg
code for `ML-SemReg: Boosting Point Cloud Registration with Multi-Level Semantic Consistency`

# Env
```shell
conda create -n mlsemreg python=3.9 
conda activate mlsemreg
pip install -r requirements.txt

# please check your CUDA version
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

# demo

a demo using KITTI medium dataset

```shell
python -m demo
python -m demo -is_vis
```


# Acknowledgements 

- [Pointcept](https://github.com/Pointcept/Pointcept)
- [SC2-PCR](https://github.com/ZhiChen902/SC2-PCR)

# First Order DARTS in PyTorch1.5.1, PaddlePaddle1.8.2, and Tensorflow2.2.0

## Introduction
As a toy project, this repository provides implementations of the first order [Differentiable Architecture Search (DARTS)](https://arxiv.org/abs/1806.09055) on the [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) dataset in three different frameworks.

I've just finished this project on my Xiaomi laptop with a 6GB Nvidia RTX 2060 GPU card. For more information about the development environment please refer to [this page](https://github.com/woodywff/dev-env-setup/blob/master/mi-laptop-rtx2060.md).

All of the three flavors (frameworks) have similar interfaces and storylines which are mostly self-explained.
To get started, just have the `Main.ipynb` run and follow its lead.
If you'd prefer not to play with it in Jupyter Notebook, do not forget to make the change from `tqdm.notebook` to `tqdm`.
The dataset (.gz) has been put under `darts_pt_pp_tf/data/fmnist/`.
The way I tune the configurations is to modify the `config.yml`.


## The main storyline
- We prepare primary operations (op).
- With op at hand, we are free to construct Cells. Normal Cell with `stride=1`, Reduction Cell with `stride=2`.
- Define the Kernel network which is piles of Cells.
- Encapsulate the Kernel network with the Shell network who has two more trainable parameters --- `alphas`.
- Searching process:
    - Update the trainable parameters of Kernel.
    - Update `alphas` in Shell.
    - Save the best-searched Cells.
- Training process:
    - Reconstruct the Kernel network with searched Cells.
    - Training and Validation.
    - Save the best model.
- Prediction process:
    - Load the best model.
    - Prediction.

The parameter update process and the training, validation processes all follow the procedures like:
- Get x, y from the data pipeline.
- Get loss value (forward).
- Backpropagation.
- Gradient descent on certain parameters.
    

## Something
- The `affine` argument in Batch Normalization is set `False` for the Searching process and `True` for the Training process.

- For `ReduceLROnPlateau`:
`patience=10, factor=0.5`
We didn't put these arguments in `config.yml` for simplicity.

- Don't iter the variable returned by `fluid.layers.create_parameter`, it will not stop at the end but give out the out boundary error.

- For tf 2.2.0:
we need this:
`tf.config.experimental.set_memory_growth(gpu_check[0], True)`
otherwise, there would be the OOM problem on my laptop.

---
## References
[woodywff/nas_3d_unet](https://github.com/woodywff/nas_3d_unet)

[quark0/darts](https://github.com/quark0/darts) 

[PaddlePaddle/PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)

[peteryuX/pcdarts-tf2](https://github.com/peteryuX/pcdarts-tf2)

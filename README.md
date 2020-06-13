# nas_playground

we've changed the drop_path()
search: affine=False in BN
train: affine=True in BN


For ReduceLROnPlateau:
patience=10, factor=0.5
We didn't put these in config.yml for simplicity.


`fluid.layers.create_parameter` doesn't support iter like:
```
for i in alphas:
    pass
```
it will not stop at the end but give out the out boundary error.


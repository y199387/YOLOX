## Modified Description

<details>
<summary>1. Subclass TorchNano</summary>

- Example
    ```python
    from bigdl.nano.pytorch import TorchNano
    class MyNano(TorchNano)
        ...
    ```
[yolox/core/trainer.py#L37](https://github.com/y199387/YOLOX/blob/main/yolox/core/trainer.py#L37)

</details>

<details>
<summary>2. Override `TorchNano` `train` method</summary>

- Example
    ```python
    class MyNano(TorchNano):
        def train():
            self.brefore_train()
            self.train_in_epoch()
            self.after_train()
    ```

[yolox/core/trainer.py#L77-L84](https://github.com/y199387/YOLOX/blob/main/yolox/core/trainer.py#L77-L84)

</details>

<details>
<summary>3. Remove all `.to(...)`, `.cuda()` etc calls</summary>

- Example
    ```python
    # remove call to .to(device)
    # model = Net.to(self.device)
    model = Net()
    ```

[utils/allreduce_norm.py#L44](https://github.com/y199387/YOLOX/blob/main/yolox/utils/allreduce_norm.py#L44)  
[yolox/core/trainer.py#L134](https://github.com/y199387/YOLOX/commit/3cc3528c373553604f4a86e82903a9eb52886436#diff-553d2d392dc26adff45a30c6c35b9aba76d3fc68be5206b52c7ba39040eb7524L134)  
[yolox/core/trainer.py#L139](https://github.com/y199387/YOLOX/commit/3cc3528c373553604f4a86e82903a9eb52886436#diff-553d2d392dc26adff45a30c6c35b9aba76d3fc68be5206b52c7ba39040eb7524L139)
</details>

<details>
<summary>4. Remove the precision-specific logic</summary>

- Example
    ```python
    # Remove the `torch.amp`
    # with torch.cpu.amp.autocast():
    output = self.model(input)
    ```

[yolox/core/trainer.py#L46](https://github.com/y199387/YOLOX/commit/3cc3528c373553604f4a86e82903a9eb52886436#diff-553d2d392dc26adff45a30c6c35b9aba76d3fc68be5206b52c7ba39040eb7524L46)  
[yolox/core/trainer.py#L104](https://github.com/y199387/YOLOX/commit/3cc3528c373553604f4a86e82903a9eb52886436#diff-553d2d392dc26adff45a30c6c35b9aba76d3fc68be5206b52c7ba39040eb7524L104)  
[yolox/core/trainer.py#L110-112](https://github.com/y199387/YOLOX/commit/3cc3528c373553604f4a86e82903a9eb52886436#diff-553d2d392dc26adff45a30c6c35b9aba76d3fc68be5206b52c7ba39040eb7524L110-L112)

</details>

<details>
<summary>5. Replace the process-specific logic with TorchNano's attributes</summary>

- Example
    ```python
    # rank = get_rank()
    # local_rank = get_local_rank()
    # world_size = get_world_size()
    # if rank == 0:
    if self.global_rank == 0:
        ...
    ```

[yolox/core/trainer.py#L204](https://github.com/y199387/YOLOX/commit/3cc3528c373553604f4a86e82903a9eb52886436#diff-553d2d392dc26adff45a30c6c35b9aba76d3fc68be5206b52c7ba39040eb7524L179)  
[yolox/core/trainer.py#L223](https://github.com/y199387/YOLOX/commit/3cc3528c373553604f4a86e82903a9eb52886436#diff-553d2d392dc26adff45a30c6c35b9aba76d3fc68be5206b52c7ba39040eb7524L198)  
[yolox/core/trainer.py#L291](https://github.com/y199387/YOLOX/commit/3cc3528c373553604f4a86e82903a9eb52886436#diff-553d2d392dc26adff45a30c6c35b9aba76d3fc68be5206b52c7ba39040eb7524L198)  
[yolox/core/trainer.py#L304](https://github.com/y199387/YOLOX/commit/3cc3528c373553604f4a86e82903a9eb52886436#diff-553d2d392dc26adff45a30c6c35b9aba76d3fc68be5206b52c7ba39040eb7524L277)  
[yolox/core/trainer.py#L362](https://github.com/y199387/YOLOX/commit/3cc3528c373553604f4a86e82903a9eb52886436#diff-553d2d392dc26adff45a30c6c35b9aba76d3fc68be5206b52c7ba39040eb7524L277)  
[yolox/core/trainer.py#L381](https://github.com/y199387/YOLOX/commit/3cc3528c373553604f4a86e82903a9eb52886436#diff-553d2d392dc26adff45a30c6c35b9aba76d3fc68be5206b52c7ba39040eb7524L354)  

</details>

<details>
<summary>6. Apply `setup()` over each model and optimizers pair and all dataloaders. </summary>

- Example
    ```python
    model = Net()
    optimizers = get_optimizers()
    dataloaders = get_dataloaders()
    model, optimizers, dataloaders = self.setup(model, optimizers, dataloaders)
    ```

[yolox/core/trainer.py#L161](https://github.com/y199387/YOLOX/blob/main/yolox/core/trainer.py#L161)

</details>

<details>
<summary>7. Replace `model.` by `model._module.module` after setup model in multi-instance training(optional)</summary>

- Example
    ```python
    if self.is_distributed:
        # model.freeze = False
        model._module.module.freeze = False
    ```

[yolox/utils/ema.py#19](https://github.com/y199387/YOLOX/blob/main/yolox/utils/ema.py#L19)  
[yolox/core/trainer.py#202](https://github.com/y199387/YOLOX/blob/main/yolox/core/trainer.py#L202)
</details>

<details>
<summary>8. Replace `loss.backward()` by `self.backward(loss)`.</summary>

- Example
    ```python
    # loss.backward()
    self.backward(loss)
    ```

[yolox/core/trainer.py#L113](https://github.com/y199387/YOLOX/blob/main/yolox/core/trainer.py#L113)
</details>

<details>
<summary>9. Instantiate your `TorchNano` subclass and its `train()` method.</summary>

- Example
    ```python
    MyNano().train()
    ```

[tools/train.py#127](https://github.com/y199387/YOLOX/blob/main/tools/train.py#L127)

</details>
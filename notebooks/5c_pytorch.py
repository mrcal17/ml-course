import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell
def _(mo):
    mo.md(r"""
# Module 5C: Deep Learning with PyTorch

In Module 2A, you built backpropagation by hand. You defined forward pass equations, derived gradient expressions with the chain rule, and computed gradients in reverse order through a computational graph. That was essential. You cannot use a tool well if you do not understand what it does.

But you also felt the pain. For a two-layer network with a handful of parameters, the manual approach was already tedious. Imagine doing it for a network with 100 million parameters, 50 layers, and operations like batch normalization, attention, and residual connections. The gradient derivations alone would fill a textbook. And implementing them without bugs? Practically impossible.

This is why frameworks exist. PyTorch automates exactly the process you did by hand — automatic differentiation through a dynamically constructed computation graph, with GPU acceleration and modular building blocks. The key insight of this module is that **nothing PyTorch does is magic**. Every API choice maps directly to a concept you already understand. `requires_grad=True` tells the framework to record operations in the graph you drew on paper. `.backward()` walks that graph in reverse, exactly as you did manually. `nn.Module` organizes the parameters you initialized by hand into a hierarchy. The training loop is the same SGD update rule from Module 0F, just written with cleaner abstractions.

Understanding this mapping — from manual to automated — is what separates someone who uses PyTorch from someone who understands it.

> **Reading**: [DLBook Ch 6.5](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the computational graph formalism, [Geron Ch 12](file:///C:/Users/landa/ml-course/textbooks/Geron.pdf) for a practical introduction to TensorFlow/PyTorch (the concepts transfer).
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 1. From NumPy to Tensors — Why Frameworks Exist

You implemented backpropagation manually in Module 2A. That was essential for understanding. But for real models with millions of parameters, you need three things that NumPy cannot provide:

1. **Automatic differentiation.** No hand-derived gradients. Define the forward pass, and the framework computes all gradients for you — correctly, efficiently, and for arbitrarily complex computation graphs.

2. **GPU acceleration.** Matrix multiplications on a GPU are 10-100x faster than on a CPU. Modern deep learning is computationally infeasible without this.

3. **Modular building blocks.** Layers, loss functions, and optimizers that you can compose like Lego bricks. You should not have to re-derive the gradient of batch normalization every time you use it.

PyTorch's core abstraction is the **tensor** — a multi-dimensional array, just like a NumPy ndarray, but with two critical additions: it can live on a GPU, and it can track the operations performed on it for automatic differentiation.

The API is deliberately NumPy-like. If you know NumPy, you already know 80% of PyTorch's tensor interface. This is by design — the PyTorch developers wanted the learning curve to be about deep learning concepts, not about memorizing a new array API.
""")
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return torch, nn, optim, DataLoader, TensorDataset


@app.cell
def _(torch, np):
    def _run():
        # Creating tensors — the syntax mirrors NumPy almost exactly
        # From a Python list
        t1 = torch.tensor([1.0, 2.0, 3.0])
        print("From list:     ", t1, f"  dtype={t1.dtype}")

        # Random tensors (like np.random.randn)
        t2 = torch.randn(3, 4)
        print(f"Random (3x4):  shape={t2.shape}")

        # Zeros and ones (like np.zeros, np.ones)
        t3 = torch.zeros(2, 3)
        t4 = torch.ones(2, 3)

        # From NumPy — shares memory (no copy!)
        np_arr = np.array([1.0, 2.0, 3.0])
        t5 = torch.from_numpy(np_arr)
        np_arr[0] = 999.0  # modifying NumPy array changes the tensor
        print(f"Shared memory:  np_arr[0]={np_arr[0]}, tensor[0]={t5[0].item()}")

        # Operations: broadcasting, matrix multiply, element-wise — same as NumPy
        A = torch.randn(3, 4)
        B = torch.randn(4, 2)
        C = A @ B  # matrix multiply
        print(f"Matrix multiply: ({A.shape}) @ ({B.shape}) = {C.shape}")

        # Element-wise operations
        x = torch.tensor([1.0, 4.0, 9.0])
        print(f"sqrt({x.tolist()}) = {torch.sqrt(x).tolist()}")
        print(f"exp({x.tolist()})  = {torch.exp(x).tolist()}")

    _run()
    return


@app.cell
def _(torch):
    def _run():
        # Device management: moving tensors to GPU and back
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Create on CPU, then move
        t_cpu = torch.randn(1000, 1000)
        t_device = t_cpu.to(device)
        print(f"CPU tensor device:    {t_cpu.device}")
        print(f"Moved tensor device:  {t_device.device}")

        # Timing comparison: matrix multiply CPU vs GPU
        import time

        size = 2000
        a = torch.randn(size, size)
        b = torch.randn(size, size)

        start = time.perf_counter()
        _ = a @ b
        cpu_time = time.perf_counter() - start
        print(f"\nCPU matmul ({size}x{size}): {cpu_time*1000:.1f} ms")

        if torch.cuda.is_available():
            a_gpu = a.to("cuda")
            b_gpu = b.to("cuda")
            # Warmup
            _ = a_gpu @ b_gpu
            torch.cuda.synchronize()

            start = time.perf_counter()
            _ = a_gpu @ b_gpu
            torch.cuda.synchronize()
            gpu_time = time.perf_counter() - start
            print(f"GPU matmul ({size}x{size}): {gpu_time*1000:.1f} ms")
            print(f"Speedup: {cpu_time/gpu_time:.1f}x")
        else:
            print("(No GPU available — speedup demo skipped)")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 2. Autograd — The Computation Graph You Built By Hand

This is the most important section in this module. Everything else — `nn.Module`, optimizers, data loaders — is convenience built on top of autograd. If you understand autograd, you understand PyTorch.

Recall what you did in Module 2A. You had a computation like:

$$z = wx + b, \quad \hat{y} = z, \quad L = (\hat{y} - t)^2$$

You drew the computation graph. You derived $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$ using the chain rule. You walked the graph in reverse, computing each gradient from the ones downstream.

PyTorch does **exactly** this, automatically:

1. **`requires_grad=True`** tells PyTorch to track every operation on this tensor. It is recording the computation graph as you go.

2. **Every operation** (addition, multiplication, matrix multiply, activation functions) creates a new node in the graph. The node stores what operation was performed and what the inputs were — enough information to compute the local gradient later.

3. **`.backward()`** walks the graph in reverse (just like you did), applying the chain rule at each node, and fills the `.grad` attribute of every leaf tensor (the ones with `requires_grad=True`).

The beauty of this design is that you never write a backward pass. You define the forward computation — which is just regular Python code — and PyTorch derives the backward pass for you. This is called **reverse-mode automatic differentiation**, and it is the same algorithm you implemented by hand in Module 2A.

> **Reading**: [DLBook §6.5.1-6.5.6](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the formal treatment of computation graphs and backpropagation as reverse-mode AD.
""")
    return


@app.cell
def _(torch):
    def _run():
        # The exact computation from Module 2A, now with autograd
        # y = w*x + b, loss = (y - target)^2

        x = torch.tensor(2.0)
        target = torch.tensor(7.0)

        # These are the learnable parameters — requires_grad=True
        w = torch.tensor(3.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)

        # Forward pass — PyTorch builds the computation graph as we go
        y = w * x + b          # y = 3*2 + 1 = 7
        loss = (y - target)**2  # loss = (7 - 7)^2 = 0

        print(f"y = {y.item():.4f}")
        print(f"loss = {loss.item():.4f}")

        # Backward pass — PyTorch walks the graph in reverse
        loss.backward()

        print(f"\ndL/dw (PyTorch): {w.grad.item():.4f}")
        print(f"dL/db (PyTorch): {b.grad.item():.4f}")

        # VERIFY BY HAND:
        # loss = (wx + b - t)^2
        # dL/dy = 2(y - t) = 2(7 - 7) = 0
        # dy/dw = x = 2
        # dy/db = 1
        # dL/dw = dL/dy * dy/dw = 0 * 2 = 0
        # dL/db = dL/dy * dy/db = 0 * 1 = 0
        print(f"\ndL/dw (by hand): {2 * (y.item() - target.item()) * x.item():.4f}")
        print(f"dL/db (by hand): {2 * (y.item() - target.item()) * 1:.4f}")

    _run()
    return


@app.cell
def _(torch):
    def _run():
        # A more interesting example where loss != 0
        x = torch.tensor(2.0)
        target = torch.tensor(7.0)

        w = torch.tensor(1.5, requires_grad=True)
        b = torch.tensor(0.5, requires_grad=True)

        y = w * x + b          # y = 1.5*2 + 0.5 = 3.5
        loss = (y - target)**2  # loss = (3.5 - 7)^2 = 12.25

        loss.backward()

        print(f"y = {y.item():.4f}, loss = {loss.item():.4f}")
        print(f"dL/dw (PyTorch): {w.grad.item():.4f}")
        print(f"dL/db (PyTorch): {b.grad.item():.4f}")

        # By hand:
        # dL/dy = 2(3.5 - 7) = -7
        # dL/dw = -7 * 2 = -14
        # dL/db = -7 * 1 = -7
        dLdy = 2 * (3.5 - 7.0)
        print(f"\nVerification:")
        print(f"  dL/dy = 2(y-t) = {dLdy:.4f}")
        print(f"  dL/dw = dL/dy * x = {dLdy * 2.0:.4f}")
        print(f"  dL/db = dL/dy * 1 = {dLdy * 1.0:.4f}")
        print("  Matches!" if w.grad.item() == dLdy * 2.0 else "  Mismatch!")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Key Gotchas — and WHY They Exist

There are three autograd behaviors that trip up every beginner. Each exists for a good reason.

**1. `torch.no_grad()` — Skip graph construction.**
During inference (making predictions on new data), you do not need gradients. Building the computation graph costs memory — every intermediate tensor is kept alive so backprop can use it later. Wrapping inference code in `with torch.no_grad():` tells PyTorch not to build the graph, saving significant memory and computation. This is not optional for production models — without it, you will run out of GPU memory.

**2. `.detach()` — Remove a tensor from the graph.**
Sometimes you want to use a tensor's value but stop gradient from flowing through it. For example, in reinforcement learning, you compute a target value that should be treated as a constant, not differentiated through. `.detach()` creates a new tensor with the same data but no connection to the computation graph. The gradient stops there.

**3. Gradient accumulation — PyTorch ADDS gradients by default.**
This is the one that causes the most bugs. When you call `.backward()`, PyTorch does not replace `.grad` — it adds to it. Why? Because **gradient accumulation is actually useful**: if your GPU cannot fit a large batch, you can process several small batches and accumulate their gradients before taking a step. The accumulated gradient approximates the gradient of the larger batch. But during standard training, where each batch is independent, you must call `optimizer.zero_grad()` (or manually zero the `.grad` tensors) before each backward pass. Forgetting this is one of the most common PyTorch bugs.
""")
    return


@app.cell
def _(torch):
    def _run():
        # DEMO: What happens when you forget zero_grad
        # Gradients ACCUMULATE — they grow each iteration
        w = torch.tensor(2.0, requires_grad=True)
        x = torch.tensor(3.0)
        target = torch.tensor(1.0)

        print("Without zero_grad — gradients accumulate:")
        for i in range(5):
            y = w * x
            loss = (y - target)**2
            loss.backward()
            print(f"  Iter {i}: loss={loss.item():.2f}, "
                  f"w.grad={w.grad.item():.2f} (accumulated!)")

        print(f"\nFinal w.grad = {w.grad.item():.2f}")
        print("This is 5x what it should be!")

        # Now with proper zeroing
        w2 = torch.tensor(2.0, requires_grad=True)
        print("\nWith zero_grad — correct behavior:")
        for i in range(5):
            if w2.grad is not None:
                w2.grad.zero_()  # <-- THIS IS CRITICAL
            y2 = w2 * x
            loss2 = (y2 - target)**2
            loss2.backward()
            print(f"  Iter {i}: loss={loss2.item():.2f}, "
                  f"w.grad={w2.grad.item():.2f} (correct)")

    _run()
    return


@app.cell
def _(torch):
    def _run():
        # torch.no_grad() demo — saves memory, disables graph
        w = torch.tensor(2.0, requires_grad=True)
        x = torch.tensor(3.0)

        # Normal mode: graph is built
        y1 = w * x
        print(f"With graph:    y.requires_grad = {y1.requires_grad}")

        # no_grad mode: no graph, saves memory
        with torch.no_grad():
            y2 = w * x
            print(f"Without graph: y.requires_grad = {y2.requires_grad}")

        # .detach() demo
        y3 = w * x
        y3_detached = y3.detach()
        print(f"\nOriginal:  requires_grad = {y3.requires_grad}")
        print(f"Detached:  requires_grad = {y3_detached.requires_grad}")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 3. nn.Module — The Building Block

You could build neural networks using raw tensors and autograd alone. Define weight tensors with `requires_grad=True`, write the forward pass as tensor operations, call `.backward()`, and update weights manually. This works — you will do exactly this in Exercise 1.

But for anything beyond a toy example, you need organization. A ResNet has 50+ layers, each with weights and biases. You need to:
- **Find all learnable parameters** automatically (for the optimizer)
- **Switch modes** between training and evaluation (dropout and batch norm behave differently)
- **Save and load** model weights
- **Compose** smaller modules into larger ones

This is what `nn.Module` provides. It is a base class that gives you all four capabilities. Every PyTorch model — from a single linear layer to GPT-4 — inherits from `nn.Module`.

The contract is simple: you define two things.

1. **`__init__`**: Create your layers (sub-modules) and register them as attributes. `nn.Module` automatically discovers all parameters in all sub-modules.

2. **`forward`**: Define the computation. Take an input tensor, pass it through your layers, return the output. This is called when you do `model(x)` — never call `model.forward(x)` directly.

Why a class hierarchy instead of just functions? Because the class **owns state**. The weights live inside the module. When you call `model.parameters()`, it traverses the entire tree of sub-modules and collects every parameter. When you call `model.eval()`, it switches every sub-module to evaluation mode. When you call `model.state_dict()`, it serializes every parameter into a dictionary you can save to disk. Functions cannot do any of this.
""")
    return


@app.cell
def _(nn, torch):
    class TwoLayerNet(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.layer1 = nn.Linear(in_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(hidden_dim, out_dim)

        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            return x

    # Instantiate and inspect
    model_demo = TwoLayerNet(in_dim=784, hidden_dim=128, out_dim=10)

    # Module.parameters() finds all learnable weights automatically
    total_params = sum(p.numel() for p in model_demo.parameters())
    print(f"Architecture: 784 -> 128 -> 10")
    print(f"Total parameters: {total_params:,}")
    print()

    # named_parameters shows exactly what the module contains
    for name, param in model_demo.named_parameters():
        print(f"  {name:20s}  shape={str(list(param.shape)):15s}  "
              f"params={param.numel():,}")

    # Quick forward pass test
    x_test = torch.randn(4, 784)  # batch of 4 images
    out_test = model_demo(x_test)
    print(f"\nInput shape:  {list(x_test.shape)}")
    print(f"Output shape: {list(out_test.shape)}")
    return TwoLayerNet, model_demo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### nn.Sequential: When Forward Is Just a Chain

If your forward pass is simply "apply layer 1, then layer 2, then layer 3" with no branching or skipping, you can use `nn.Sequential` instead of writing a class. It chains modules in order:
""")
    return


@app.cell
def _(nn, torch):
    def _run():
        # nn.Sequential — skip the class when forward is just a chain
        model_seq = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        # Same architecture as TwoLayerNet, same parameter count
        total = sum(p.numel() for p in model_seq.parameters())
        print(f"Sequential model parameters: {total:,}")

        # Forward pass: just call the model
        x = torch.randn(4, 784)
        out = model_seq(x)
        print(f"Output shape: {list(out.shape)}")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 4. The Training Loop

PyTorch does not hide the training loop behind a `.fit()` method. This is a deliberate design choice. The explicit loop gives you full control — you can add gradient clipping, learning rate scheduling, mixed precision training, logging, or any custom logic. The cost is that you must write it yourself. The benefit is that nothing is hidden.

The core is five lines:

```python
logits = model(x_batch)           # 1. Forward pass
loss = loss_fn(logits, y_batch)   # 2. Compute loss
loss.backward()                   # 3. Backward pass (compute gradients)
optimizer.step()                  # 4. Update parameters
optimizer.zero_grad()             # 5. Reset gradients for next iteration
```

Every training loop in PyTorch — from a simple classifier to a large language model — is a variation on these five lines. Let us examine each component.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Loss Functions — The Subtleties

**`nn.CrossEntropyLoss`** is PyTorch's standard loss for classification. It expects **raw logits** — the unscaled output of your network's final linear layer. Internally, it applies `log_softmax` and then `NLLLoss` (negative log-likelihood). This is not just a convenience — it is numerically critical.

Why? Because computing `softmax` followed by `log` is numerically unstable. The softmax involves `exp(z_i)`, which can overflow for large logits. The log of very small softmax outputs can underflow. By combining the two operations into `log_softmax`, PyTorch uses the log-sum-exp trick to avoid both problems. If you compute `softmax` yourself and pass the result to `CrossEntropyLoss`, you are defeating this stability mechanism. This is one of the most common bugs in PyTorch code.

**`nn.MSELoss`** is for regression. It computes mean squared error: $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$.

### Optimizers

**SGD**: The same stochastic gradient descent from Module 0F. Optionally with momentum (Module 2B).

**Adam**: Adaptive per-parameter learning rates using first and second moment estimates of the gradient (Module 2B). The default choice when you do not know what to use.

**AdamW**: Adam with decoupled weight decay (Module 2C). Proper L2 regularization that is not distorted by the adaptive learning rate. Preferred over Adam when using weight decay.

### train() vs eval() — What Actually Changes

Calling `model.train()` and `model.eval()` switches two things:

1. **Dropout**: In train mode, randomly zeros neurons (regularization). In eval mode, uses all neurons but scales outputs. If you forget `model.eval()` at test time, dropout is still randomly zeroing neurons, and your test accuracy will appear lower than it actually is.

2. **Batch normalization**: In train mode, normalizes using the current batch's mean and variance. In eval mode, uses running averages accumulated during training. If you forget `model.eval()`, batch norm uses the test batch statistics, which may differ significantly from the training distribution.

Forgetting `model.eval()` before evaluation is a classic bug. Your test accuracy drops for no obvious reason. The model is fine — you just forgot to tell it to stop being stochastic.
""")
    return


@app.cell
def _(torch, nn, optim, np, plt):
    def _run():
        # Full training loop on MNIST
        from torchvision import datasets, transforms

        # Load MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        train_dataset = datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000,
                                                  shuffle=False)

        # Model: simple feedforward network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ).to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Training loop
        n_epochs = 5
        train_losses = []
        test_losses = []
        test_accs = []

        for epoch in range(n_epochs):
            # ---- Train ----
            model.train()  # enable dropout, batch norm in train mode
            epoch_loss = 0.0
            n_batches = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                logits = model(x_batch)           # 1. Forward pass
                loss = loss_fn(logits, y_batch)   # 2. Compute loss
                loss.backward()                   # 3. Backward pass
                optimizer.step()                  # 4. Update params
                optimizer.zero_grad()             # 5. Zero gradients

                epoch_loss += loss.item()
                n_batches += 1

            train_losses.append(epoch_loss / n_batches)

            # ---- Evaluate ----
            model.eval()  # disable dropout, use running batch norm stats
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():  # no graph needed for evaluation
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    logits = model(x_batch)
                    test_loss += loss_fn(logits, y_batch).item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)

            test_losses.append(test_loss / len(test_loader))
            test_accs.append(correct / total)

            print(f"Epoch {epoch+1}/{n_epochs}  "
                  f"train_loss={train_losses[-1]:.4f}  "
                  f"test_loss={test_losses[-1]:.4f}  "
                  f"test_acc={test_accs[-1]:.4f}")

        # Plot training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(range(1, n_epochs+1), train_losses, 'b-o', label='Train loss')
        ax1.plot(range(1, n_epochs+1), test_losses, 'r-o', label='Test loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss curves')
        ax1.legend()

        ax2.plot(range(1, n_epochs+1), test_accs, 'g-o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Test accuracy')
        ax2.set_ylim(0.9, 1.0)

        plt.tight_layout()
        fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
The training loop above achieves roughly 97-98% accuracy on MNIST with a simple feedforward network in 5 epochs. Every line maps to a concept you already know: the forward pass computes the prediction, the loss measures how wrong it is, `backward()` computes the gradients you derived by hand in 2A, `step()` applies the update rule from 0F, and `zero_grad()` resets for the next batch.

Notice how `model.eval()` and `torch.no_grad()` work together during evaluation. `model.eval()` changes the behavior of dropout and batch norm. `torch.no_grad()` avoids building the computation graph, saving memory. Both are necessary.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 5. DataLoaders — Feeding the Network

In the training loop above, we used `DataLoader` without much explanation. Let us now understand why it exists and what it does.

The two core abstractions are:

**`Dataset`**: Wraps your data and provides two methods: `__getitem__(index)` returns one sample, and `__len__()` returns the total count. For standard benchmarks like MNIST, `torchvision.datasets` provides ready-made Dataset classes. For your own data, you subclass `Dataset` and implement these two methods.

**`DataLoader`**: Takes a Dataset and handles batching, shuffling, and parallel data loading. This is where the engineering meets the math:

- **Batching**: Instead of computing the gradient on one sample (high variance) or the entire dataset (too slow), we compute it on a mini-batch. This is the SGD from Module 0F — the batch size controls the variance of the gradient estimate.

- **Shuffling**: If the data is ordered (e.g., all 0s first, then all 1s), the model learns the order, not the content. Shuffling each epoch ensures the model sees different batch compositions, which acts as a form of regularization.

- **Transforms**: A pipeline of preprocessing operations applied to each sample. `transforms.ToTensor()` converts PIL images to tensors. `transforms.Normalize()` standardizes pixel values. You can add data augmentation (random flips, crops, rotations) here — this is a form of regularization from Module 2C.
""")
    return


@app.cell
def _(torch):
    def _run():
        from torchvision import datasets, transforms

        # The transform pipeline
        transform = transforms.Compose([
            transforms.ToTensor(),       # PIL Image -> tensor, scales to [0, 1]
            transforms.Normalize((0.1307,), (0.3081,))  # standardize
        ])

        # Dataset: provides __getitem__ and __len__
        dataset = datasets.MNIST(root='./data', train=True,
                                 download=True, transform=transform)
        print(f"Dataset size: {len(dataset)}")

        # Look at one sample
        image, label = dataset[0]
        print(f"Sample shape: {list(image.shape)}  (channels, height, width)")
        print(f"Sample label: {label}")
        print(f"Pixel range: [{image.min():.2f}, {image.max():.2f}] (after normalization)")

        # DataLoader: batching, shuffling, parallel loading
        loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                             shuffle=True)

        # Iterate one batch
        x_batch, y_batch = next(iter(loader))
        print(f"\nBatch shape:  {list(x_batch.shape)}  (batch, channels, H, W)")
        print(f"Labels shape: {list(y_batch.shape)}")
        print(f"Labels: {y_batch.tolist()}")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 6. Building a CNN

In Module 2D, you learned the theory behind convolutional neural networks: locality, parameter sharing, translation equivariance, hierarchical feature extraction. You implemented convolution from scratch with nested loops. Now we use PyTorch's `nn.Conv2d` to build real CNNs.

The key layers:

- **`nn.Conv2d(in_channels, out_channels, kernel_size)`**: A 2D convolution layer. `in_channels` is the depth of the input (1 for grayscale, 3 for RGB). `out_channels` is the number of filters (each learns a different pattern). `kernel_size` is the spatial size of the filter.

- **`nn.MaxPool2d(kernel_size)`**: Downsamples by taking the maximum in each spatial window. Reduces spatial dimensions and provides some translation invariance.

- **`nn.BatchNorm2d(num_features)`**: Normalizes activations across the batch. Stabilizes training and allows higher learning rates (Module 2C).

We will build a CNN for Fashion-MNIST — a harder drop-in replacement for MNIST where the classes are clothing items (T-shirt, trouser, pullover, etc.) instead of digits. Same image size (28x28 grayscale), same 10 classes, but the patterns are more complex and the feedforward baseline does worse. This makes the comparison between feedforward and CNN more informative.
""")
    return


@app.cell
def _(nn, torch):
    class FeedforwardNet(nn.Module):
        """Feedforward baseline for comparison."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.net(x)

    class ConvNet(nn.Module):
        """CNN with 2 conv blocks + 2 FC layers."""
        def __init__(self):
            super().__init__()
            # Conv block 1: 1 -> 32 channels
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 28x28 -> 14x14
            )
            # Conv block 2: 32 -> 64 channels
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 14x14 -> 7x7
            )
            # FC layers
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.fc(x)
            return x

    # Compare parameter counts
    ff = FeedforwardNet()
    cnn = ConvNet()
    ff_params = sum(p.numel() for p in ff.parameters())
    cnn_params = sum(p.numel() for p in cnn.parameters())
    print(f"Feedforward parameters: {ff_params:>10,}")
    print(f"CNN parameters:         {cnn_params:>10,}")
    print(f"CNN uses {ff_params / cnn_params:.1f}x fewer parameters")
    return FeedforwardNet, ConvNet


@app.cell
def _(mo):
    lr_slider = mo.ui.slider(start=-4.0, stop=-2.0, step=0.25, value=-3.0,
                             label="log10(learning rate)")
    epochs_slider = mo.ui.slider(start=2, stop=10, step=1, value=5,
                                 label="Training epochs")
    mo.vstack([lr_slider, epochs_slider])
    return lr_slider, epochs_slider


@app.cell
def _(FeedforwardNet, ConvNet, nn, optim, torch, np, plt,
      lr_slider, epochs_slider):
    def _run():
        from torchvision import datasets, transforms

        # Load Fashion-MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST stats
        ])
        train_data = datasets.FashionMNIST(root='./data', train=True,
                                           download=True, transform=transform)
        test_data = datasets.FashionMNIST(root='./data', train=False,
                                          download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000,
                                                  shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lr = 10 ** lr_slider.value
        n_epochs = epochs_slider.value

        def train_and_evaluate(model, name):
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()

            train_losses = []
            test_accs = []

            for epoch in range(n_epochs):
                model.train()
                running_loss = 0.0
                n_batches = 0
                for x_b, y_b in train_loader:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    logits = model(x_b)
                    loss = loss_fn(logits, y_b)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    running_loss += loss.item()
                    n_batches += 1
                train_losses.append(running_loss / n_batches)

                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for x_b, y_b in test_loader:
                        x_b, y_b = x_b.to(device), y_b.to(device)
                        preds = model(x_b).argmax(dim=1)
                        correct += (preds == y_b).sum().item()
                        total += y_b.size(0)
                test_accs.append(correct / total)

                print(f"  [{name}] Epoch {epoch+1}: "
                      f"train_loss={train_losses[-1]:.4f}, "
                      f"test_acc={test_accs[-1]:.4f}")

            return train_losses, test_accs

        print(f"Training with lr={lr:.1e}, epochs={n_epochs}")
        print()
        ff_losses, ff_accs = train_and_evaluate(FeedforwardNet(), "Feedforward")
        print()
        cnn_losses, cnn_accs = train_and_evaluate(ConvNet(), "CNN")

        # Plot comparison
        epochs = range(1, n_epochs + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(epochs, ff_losses, 'b-o', label='Feedforward')
        ax1.plot(epochs, cnn_losses, 'r-o', label='CNN')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Loss')
        ax1.set_title('Training Loss')
        ax1.legend()

        ax2.plot(epochs, ff_accs, 'b-o', label='Feedforward')
        ax2.plot(epochs, cnn_accs, 'r-o', label='CNN')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title('Test Accuracy')
        ax2.legend()

        plt.tight_layout()
        fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
The CNN should achieve roughly 89-91% test accuracy on Fashion-MNIST, compared to 85-88% for the feedforward network — and it does so with **fewer parameters**. This is the power of inductive bias. By building in the assumptions of locality and translation invariance (Module 2D), the CNN extracts spatial patterns far more efficiently. The feedforward network has to learn from scratch that pixel (14, 14) is near pixel (14, 15); the CNN knows this by construction.

Use the sliders above to experiment with the learning rate and number of epochs. Too high a learning rate causes instability; too low and the model barely trains in the allotted epochs. This is the same learning rate sensitivity you explored in Module 2B.
""")
    return


@app.cell
def _(ConvNet, torch, np, plt):
    def _run():
        # Visualize the learned first-layer filters
        model_viz = ConvNet()

        # Get first conv layer weights: shape (32, 1, 3, 3)
        filters = model_viz.conv1[0].weight.detach().cpu().numpy()
        n_filters = min(filters.shape[0], 16)  # show up to 16

        fig, axes = plt.subplots(2, 8, figsize=(10, 3))
        for i, ax in enumerate(axes.flat):
            if i < n_filters:
                ax.imshow(filters[i, 0], cmap='gray')
            ax.axis('off')
        fig.suptitle('First-layer conv filters (random init)', fontsize=12)
        plt.tight_layout()
        fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
The filters above are from a randomly initialized network. After training, these filters would show meaningful patterns — edge detectors at various orientations, corner detectors, and gradient patterns. This is exactly what Module 2D predicted: the first convolutional layer learns low-level features, and deeper layers compose them into higher-level representations.

> **Reading**: [DLBook Ch 9](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the full treatment of CNNs, [Geron Ch 14](file:///C:/Users/landa/ml-course/textbooks/Geron.pdf) for practical CNN architectures in PyTorch.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 7. Interactive Exploration

Use the controls below to build a custom network architecture and see how hyperparameter choices affect training dynamics. This connects directly to the theory from Module 2B (optimizer behavior) and Module 2C (regularization effects).
""")
    return


@app.cell
def _(mo):
    arch_dropdown = mo.ui.dropdown(
        options=["1-hidden (128)", "2-hidden (256-128)", "3-hidden (512-256-128)"],
        value="2-hidden (256-128)",
        label="Architecture"
    )
    opt_dropdown = mo.ui.dropdown(
        options=["SGD", "SGD+Momentum", "Adam", "AdamW"],
        value="Adam",
        label="Optimizer"
    )
    dropout_slider = mo.ui.slider(start=0.0, stop=0.5, step=0.05, value=0.0,
                                  label="Dropout rate")
    mo.vstack([arch_dropdown, opt_dropdown, dropout_slider])
    return arch_dropdown, opt_dropdown, dropout_slider


@app.cell
def _(nn, optim, torch, plt, arch_dropdown, opt_dropdown, dropout_slider):
    def _run():
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST(root='./data', train=True,
                                    download=True, transform=transform)
        test_data = datasets.MNIST(root='./data', train=False,
                                   download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000,
                                                  shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        drop_rate = dropout_slider.value

        # Build architecture based on dropdown
        arch = arch_dropdown.value
        if arch == "1-hidden (128)":
            layers = [nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
                      nn.Dropout(drop_rate), nn.Linear(128, 10)]
        elif arch == "2-hidden (256-128)":
            layers = [nn.Flatten(),
                      nn.Linear(784, 256), nn.ReLU(), nn.Dropout(drop_rate),
                      nn.Linear(256, 128), nn.ReLU(), nn.Dropout(drop_rate),
                      nn.Linear(128, 10)]
        else:  # 3-hidden
            layers = [nn.Flatten(),
                      nn.Linear(784, 512), nn.ReLU(), nn.Dropout(drop_rate),
                      nn.Linear(512, 256), nn.ReLU(), nn.Dropout(drop_rate),
                      nn.Linear(256, 128), nn.ReLU(), nn.Dropout(drop_rate),
                      nn.Linear(128, 10)]

        model = nn.Sequential(*layers).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        # Select optimizer
        opt_name = opt_dropdown.value
        if opt_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        elif opt_name == "SGD+Momentum":
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        elif opt_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
        else:  # AdamW
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        loss_fn = nn.CrossEntropyLoss()
        n_epochs = 5
        train_losses, test_losses, test_accs = [], [], []

        for epoch in range(n_epochs):
            model.train()
            running = 0.0
            n_b = 0
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                loss = loss_fn(model(x_b), y_b)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running += loss.item()
                n_b += 1
            train_losses.append(running / n_b)

            model.eval()
            t_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for x_b, y_b in test_loader:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    logits = model(x_b)
                    t_loss += loss_fn(logits, y_b).item()
                    correct += (logits.argmax(1) == y_b).sum().item()
                    total += y_b.size(0)
            test_losses.append(t_loss / len(test_loader))
            test_accs.append(correct / total)

        print(f"Architecture: {arch} | Optimizer: {opt_name} | "
              f"Dropout: {drop_rate} | Params: {n_params:,}")
        print(f"Final test accuracy: {test_accs[-1]:.4f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        epochs = range(1, n_epochs + 1)
        ax1.plot(epochs, train_losses, 'b-o', label='Train')
        ax1.plot(epochs, test_losses, 'r-o', label='Test')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.set_title(f'{opt_name} | dropout={drop_rate}')
        ax1.legend()

        ax2.plot(epochs, test_accs, 'g-o')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Test Accuracy ({n_params:,} params)')
        ax2.set_ylim(0.9, 1.0)

        plt.tight_layout()
        fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
Observations to look for as you adjust the controls:

- **SGD vs Adam**: SGD with no momentum is noticeably slower to converge. Adding momentum helps substantially. Adam converges fastest because of its adaptive learning rates — but the final accuracy is often similar given enough epochs. This confirms the theory from Module 2B: adaptive methods are faster to converge but do not necessarily find better minima.

- **Dropout**: Adding dropout hurts training loss (the network is being randomly crippled during training) but can help test accuracy, especially for larger architectures that are more prone to overfitting. This is the regularization tradeoff from Module 2C.

- **Architecture depth**: More layers means more parameters and more representational capacity. On MNIST (a simple problem), adding depth past 2 hidden layers provides diminishing returns.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 8. Code It

These exercises build on everything above. They progress from raw autograd (no framework abstractions) to full model training, mirroring the progression from Module 2A (manual backprop) to production-grade deep learning.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Exercise 1: Linear Regression with Raw Autograd

Implement linear regression using **only tensors and autograd** — no `nn.Module`, no `nn.Linear`, no optimizer. Create weight and bias tensors with `requires_grad=True`, write the forward pass and MSE loss as tensor operations, call `.backward()`, and update the parameters manually with `w -= lr * w.grad`.

This exercise connects Module 2A (manual backprop) to PyTorch: you are doing the same thing, but letting autograd handle the gradient computation.
""")
    return


@app.cell
def _(torch):
    def _run():
        # Generate synthetic data: y = 3x + 2 + noise
        torch.manual_seed(42)
        x_train = torch.randn(100, 1)
        y_train = 3.0 * x_train + 2.0 + 0.3 * torch.randn(100, 1)

        # TODO: Initialize parameters with requires_grad=True
        # w = torch.randn(1, requires_grad=True)
        # b = torch.zeros(1, requires_grad=True)

        # TODO: Training loop (200 iterations)
        # lr = 0.1
        # for i in range(200):
        #     y_pred = ...               # forward pass: w*x + b
        #     loss = ...                 # MSE loss: mean((y_pred - y_train)^2)
        #     loss.backward()            # compute gradients
        #     with torch.no_grad():      # update without tracking
        #         w -= lr * w.grad
        #         b -= lr * b.grad
        #     w.grad.zero_()             # zero gradients
        #     b.grad.zero_()

        # print(f"Learned: w={w.item():.3f} (true: 3.0), b={b.item():.3f} (true: 2.0)")

        print("Exercise 1: Implement linear regression with raw autograd.")
        print("No nn.Module, no optimizer — just tensors and .backward().")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Exercise 2: Three-Layer Feedforward Network with nn.Module

Build a 3-layer feedforward network (784 -> 512 -> 256 -> 10) using `nn.Module`. Train it on MNIST for 5 epochs. Track and plot training loss and test accuracy.

This is the standard workflow: define a Module class, instantiate it, write the training loop.
""")
    return


@app.cell
def _(nn, torch):
    def _run():
        # TODO: Define the network class
        # class ThreeLayerNet(nn.Module):
        #     def __init__(self):
        #         super().__init__()
        #         self.net = nn.Sequential(
        #             nn.Flatten(),
        #             nn.Linear(784, 512),
        #             nn.ReLU(),
        #             nn.Linear(512, 256),
        #             nn.ReLU(),
        #             nn.Linear(256, 10),
        #         )
        #     def forward(self, x):
        #         return self.net(x)

        # TODO: Load MNIST, create DataLoaders
        # TODO: Training loop for 5 epochs
        # TODO: Print final test accuracy
        # TODO: Plot training loss and test accuracy curves

        print("Exercise 2: Build a 3-layer MLP, train on MNIST.")
        print("Target: >97% test accuracy in 5 epochs.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Exercise 3: Add Dropout and Batch Normalization

Take your network from Exercise 2 and create two new versions:
1. Add `nn.Dropout(0.3)` after each ReLU
2. Add `nn.BatchNorm1d(hidden_size)` before each ReLU, plus dropout

Train all three variants (original, +dropout, +dropout+batchnorm) and compare their training curves. Does dropout help test accuracy? Does batch norm speed up convergence?

Remember: you must call `model.eval()` before testing when using dropout or batch norm. If you forget, your test metrics will be wrong.
""")
    return


@app.cell
def _(nn, torch):
    def _run():
        # TODO: Define three model variants
        # model_plain = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(784, 512), nn.ReLU(),
        #     nn.Linear(512, 256), nn.ReLU(),
        #     nn.Linear(256, 10),
        # )
        #
        # model_dropout = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.3),
        #     nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
        #     nn.Linear(256, 10),
        # )
        #
        # model_full = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(784, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
        #     nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        #     nn.Linear(256, 10),
        # )

        # TODO: Train all three, compare curves
        # Key observation: batchnorm should converge faster, dropout should
        # reduce the gap between train and test loss

        print("Exercise 3: Compare plain vs +dropout vs +dropout+batchnorm.")
        print("Key question: what happens if you forget model.eval() before testing?")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Exercise 4: CNN on Fashion-MNIST

Build a CNN for Fashion-MNIST and beat the feedforward baseline from Section 6. Your CNN should have:
- At least 2 convolutional blocks (conv + batchnorm + relu + pool)
- At least 1 fully connected layer before the output
- Fewer parameters than the feedforward baseline

Train for 5-10 epochs. Report the final test accuracy and parameter count. Visualize some predictions (show images with predicted vs true labels).
""")
    return


@app.cell
def _(nn, torch):
    def _run():
        # TODO: Define your CNN
        # Suggested architecture:
        # Conv2d(1, 32, 3, padding=1) -> BN -> ReLU -> MaxPool(2)
        # Conv2d(32, 64, 3, padding=1) -> BN -> ReLU -> MaxPool(2)
        # Flatten -> Linear(64*7*7, 128) -> ReLU -> Dropout -> Linear(128, 10)

        # TODO: Load Fashion-MNIST
        # TODO: Train the CNN
        # TODO: Report accuracy and parameter count
        # TODO: Visualize predictions on test set

        print("Exercise 4: Build a CNN for Fashion-MNIST.")
        print("Target: >89% test accuracy with fewer params than feedforward.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Pencil Exercises

These are not coding exercises — they test your understanding of the concepts.

**P1.** What happens if you forget to call `model.eval()` before testing a model that uses dropout with rate 0.5? Will test accuracy be higher, lower, or the same as correct evaluation? Explain quantitatively.

**P2.** You are training a model and notice that the loss decreases for the first 100 iterations, then suddenly jumps to a very large value and stays there. You check your code and find you forgot `optimizer.zero_grad()`. Explain mechanically what happened: what are the gradients doing, and why does the loss eventually explode?

**P3.** Consider the computation $y = \text{ReLU}(w_2 \cdot \text{ReLU}(w_1 \cdot x))$. Draw the computation graph. If $x = 1$, $w_1 = -0.5$, $w_2 = 2.0$, compute $y$ and then $\frac{\partial y}{\partial w_1}$ by hand. Verify that PyTorch's autograd gives the same answer.

**P4.** You train a model with `nn.CrossEntropyLoss` and accidentally apply `torch.softmax` to your logits before passing them to the loss function. The model trains but achieves lower accuracy than expected. Explain why: what is the loss function actually computing, and why does it lead to worse training?
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

> **Next**: 5D — Transfer Learning & Representation Learning
""")
    return


if __name__ == "__main__":
    app.run()

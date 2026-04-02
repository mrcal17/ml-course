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
    return (np, plt)


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    return (torch, nn, optim, DataLoader, TensorDataset)


@app.cell
def _(mo):
    mo.md(r"""
    # Module 5D: Transfer Learning & Representation Learning

    Here is the single most important practical insight in modern deep learning: **the best models are the ones you do not train from scratch.**

    Think about the trajectory of the field. In classical ML (pre-2012), you designed features by hand --- SIFT descriptors for images, TF-IDF vectors for text, spectral coefficients for audio. An expert spent months engineering the right representation, then plugged it into a simple classifier. In the early deep learning era (2012--2017), you trained a neural network end-to-end from scratch, learning features and classifier simultaneously. This was a revolution, but it required massive datasets and massive compute.

    The current era works differently. You start with a model that someone else has already trained on a huge dataset --- ImageNet, BookCorpus, Common Crawl --- and you *adapt* it to your task. The pretrained model has already learned rich, general-purpose representations. You just need to steer those representations toward your specific problem. This is transfer learning, and it is the default workflow in modern practice.

    This module makes you fluent in that workflow. We will cover feature extraction (frozen backbones), fine-tuning (adaptive backbones), sequence modeling in PyTorch, and autoencoders for unsupervised representation learning. Each of these builds on theory you already have: CNNs from Module 2D, RNNs and LSTMs from Module 2E, and the autoencoder/VAE framework from Module 3B. Now we implement them.

    > **Reading:** [Goodfellow et al., Ch 15: Representation Learning](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) provides the theoretical foundation. [Geron, Ch 11: Training Deep Neural Networks](file:///C:/Users/landa/ml-course/textbooks/Geron.pdf) covers transfer learning from a practical standpoint.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. The Pretrained Revolution

    To understand why transfer learning works, you need to understand what a deep neural network actually learns at each layer. This is not abstract theory --- it has been empirically verified many times since Zeiler and Fergus (2014) first visualized CNN features.

    A CNN trained on ImageNet (1.2 million images, 1000 classes) learns a hierarchy of features:

    - **Layer 1** learns edge detectors and color gradients. These are essentially Gabor filters --- oriented edges at various angles and frequencies. Every image in existence contains edges. These features are universal.
    - **Layer 2** combines edges into textures, corners, and simple patterns. A corner is just two edges meeting at a point. A texture is a repeating pattern of edges. Still universal.
    - **Layer 3** starts assembling textures into parts --- circles, grids, honeycomb patterns, curve fragments. Getting more specific, but still broadly useful.
    - **Layers 4--5** learn object parts and compositions --- dog faces, wheels, window frames. These are domain-specific. A wheel detector is useful for vehicle recognition but irrelevant for medical imaging.

    The critical insight: **the early layers are task-agnostic**. Edges are edges whether you are classifying dogs, detecting tumors, or reading license plates. This is why a network trained on ImageNet transfers well to wildly different tasks --- the bottom layers provide a universal visual vocabulary.

    The economics reinforce this. Training a ResNet-50 from scratch requires roughly 10,000 GPU-hours on ImageNet, demanding millions of labeled images and thousands of dollars in compute. Downloading pretrained weights takes 30 seconds and costs nothing. For most practitioners, training from scratch is not just unnecessary --- it is wasteful.

    **When transfer works.** The source and target domains must share low-level structure. Natural images to natural images (ImageNet to bird species classification) works extremely well. Natural images to satellite imagery works reasonably. Natural images to medical histopathology is borderline --- the textures are quite different, but edges and gradients still transfer.

    **When transfer fails.** If the domains share no structural similarity, the pretrained features are noise. Transferring an ImageNet model to spectrograms or point clouds is unlikely to help without heavy fine-tuning.

    > **Reading:** [Geron, Ch 11](file:///C:/Users/landa/ml-course/textbooks/Geron.pdf) walks through the practical transfer learning workflow. [DLBook, Section 15.2](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) discusses transfer learning from the representation learning perspective.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Feature Hierarchy Visualization

    To make this concrete, consider what happens when you forward-pass an image through a pretrained network and visualize the activations at each layer. The following schematic illustrates the hierarchy --- each layer builds on the previous one, composing simple patterns into complex features.
    """)
    return


@app.cell
def _(np, plt):
    # Visualize the concept: features become more abstract at deeper layers
    # We simulate "activation maps" at different network depths
    rng_vis = np.random.default_rng(42)

    fig_hier, axes_hier = plt.subplots(1, 4, figsize=(12, 3))
    titles = ["Layer 1: Edges", "Layer 2: Textures", "Layer 3: Parts", "Layer 4+: Objects"]

    # Layer 1: edge-like patterns (high-frequency Gabor-like)
    x_g = np.linspace(-2, 2, 32)
    xx, yy = np.meshgrid(x_g, x_g)
    edge_map = np.sin(5 * xx) * np.exp(-(xx**2 + yy**2))
    axes_hier[0].imshow(edge_map, cmap="gray")

    # Layer 2: texture-like (combination of orientations)
    texture_map = np.sin(4 * xx) * np.cos(4 * yy) * np.exp(-0.3 * (xx**2 + yy**2))
    axes_hier[1].imshow(texture_map, cmap="gray")

    # Layer 3: part-like (circular pattern)
    r = np.sqrt(xx**2 + yy**2)
    part_map = np.exp(-2 * (r - 1)**2) - 0.5 * np.exp(-2 * (r - 0.3)**2)
    axes_hier[2].imshow(part_map, cmap="gray")

    # Layer 4: object-like (blob with structure)
    obj_map = (np.exp(-((xx - 0.5)**2 + (yy)**2)) +
               0.7 * np.exp(-((xx + 0.5)**2 + (yy - 0.3)**2) * 2) +
               0.5 * np.exp(-((xx)**2 + (yy + 0.8)**2) * 3))
    axes_hier[3].imshow(obj_map, cmap="gray")

    for ax, title in zip(axes_hier, titles):
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig_hier.suptitle("Conceptual Feature Hierarchy in a Deep CNN", fontsize=12, y=1.02)
    plt.tight_layout()
    fig_hier
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Feature Extraction --- The Frozen Backbone

    The simplest form of transfer learning is **feature extraction**. The strategy: take a pretrained network, freeze every layer, remove the final classification head, and replace it with a new head for your task. Only the new head gets trained. The pretrained layers serve as a fixed feature extractor.

    Why does this work? The pretrained backbone maps raw pixels to a rich feature space. For ResNet-18, the layer before the final classifier produces a 512-dimensional vector that encodes texture, shape, spatial relationships, and object-part information. This 512-dimensional representation is vastly more informative than raw pixels. A simple linear classifier on top of these features can often match or beat a complex model trained on raw pixels.

    The implementation has a few critical details:

    **Freezing parameters.** Set `requires_grad = False` for every parameter in the pretrained model. This prevents gradient computation and weight updates for those parameters, saving both memory and compute.

    **Optimizer scope.** When creating the optimizer, pass only the trainable parameters: `filter(lambda p: p.requires_grad, model.parameters())`. If you pass all parameters, the optimizer creates momentum buffers and state for frozen parameters --- wasting memory and potentially introducing subtle bugs.

    **BatchNorm behavior.** This is the most common mistake in transfer learning. Even when all parameters are frozen, you must call `model.eval()` during both training and inference. Here is why: BatchNorm layers have two modes. In training mode, they compute batch statistics (mean and variance of the current mini-batch) and use those for normalization. In eval mode, they use running statistics accumulated during the original pretraining. If you leave the model in training mode, BatchNorm layers will recompute statistics from your (potentially small, differently distributed) batches, corrupting the pretrained features. This single oversight can drop accuracy by 10+ percentage points.
    """)
    return


@app.cell
def _(nn, torch):
    # Feature extraction setup: freeze a pretrained ResNet, replace the head
    from torchvision.models import resnet18, ResNet18_Weights

    def build_feature_extractor(num_classes):
        """Load pretrained ResNet-18, freeze all layers, replace the classifier head."""
        model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze every parameter in the pretrained backbone
        for param in model.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer (originally 512 -> 1000 for ImageNet)
        in_features = model.fc.in_features  # 512 for ResNet-18
        model.fc = nn.Linear(in_features, num_classes)
        # The new head's parameters have requires_grad=True by default

        return model

    demo_model = build_feature_extractor(num_classes=10)

    # Count parameters: total vs trainable
    total_params = sum(p.numel() for p in demo_model.parameters())
    trainable_params = sum(p.numel() for p in demo_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total parameters:     {total_params:>10,}")
    print(f"Frozen parameters:    {frozen_params:>10,}")
    print(f"Trainable parameters: {trainable_params:>10,}")
    print(f"Trainable fraction:   {trainable_params / total_params:.4%}")
    return (build_feature_extractor, resnet18, ResNet18_Weights)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at those numbers. Out of roughly 11 million parameters, you are training only 5,130 --- the weights and bias of the new linear head. That is 0.05% of the model. The other 99.95% were trained by someone else on ImageNet, and you are reusing their work for free.

    Now let us see this in action. We will use CIFAR-10 as a stand-in for a real transfer learning scenario. The setup: pretend you only have 1,000 labeled training images (100 per class). This simulates a common real-world situation where labeled data is scarce.

    We will compare three strategies:
    1. **From scratch**: train a small CNN on the 1,000 images
    2. **Frozen backbone**: use ResNet-18 features with a linear head
    3. **Fine-tuned** (next section): unfreeze and adapt the full ResNet

    The comparison will demonstrate why transfer learning dominates when data is limited.
    """)
    return


@app.cell
def _(DataLoader, TensorDataset, np, torch):
    from torchvision import datasets, transforms

    # CIFAR-10 with ResNet-compatible preprocessing
    # ResNet expects 224x224 inputs normalized with ImageNet stats
    transform_resnet = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load full CIFAR-10
    cifar_train_full = datasets.CIFAR10(
        root="../data", train=True, download=True, transform=transform_resnet
    )
    cifar_test_full = datasets.CIFAR10(
        root="../data", train=False, download=True, transform=transform_resnet
    )

    # Subsample: only 1000 training images (100 per class) to simulate scarce data
    rng_sub = np.random.default_rng(42)
    targets_array = np.array(cifar_train_full.targets)
    subset_indices = []
    for cls in range(10):
        cls_indices = np.where(targets_array == cls)[0]
        chosen = rng_sub.choice(cls_indices, size=100, replace=False)
        subset_indices.extend(chosen.tolist())
    rng_sub.shuffle(subset_indices)

    cifar_train_subset = torch.utils.data.Subset(cifar_train_full, subset_indices)

    train_loader = DataLoader(cifar_train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(cifar_test_full, batch_size=256, shuffle=False)

    print(f"Training samples: {len(cifar_train_subset)} (subset)")
    print(f"Test samples:     {len(cifar_test_full)}")
    return (
        cifar_train_full, cifar_test_full, cifar_train_subset,
        train_loader, test_loader, transform_resnet, datasets, transforms,
    )


@app.cell
def _(nn, np, torch, optim, train_loader, test_loader, build_feature_extractor):
    def train_and_evaluate(model, train_ldr, test_ldr, epochs, lr, device, label=""):
        """Train a model and return per-epoch train loss and test accuracy."""
        model = model.to(device)
        # Only optimize parameters that require gradients
        params_to_train = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params_to_train, lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        test_accs = []

        for epoch in range(epochs):
            # Training
            model.train()
            # CRITICAL: if using frozen backbone, keep batchnorm in eval mode
            # We handle this by checking if most params are frozen
            frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
            total_count = sum(1 for p in model.parameters())
            if frozen_count > total_count * 0.5:
                model.eval()  # keep BatchNorm using running stats
                # but we still need the head to be "active" for dropout etc.
                # (ResNet head has no dropout, so eval is fine throughout)

            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in train_ldr:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            train_losses.append(epoch_loss / n_batches)

            # Evaluation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in test_ldr:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    preds = model(X_batch).argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)
            test_accs.append(correct / total)

            if (epoch + 1) % 3 == 0 or epoch == 0:
                print(f"  [{label}] Epoch {epoch+1}/{epochs} -- "
                      f"loss: {train_losses[-1]:.4f}, test acc: {test_accs[-1]:.4f}")

        return train_losses, test_accs

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str}")

    # --- Strategy A: Small CNN from scratch ---
    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    print("\n--- Strategy A: Small CNN from scratch ---")
    scratch_model = SmallCNN()
    scratch_losses, scratch_accs = train_and_evaluate(
        scratch_model, train_loader, test_loader, epochs=15, lr=1e-3,
        device=device_str, label="Scratch"
    )

    # --- Strategy B: Frozen ResNet backbone ---
    print("\n--- Strategy B: Frozen ResNet-18 backbone ---")
    frozen_model = build_feature_extractor(num_classes=10)
    frozen_losses, frozen_accs = train_and_evaluate(
        frozen_model, train_loader, test_loader, epochs=15, lr=1e-3,
        device=device_str, label="Frozen"
    )

    print(f"\nFinal test accuracy -- Scratch: {scratch_accs[-1]:.4f}, "
          f"Frozen: {frozen_accs[-1]:.4f}")
    return (
        train_and_evaluate, device_str,
        scratch_losses, scratch_accs,
        frozen_losses, frozen_accs,
        SmallCNN,
    )


@app.cell
def _(plt, scratch_accs, frozen_accs):
    fig_comp, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4))

    epochs_range = range(1, len(scratch_accs) + 1)

    ax_a.plot(epochs_range, scratch_accs, "o-", label="From scratch (small CNN)", markersize=4)
    ax_a.plot(epochs_range, frozen_accs, "s-", label="Frozen ResNet-18", markersize=4)
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Test Accuracy")
    ax_a.set_title("Test Accuracy: Scratch vs Frozen Backbone")
    ax_a.legend()
    ax_a.set_ylim(0, 1)
    ax_a.grid(True, alpha=0.3)

    # Bar chart of final accuracies
    names = ["From Scratch", "Frozen ResNet"]
    finals = [scratch_accs[-1], frozen_accs[-1]]
    colors = ["#d9534f", "#5cb85c"]
    ax_l.bar(names, finals, color=colors, edgecolor="black", linewidth=0.5)
    ax_l.set_ylabel("Final Test Accuracy")
    ax_l.set_title("Final Comparison (1000 training images)")
    ax_l.set_ylim(0, 1)
    for i, v in enumerate(finals):
        ax_l.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig_comp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The gap should be striking. With only 1,000 training images, the frozen ResNet backbone dramatically outperforms the small CNN trained from scratch. The pretrained features --- learned from 1.2 million ImageNet images --- provide a massive head start. The small CNN must learn everything from those 1,000 images: edge detectors, texture detectors, shape representations, AND the classifier. The frozen ResNet already has all of those features baked in. It only needs to learn the final mapping from features to CIFAR-10 classes.

    This is the core value proposition of transfer learning: **amortize the cost of feature learning across tasks**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Fine-Tuning --- When and How

    Frozen feature extraction works well when the source and target domains are similar. But what if they are not? What if your target images have textures, colors, or structures that ImageNet never saw? In that case, the pretrained features are a reasonable starting point but not a perfect fit. You need to **adapt** them.

    Fine-tuning means unfreezing some or all of the pretrained layers and training them alongside the new head, using a small learning rate. The key word is *small*. If you use the same learning rate you would use for training from scratch (say, 1e-2), you will destroy the pretrained features in a few gradient steps. The delicate structure that took thousands of GPU-hours to learn will be overwritten by noisy gradients from your small dataset. This is sometimes called **catastrophic forgetting**.

    The practical recipe:

    1. **Start with frozen features** to train the new head. This gives the head a reasonable initialization before you start moving the backbone.
    2. **Unfreeze all layers** (or just the later layers).
    3. **Use a learning rate 10--100x smaller** than you would for training from scratch. Typical values: 1e-4 to 1e-5 for the backbone, 1e-3 for the head.

    ### Differential Learning Rates

    An even better approach is **differential learning rates**: use a different learning rate for each group of layers. The intuition is direct:

    - **Early layers** (edges, textures): barely need to change. Use the smallest LR.
    - **Middle layers** (parts, compositions): may need moderate adaptation. Medium LR.
    - **Late layers + head** (task-specific): need the most change. Highest LR.

    PyTorch supports this through **parameter groups** in the optimizer:

    ```python
    optimizer = optim.Adam([
        {"params": model.layer1.parameters(), "lr": 1e-5},
        {"params": model.layer2.parameters(), "lr": 3e-5},
        {"params": model.layer3.parameters(), "lr": 1e-4},
        {"params": model.layer4.parameters(), "lr": 3e-4},
        {"params": model.fc.parameters(),     "lr": 1e-3},
    ])
    ```

    This gives you fine-grained control over how much each part of the network adapts. In practice, even a simple two-group split (backbone at 1e-4, head at 1e-3) works well.

    > **Reading:** [Geron, Ch 11](file:///C:/Users/landa/ml-course/textbooks/Geron.pdf) covers fine-tuning strategies, including the "freeze then unfreeze" protocol.
    """)
    return


@app.cell
def _(nn, torch, optim, train_loader, test_loader, resnet18, ResNet18_Weights, train_and_evaluate, device_str):
    # Fine-tuning with differential learning rates
    def build_finetuned_model(num_classes):
        """Load pretrained ResNet-18 for fine-tuning with differential LRs."""
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        # All parameters are trainable (requires_grad=True by default)
        return model

    print("--- Strategy C: Fine-tuned ResNet-18 ---")
    ft_model = build_finetuned_model(num_classes=10)
    ft_model = ft_model.to(device_str)

    # Differential learning rates: backbone gets 1e-4, head gets 1e-3
    backbone_params = []
    head_params = []
    for name, param in ft_model.named_parameters():
        if "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    ft_optimizer = optim.Adam([
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params, "lr": 1e-3},
    ])
    criterion_ft = nn.CrossEntropyLoss()

    ft_losses = []
    ft_accs = []

    for epoch in range(15):
        ft_model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device_str), y_b.to(device_str)
            ft_optimizer.zero_grad()
            out = ft_model(X_b)
            loss = criterion_ft(out, y_b)
            loss.backward()
            ft_optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        ft_losses.append(epoch_loss / n_batches)

        ft_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(device_str), y_b.to(device_str)
                preds = ft_model(X_b).argmax(dim=1)
                correct += (preds == y_b).sum().item()
                total += y_b.size(0)
        ft_accs.append(correct / total)

        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"  [Fine-tune] Epoch {epoch+1}/15 -- "
                  f"loss: {ft_losses[-1]:.4f}, test acc: {ft_accs[-1]:.4f}")

    print(f"\nFinal fine-tuned test accuracy: {ft_accs[-1]:.4f}")
    return (ft_losses, ft_accs, build_finetuned_model)


@app.cell
def _(plt, scratch_accs, frozen_accs, ft_accs):
    # Three-way comparison: scratch vs frozen vs fine-tuned
    fig_three, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs_r = range(1, len(scratch_accs) + 1)
    ax_curve.plot(epochs_r, scratch_accs, "o-", label="A: From scratch", markersize=4)
    ax_curve.plot(epochs_r, frozen_accs, "s-", label="B: Frozen backbone", markersize=4)
    ax_curve.plot(epochs_r, ft_accs, "^-", label="C: Fine-tuned", markersize=4)
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Test Accuracy")
    ax_curve.set_title("Three Strategies on CIFAR-10 (1000 images)")
    ax_curve.legend()
    ax_curve.set_ylim(0, 1)
    ax_curve.grid(True, alpha=0.3)

    names_3 = ["From Scratch", "Frozen", "Fine-tuned"]
    finals_3 = [scratch_accs[-1], frozen_accs[-1], ft_accs[-1]]
    colors_3 = ["#d9534f", "#5cb85c", "#0275d8"]
    ax_bar.bar(names_3, finals_3, color=colors_3, edgecolor="black", linewidth=0.5)
    ax_bar.set_ylabel("Final Test Accuracy")
    ax_bar.set_title("Final Comparison")
    ax_bar.set_ylim(0, 1)
    for i, v in enumerate(finals_3):
        ax_bar.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig_three
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### When to Use Each Strategy

    The three strategies form a spectrum from least to most adaptation:

    | Strategy | When to use | Data needed | Compute cost |
    |---|---|---|---|
    | **Frozen backbone** | Target domain is very similar to source; very few labeled examples (tens to hundreds per class) | Minimal | Very low --- only train a linear layer |
    | **Fine-tuning** | Target domain is moderately different; small-to-medium labeled dataset (hundreds to thousands per class) | Moderate | Medium --- backprop through full network but fewer epochs |
    | **From scratch** | Target domain is radically different from any available pretrained model; large labeled dataset available | Large | High --- full training run |

    In practice, training from scratch is rare. Even for specialized domains like medical imaging or satellite analysis, starting from ImageNet weights and fine-tuning almost always beats training from scratch, unless you have truly massive domain-specific datasets. The pretrained features provide a form of regularization --- they constrain the network to stay near a known-good region of weight space.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Sequence Modeling in Practice

    Module 2E gave you the theory of recurrent neural networks: the recurrence relation, vanishing gradients, gating mechanisms in LSTMs and GRUs. Now we implement them in PyTorch and build a working sequence predictor.

    The `nn.LSTM` interface is straightforward but has a few quirks worth understanding:

    ```python
    lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
    ```

    - **input_size**: dimension of each input element (1 for a scalar time series, more for multivariate)
    - **hidden_size**: dimension of the hidden state vector
    - **num_layers**: number of stacked LSTM layers (layer 2 takes layer 1's output as input)
    - **batch_first=True**: input shape is `(batch, seq_len, features)` instead of the default `(seq_len, batch, features)`

    The output is a tuple: `(output, (h_n, c_n))`, where:
    - `output` has shape `(batch, seq_len, hidden_size)` --- the hidden state at EVERY timestep
    - `h_n` has shape `(num_layers, batch, hidden_size)` --- the FINAL hidden state for each layer
    - `c_n` has shape `(num_layers, batch, hidden_size)` --- the FINAL cell state for each layer

    For **many-to-one** tasks (predict a single value from a sequence), you typically use `output[:, -1, :]` --- the hidden state at the last timestep. For **many-to-many** tasks (predict at every timestep), you use the full `output` tensor.

    > **Reading:** [DLBook, Section 10.7](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) discusses practical RNN design choices. [Geron, Ch 15](file:///C:/Users/landa/ml-course/textbooks/Geron.pdf) covers RNN implementation details.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Building a Sequence Predictor: Noisy Sine Wave

    We will build an LSTM that predicts the next value in a noisy sine wave. Why a sine wave? It has clean periodic structure, making it easy to verify that the model is actually learning temporal patterns rather than memorizing. We add noise to make the task non-trivial.

    The setup:
    - Generate a long sine wave with Gaussian noise
    - Create sliding windows: input = `[t_i, ..., t_{i+L}]`, target = `t_{i+L+1}`
    - **Critical**: split train/test *temporally* --- the test set is the future, not a random subset. Shuffling across time would leak future information into training.
    """)
    return


@app.cell
def _(np, plt):
    # Generate a noisy sine wave
    rng_seq = np.random.default_rng(42)
    t_points = np.linspace(0, 20 * np.pi, 2000)
    signal = np.sin(t_points) + 0.1 * rng_seq.standard_normal(len(t_points))

    fig_sig, ax_sig = plt.subplots(figsize=(12, 3))
    ax_sig.plot(t_points[:500], signal[:500], linewidth=0.8)
    ax_sig.set_xlabel("t")
    ax_sig.set_ylabel("signal(t)")
    ax_sig.set_title("Noisy Sine Wave (first 500 points)")
    ax_sig.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_sig
    return (signal, t_points)


@app.cell
def _(DataLoader, TensorDataset, np, torch, signal):
    # Create sliding window dataset
    window_size = 50  # use 50 timesteps to predict the next one

    def create_sequences(data, window):
        X_list, y_list = [], []
        for i in range(len(data) - window):
            X_list.append(data[i : i + window])
            y_list.append(data[i + window])
        return np.array(X_list), np.array(y_list)

    X_seq, y_seq = create_sequences(signal, window_size)

    # Temporal split: first 80% for training, last 20% for testing
    split_idx = int(0.8 * len(X_seq))
    X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
    y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]

    # Convert to PyTorch tensors -- LSTM expects (batch, seq_len, features)
    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(-1)
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32).unsqueeze(-1)
    y_test_t = torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(-1)

    seq_train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True
    )

    print(f"Training sequences: {X_train_t.shape[0]}")
    print(f"Test sequences:     {X_test_t.shape[0]}")
    print(f"Input shape:        {X_train_t.shape}  (batch, seq_len, features)")
    return (
        window_size, create_sequences,
        X_train_t, y_train_t, X_test_t, y_test_t,
        seq_train_loader,
    )


@app.cell
def _(nn, torch, optim, seq_train_loader, X_test_t, y_test_t):
    class LSTMPredictor(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            # output shape: (batch, seq_len, hidden_size)
            output, (h_n, c_n) = self.lstm(x)
            # Use the last timestep's hidden state for prediction
            last_hidden = output[:, -1, :]  # (batch, hidden_size)
            return self.fc(last_hidden)     # (batch, 1)

    lstm_model = LSTMPredictor(input_size=1, hidden_size=64, num_layers=2)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)
    lstm_criterion = nn.MSELoss()

    lstm_train_losses = []

    for epoch in range(30):
        lstm_model.train()
        epoch_loss = 0.0
        n_b = 0
        for X_b, y_b in seq_train_loader:
            lstm_optimizer.zero_grad()
            pred = lstm_model(X_b)
            loss = lstm_criterion(pred, y_b)
            loss.backward()
            lstm_optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        lstm_train_losses.append(epoch_loss / n_b)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30 -- train MSE: {lstm_train_losses[-1]:.6f}")

    # Evaluate on test set
    lstm_model.eval()
    with torch.no_grad():
        test_pred = lstm_model(X_test_t).squeeze().numpy()
        test_actual = y_test_t.squeeze().numpy()
        test_mse = ((test_pred - test_actual) ** 2).mean()
    print(f"\nTest MSE: {test_mse:.6f}")
    return (lstm_model, test_pred, test_actual, lstm_train_losses, LSTMPredictor)


@app.cell
def _(plt, test_pred, test_actual):
    fig_lstm, (ax_pred, ax_loss_l) = plt.subplots(2, 1, figsize=(12, 7))

    # Plot predicted vs actual on test region
    n_show = min(300, len(test_actual))
    ax_pred.plot(range(n_show), test_actual[:n_show], label="Actual", linewidth=1.2)
    ax_pred.plot(range(n_show), test_pred[:n_show], label="Predicted", linewidth=1.2,
                 linestyle="--", alpha=0.8)
    ax_pred.set_xlabel("Test timestep")
    ax_pred.set_ylabel("Value")
    ax_pred.set_title("LSTM Predictions vs Actual (Test Region)")
    ax_pred.legend()
    ax_pred.grid(True, alpha=0.3)

    # Scatter plot: predicted vs actual
    ax_loss_l.scatter(test_actual[:n_show], test_pred[:n_show], alpha=0.3, s=10)
    lims = [min(test_actual.min(), test_pred.min()) - 0.1,
            max(test_actual.max(), test_pred.max()) + 0.1]
    ax_loss_l.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax_loss_l.set_xlabel("Actual")
    ax_loss_l.set_ylabel("Predicted")
    ax_loss_l.set_title("Predicted vs Actual Scatter")
    ax_loss_l.legend()
    ax_loss_l.grid(True, alpha=0.3)
    ax_loss_l.set_aspect("equal")

    plt.tight_layout()
    fig_lstm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Experimenting with Capacity

    The LSTM has two main capacity knobs: **hidden size** (dimension of the hidden state) and **number of layers** (depth of the recurrent stack). More capacity means the model can represent more complex temporal patterns, but also means more parameters and greater risk of overfitting.

    For a simple periodic signal like a sine wave, a small LSTM (hidden_size=16, 1 layer) is sufficient. But for complex, multi-scale temporal patterns, you need more capacity. The interactive widget below lets you explore this tradeoff.
    """)
    return


@app.cell
def _(mo):
    hidden_slider = mo.ui.slider(
        start=8, stop=128, step=8, value=32, label="Hidden size"
    )
    layers_dropdown = mo.ui.dropdown(
        options={"1 layer": "1", "2 layers": "2", "3 layers": "3"},
        value="1", label="Number of LSTM layers"
    )
    mo.hstack([hidden_slider, layers_dropdown])
    return (hidden_slider, layers_dropdown)


@app.cell
def _(mo, hidden_slider, layers_dropdown, nn, torch, optim, seq_train_loader, X_test_t, y_test_t, LSTMPredictor):
    _hidden = hidden_slider.value
    _layers = int(layers_dropdown.value)

    _model = LSTMPredictor(input_size=1, hidden_size=_hidden, num_layers=_layers)
    _opt = optim.Adam(_model.parameters(), lr=1e-3)
    _crit = nn.MSELoss()

    _n_params = sum(p.numel() for p in _model.parameters())

    for _ep in range(20):
        _model.train()
        for _xb, _yb in seq_train_loader:
            _opt.zero_grad()
            _loss = _crit(_model(_xb), _yb)
            _loss.backward()
            _opt.step()

    _model.eval()
    with torch.no_grad():
        _tp = _model(X_test_t).squeeze().numpy()
        _ta = y_test_t.squeeze().numpy()
        _mse = ((_tp - _ta) ** 2).mean()

    mo.md(f"""
    **Hidden size = {_hidden}, Layers = {_layers}** --- Parameters: {_n_params:,} --- Test MSE: {_mse:.6f}

    For a simple sine wave, even small LSTMs perform well. Increasing capacity beyond what the signal requires does not help and can hurt (overfitting to noise). The key takeaway: match model capacity to signal complexity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Autoencoders and Latent Spaces

    Module 3B introduced autoencoders theoretically: an encoder compresses the input through a bottleneck, a decoder reconstructs it, and the bottleneck forces the network to learn a compressed representation. Now we build one and explore what that representation looks like.

    The key insight worth repeating: **autoencoders learn representations without labels**. This is unsupervised learning. The only supervision signal is reconstruction error --- can the network reconstruct its input after compressing it through a bottleneck? If yes, the bottleneck must capture the essential structure of the data.

    ### Architecture for MNIST

    We will build an autoencoder for MNIST with a **2-dimensional latent space**. Why 2D? Not because 2D is optimal for reconstruction --- it is far too small for that. We choose 2D so we can *visualize* the latent space directly. Plotting z1 vs z2 will reveal how the network organizes digit representations in latent space.

    The architecture:
    - **Encoder**: 784 (28x28 flattened) -> 256 -> 64 -> 2
    - **Decoder**: 2 -> 64 -> 256 -> 784
    - **Output activation**: sigmoid (pixel values in [0, 1])
    - **Loss**: mean squared error between input and reconstruction

    The progressive compression (784 -> 256 -> 64 -> 2) forces increasingly aggressive abstraction at each layer. By the time information reaches the 2D bottleneck, the network must have distilled each digit down to just two numbers. Those two numbers constitute the learned representation.

    > **Reading:** [DLBook, Section 14.1--14.2](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) covers autoencoder theory. The connection to PCA (linear autoencoders learn PCA subspaces) is in [DLBook, Section 14.1](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf).
    """)
    return


@app.cell
def _(DataLoader, torch, nn):
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor

    # Load MNIST
    mnist_train = MNIST(root="../data", train=True, download=True, transform=ToTensor())
    mnist_test = MNIST(root="../data", train=False, download=True, transform=ToTensor())

    ae_train_loader = DataLoader(mnist_train, batch_size=256, shuffle=True)
    ae_test_loader = DataLoader(mnist_test, batch_size=256, shuffle=False)

    class Autoencoder(nn.Module):
        def __init__(self, latent_dim=2):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 256), nn.ReLU(),
                nn.Linear(256, 64), nn.ReLU(),
                nn.Linear(64, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64), nn.ReLU(),
                nn.Linear(64, 256), nn.ReLU(),
                nn.Linear(256, 784), nn.Sigmoid(),  # pixel values in [0, 1]
            )

        def forward(self, x):
            z = self.encoder(x)
            x_recon = self.decoder(z)
            return x_recon, z

    ae_model = Autoencoder(latent_dim=2)
    n_ae_params = sum(p.numel() for p in ae_model.parameters())
    print(f"Autoencoder parameters: {n_ae_params:,}")
    print(f"Encoder: 784 -> 256 -> 64 -> 2")
    print(f"Decoder: 2 -> 64 -> 256 -> 784")
    return (mnist_train, mnist_test, ae_train_loader, ae_test_loader, Autoencoder, ae_model)


@app.cell
def _(nn, torch, optim, ae_model, ae_train_loader):
    ae_optimizer = optim.Adam(ae_model.parameters(), lr=1e-3)
    ae_criterion = nn.MSELoss()

    ae_losses = []
    for epoch in range(20):
        ae_model.train()
        epoch_loss = 0.0
        n_b = 0
        for images, _ in ae_train_loader:  # labels are ignored!
            x_flat = images.view(images.size(0), -1)
            ae_optimizer.zero_grad()
            x_recon, z = ae_model(images)
            loss = ae_criterion(x_recon, x_flat)
            loss.backward()
            ae_optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        ae_losses.append(epoch_loss / n_b)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/20 -- reconstruction MSE: {ae_losses[-1]:.6f}")

    print(f"\nFinal reconstruction MSE: {ae_losses[-1]:.6f}")
    return (ae_losses,)


@app.cell
def _(torch, ae_model, ae_test_loader, plt, np):
    # Visualize reconstructions
    ae_model.eval()
    with torch.no_grad():
        sample_images, sample_labels = next(iter(ae_test_loader))
        x_flat = sample_images.view(sample_images.size(0), -1)
        recons, latent_codes = ae_model(sample_images)

    n_display = 8
    fig_recon, axes_recon = plt.subplots(2, n_display, figsize=(14, 3.5))
    for i in range(n_display):
        axes_recon[0, i].imshow(x_flat[i].reshape(28, 28).numpy(), cmap="gray")
        axes_recon[0, i].axis("off")
        if i == 0:
            axes_recon[0, i].set_title("Original", fontsize=10)
        axes_recon[1, i].imshow(recons[i].reshape(28, 28).numpy(), cmap="gray")
        axes_recon[1, i].axis("off")
        if i == 0:
            axes_recon[1, i].set_title("Reconstructed", fontsize=10)

    fig_recon.suptitle("Autoencoder Reconstructions (2D Latent Space)", y=1.02)
    plt.tight_layout()
    fig_recon
    return (latent_codes, sample_labels)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The reconstructions are blurry --- that is expected with a 2D bottleneck. You are compressing 784 dimensions down to 2. The fact that the reconstructions are recognizable at all is remarkable. The network has learned to encode the *essential identity* of each digit in just two numbers.

    ### Latent Space Visualization

    Now for the most revealing visualization. We encode the entire test set through the trained encoder and plot the resulting 2D codes, colored by digit label. Remember: the autoencoder never saw any labels during training. Any structure we see in the latent space was discovered purely from reconstruction pressure.
    """)
    return


@app.cell
def _(torch, ae_model, ae_test_loader, plt, np):
    # Encode the full test set
    ae_model.eval()
    all_z = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls in ae_test_loader:
            _, z_batch = ae_model(imgs)
            all_z.append(z_batch.numpy())
            all_labels.append(lbls.numpy())

    all_z = np.concatenate(all_z, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    fig_latent, ax_latent = plt.subplots(figsize=(8, 7))
    scatter = ax_latent.scatter(
        all_z[:, 0], all_z[:, 1],
        c=all_labels, cmap="tab10", alpha=0.4, s=5, edgecolors="none"
    )
    cbar = plt.colorbar(scatter, ax=ax_latent, ticks=range(10))
    cbar.set_label("Digit")
    ax_latent.set_xlabel("z1")
    ax_latent.set_ylabel("z2")
    ax_latent.set_title("Latent Space: 10,000 Test Digits Encoded to 2D")
    ax_latent.grid(True, alpha=0.2)
    plt.tight_layout()
    fig_latent
    return (all_z, all_labels)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The digits cluster. Without ever seeing a label, the autoencoder has discovered that certain images are similar and should map to nearby points in latent space. Ones are far from eights. Fours are far from zeros. Similar digits (3 and 5, or 4 and 9) may overlap --- reflecting genuine visual similarity.

    This is representation learning in its purest form: the network learns a coordinate system where distance corresponds to semantic similarity, driven solely by the pressure to reconstruct.

    ### Interpolation: The Hallmark of a Good Representation

    If the latent space is smooth and well-structured, then linearly interpolating between two latent codes should produce a smooth morphing between the corresponding digits. This is a stringent test: it requires that the "space between" two digits is filled with plausible intermediate forms, not garbage.
    """)
    return


@app.cell
def _(torch, ae_model, np, plt):
    # Interpolation between two latent points
    ae_model.eval()

    # Pick two points in latent space (encode two specific test digits)
    z_start = torch.tensor([[-3.0, -1.0]], dtype=torch.float32)  # a region of latent space
    z_end = torch.tensor([[3.0, 2.0]], dtype=torch.float32)      # another region

    n_interp = 10
    alphas = np.linspace(0, 1, n_interp)
    z_interp = torch.stack([
        (1 - a) * z_start + a * z_end for a in alphas
    ]).squeeze(1)

    with torch.no_grad():
        decoded = ae_model.decoder(z_interp).numpy()

    fig_interp, axes_interp = plt.subplots(1, n_interp, figsize=(14, 2))
    for i in range(n_interp):
        axes_interp[i].imshow(decoded[i].reshape(28, 28), cmap="gray")
        axes_interp[i].axis("off")
        axes_interp[i].set_title(f"a={alphas[i]:.1f}", fontsize=8)

    fig_interp.suptitle("Latent Space Interpolation", y=1.05)
    plt.tight_layout()
    fig_interp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Grid Sampling: What Does the Decoder "Imagine"?

    We can also decode a uniform grid of latent points to see what the decoder produces for each region of latent space. This creates a map of the decoder's "imagination" --- each grid cell shows what the network thinks a digit at that latent coordinate should look like.
    """)
    return


@app.cell
def _(torch, ae_model, np, plt):
    # Decode a grid of latent points
    ae_model.eval()

    grid_size = 15
    # Determine grid range from the latent space extent
    z_range = np.linspace(-4, 4, grid_size)

    canvas = np.zeros((28 * grid_size, 28 * grid_size))

    with torch.no_grad():
        for i, z1_val in enumerate(z_range):
            for j, z2_val in enumerate(reversed(z_range)):
                z_grid = torch.tensor([[z1_val, z2_val]], dtype=torch.float32)
                decoded_img = ae_model.decoder(z_grid).numpy().reshape(28, 28)
                canvas[j * 28 : (j + 1) * 28, i * 28 : (i + 1) * 28] = decoded_img

    fig_grid, ax_grid = plt.subplots(figsize=(8, 8))
    ax_grid.imshow(canvas, cmap="gray")
    ax_grid.set_title("Decoded Grid: Sampling the Latent Space")
    ax_grid.set_xlabel("z1")
    ax_grid.set_ylabel("z2")
    # Set tick labels to show the actual z values
    tick_positions = np.linspace(14, 28 * grid_size - 14, 5)
    tick_labels_z = np.linspace(-4, 4, 5)
    ax_grid.set_xticks(tick_positions)
    ax_grid.set_xticklabels([f"{v:.1f}" for v in tick_labels_z])
    ax_grid.set_yticks(tick_positions)
    ax_grid.set_yticklabels([f"{v:.1f}" for v in reversed(tick_labels_z)])
    plt.tight_layout()
    fig_grid
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the gaps and blurry regions. Standard autoencoders have "holes" in the latent space --- regions that do not correspond to any training data and produce meaningless outputs when decoded. This is exactly the problem that Variational Autoencoders (VAEs) solve by regularizing the latent space to be a smooth Gaussian, as covered in Module 3B. The VAE's KL divergence penalty fills in the gaps and ensures every point in latent space decodes to something plausible.

    But even without that regularization, the autoencoder has accomplished something impressive: it has learned a meaningful 2D coordinate system for handwritten digits, entirely unsupervised. This is representation learning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. Exercises

    These exercises reinforce the key ideas: transfer learning (frozen and fine-tuned), sequence modeling with LSTMs, and autoencoder representations. Work through them in order --- each builds on the concepts demonstrated above.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 1: Feature Extraction with a Pretrained ResNet

    Load a pretrained ResNet-18, remove the classification head, and use the backbone as a fixed feature extractor. Extract 512-dimensional feature vectors for CIFAR-10 images, then train a simple `nn.Linear` classifier on those features.

    Steps:
    1. Forward each batch through the frozen backbone (everything up to the final FC layer)
    2. Collect the resulting feature vectors and labels
    3. Train a linear classifier (`nn.Linear(512, 10)`) on the extracted features
    4. Report the test accuracy

    This separates the feature extraction step from the classification step, making the tradeoff between pretrained features and learned classifiers very explicit.
    """)
    return


@app.cell
def _(nn, torch, optim, DataLoader, TensorDataset, resnet18, ResNet18_Weights, train_loader, test_loader):
    def extract_features(dataloader, device="cpu"):
        """Extract features using a frozen ResNet-18 backbone (without the final FC layer)."""
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()  # replace classifier with identity
        backbone = backbone.to(device)
        backbone.eval()

        all_features = []
        all_labels = []

        # TODO: iterate over the dataloader, forward-pass through backbone,
        # collect features and labels. Remember to use torch.no_grad().
        # with torch.no_grad():
        #     for X_batch, y_batch in dataloader:
        #         X_batch = X_batch.to(device)
        #         feats = backbone(X_batch)          # shape: (batch, 512)
        #         all_features.append(feats.cpu())
        #         all_labels.append(y_batch)

        # features = torch.cat(all_features, dim=0)
        # labels = torch.cat(all_labels, dim=0)
        # return features, labels
        pass

    # TODO: Extract features for train and test sets
    # train_feats, train_labels = extract_features(train_loader)
    # test_feats, test_labels = extract_features(test_loader)

    # TODO: Build a DataLoader from the extracted features
    # feat_train_loader = DataLoader(TensorDataset(train_feats, train_labels),
    #                                batch_size=256, shuffle=True)

    # TODO: Train a linear classifier: nn.Linear(512, 10)
    # linear_clf = nn.Linear(512, 10)
    # optimizer = optim.Adam(linear_clf.parameters(), lr=1e-3)
    # criterion = nn.CrossEntropyLoss()
    #
    # for epoch in range(20):
    #     linear_clf.train()
    #     for feats_b, labels_b in feat_train_loader:
    #         optimizer.zero_grad()
    #         logits = linear_clf(feats_b)
    #         loss = criterion(logits, labels_b)
    #         loss.backward()
    #         optimizer.step()

    # TODO: Evaluate on test features
    # linear_clf.eval()
    # with torch.no_grad():
    #     test_logits = linear_clf(test_feats)
    #     test_preds = test_logits.argmax(dim=1)
    #     accuracy = (test_preds == test_labels).float().mean().item()
    # print(f"Linear classifier on ResNet features: {accuracy:.4f}")

    print("Exercise 1 skeleton ready -- fill in the TODOs")
    return (extract_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2: Fine-Tuning Learning Rate Experiment

    Fine-tune a pretrained ResNet-18 on the CIFAR-10 subset using three different backbone learning rates: 1e-3 (too large), 1e-4 (reasonable), and 1e-6 (too small). Keep the head learning rate at 1e-3 throughout. Compare the final test accuracies.

    Expected behavior:
    - LR=1e-3 for backbone: catastrophic forgetting --- pretrained features are destroyed, accuracy may be poor
    - LR=1e-4 for backbone: good adaptation --- features adjust without being destroyed
    - LR=1e-6 for backbone: effectively frozen --- similar to feature extraction

    Plot all three learning curves and report which backbone LR performs best.
    """)
    return


@app.cell
def _(nn, torch, optim, resnet18, ResNet18_Weights, train_loader, test_loader):
    def finetune_experiment(backbone_lr, head_lr=1e-3, epochs=10, device="cpu"):
        """Fine-tune ResNet-18 with specified backbone LR. Return test accuracies per epoch."""
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model = model.to(device)

        # TODO: Create optimizer with two parameter groups:
        #   - backbone parameters at backbone_lr
        #   - head parameters (model.fc) at head_lr
        # backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
        # head_params = list(model.fc.parameters())
        # optimizer = optim.Adam([
        #     {"params": backbone_params, "lr": backbone_lr},
        #     {"params": head_params, "lr": head_lr},
        # ])

        # TODO: Training loop for `epochs` epochs
        # Track test accuracy each epoch
        # accs = []
        # criterion = nn.CrossEntropyLoss()
        # for epoch in range(epochs):
        #     model.train()
        #     for X_b, y_b in train_loader:
        #         X_b, y_b = X_b.to(device), y_b.to(device)
        #         optimizer.zero_grad()
        #         loss = criterion(model(X_b), y_b)
        #         loss.backward()
        #         optimizer.step()
        #
        #     model.eval()
        #     correct, total = 0, 0
        #     with torch.no_grad():
        #         for X_b, y_b in test_loader:
        #             X_b, y_b = X_b.to(device), y_b.to(device)
        #             correct += (model(X_b).argmax(1) == y_b).sum().item()
        #             total += y_b.size(0)
        #     accs.append(correct / total)
        #     print(f"  backbone_lr={backbone_lr}, epoch {epoch+1}, acc={accs[-1]:.4f}")
        # return accs
        pass

    # TODO: Run for three backbone LRs and compare
    # results = {}
    # for blr in [1e-3, 1e-4, 1e-6]:
    #     print(f"\n--- Backbone LR = {blr} ---")
    #     results[blr] = finetune_experiment(blr, device="cpu")

    # TODO: Plot all three curves
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(8, 5))
    # for blr, accs in results.items():
    #     ax.plot(range(1, len(accs)+1), accs, "o-", label=f"backbone LR={blr}")
    # ax.set_xlabel("Epoch"); ax.set_ylabel("Test Accuracy")
    # ax.legend(); ax.grid(True, alpha=0.3)
    # plt.tight_layout()

    print("Exercise 2 skeleton ready -- fill in the TODOs")
    return (finetune_experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3: LSTM for Complex Signal Prediction

    Build an LSTM predictor for a more complex signal: the sum of two sine waves with different frequencies. This tests whether the LSTM can capture multi-scale temporal structure.

    Signal: `sin(t) + 0.5 * sin(3t) + noise`

    Steps:
    1. Generate the composite signal
    2. Create sliding-window sequences (use window_size=50)
    3. Temporal train/test split (80/20)
    4. Train an LSTM and evaluate on the test region
    5. Visualize predictions vs actual
    """)
    return


@app.cell
def _(np, torch, nn, optim, DataLoader, TensorDataset):
    def _run():
        rng_ex = np.random.default_rng(123)
        t_ex = np.linspace(0, 20 * np.pi, 2000)

        # TODO: Generate composite signal: sin(t) + 0.5*sin(3t) + noise
        # signal_ex = np.sin(t_ex) + 0.5 * np.sin(3 * t_ex) + 0.1 * rng_ex.standard_normal(len(t_ex))

        # TODO: Create sliding window sequences (window_size=50)
        # Use the create_sequences function pattern from Section 4

        # TODO: Temporal split (80/20)

        # TODO: Convert to PyTorch tensors with shape (batch, seq_len, 1)

        # TODO: Build LSTM model, train for 30 epochs

        # TODO: Evaluate and visualize
        # Plot predicted vs actual for the test region

        pass

    _run()
    print("Exercise 3 skeleton ready -- fill in the TODOs")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 4: Higher-Dimensional Autoencoder + PCA Comparison

    Build an autoencoder with a 32-dimensional latent space (instead of 2D). Then:

    1. Encode the MNIST test set to get 32D latent codes
    2. Run PCA on the 32D latent codes to reduce to 2D for visualization
    3. Also run PCA directly on the raw 784D pixel vectors to get 2D
    4. Plot both 2D projections side-by-side, colored by digit label
    5. Compare: which gives better cluster separation?

    The autoencoder's latent space should show tighter, more separated clusters because it learns *nonlinear* features, while PCA on raw pixels is limited to linear projections.
    """)
    return


@app.cell
def _(nn, torch, optim, DataLoader, np, plt):
    def _run():
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
        from sklearn.decomposition import PCA

        # TODO: Build autoencoder with latent_dim=32
        # class AE32(nn.Module):
        #     def __init__(self):
        #         super().__init__()
        #         self.encoder = nn.Sequential(
        #             nn.Flatten(),
        #             nn.Linear(784, 256), nn.ReLU(),
        #             nn.Linear(256, 64), nn.ReLU(),
        #             nn.Linear(64, 32),
        #         )
        #         self.decoder = nn.Sequential(
        #             nn.Linear(32, 64), nn.ReLU(),
        #             nn.Linear(64, 256), nn.ReLU(),
        #             nn.Linear(256, 784), nn.Sigmoid(),
        #         )
        #     def forward(self, x):
        #         z = self.encoder(x)
        #         return self.decoder(z), z

        # TODO: Train the autoencoder on MNIST (20 epochs)

        # TODO: Encode test set to get 32D latent codes
        # Also get raw pixel vectors for the test set

        # TODO: PCA on latent codes (32D -> 2D)
        # pca_latent = PCA(n_components=2).fit_transform(latent_codes)

        # TODO: PCA on raw pixels (784D -> 2D)
        # pca_raw = PCA(n_components=2).fit_transform(raw_pixels)

        # TODO: Plot side by side, colored by label
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        # ax1.scatter(pca_raw[:, 0], pca_raw[:, 1], c=labels, cmap="tab10", s=3, alpha=0.4)
        # ax1.set_title("PCA on Raw Pixels (784D -> 2D)")
        # ax2.scatter(pca_latent[:, 0], pca_latent[:, 1], c=labels, cmap="tab10", s=3, alpha=0.4)
        # ax2.set_title("PCA on AE Latent Codes (32D -> 2D)")

        pass

    _run()
    print("Exercise 4 skeleton ready -- fill in the TODOs")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pencil Exercise

    **Why does a frozen ResNet need `model.eval()`? What would go wrong without it?**

    Think carefully about which layers in ResNet have different behavior in training vs eval mode. There are exactly two layer types that behave differently:

    1. **BatchNorm layers** use batch statistics (mean/variance of the current mini-batch) in training mode, but running statistics (accumulated during pretraining on ImageNet) in eval mode. If you leave the model in training mode, BatchNorm will compute statistics from your small CIFAR batches, which may have very different distributions from ImageNet. This corrupts the normalization and degrades feature quality significantly.

    2. **Dropout layers** (if present) randomly zero activations in training mode but pass everything through in eval mode. ResNet does not use dropout, but other architectures (e.g., VGG, AlexNet) do. Leaving dropout active on a frozen backbone would randomly destroy pretrained features for no reason --- since those layers are not being trained, there is no regularization benefit.

    The key insight: `model.eval()` is not about training vs inference. It is about which statistics and stochastic behaviors to use. For a frozen backbone, you always want the pretrained behavior, which means eval mode.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    > **Back to**: [Course Home](./home/) | [Algorithm Study Guide](./6a_study_guide/)
    """)
    return


if __name__ == "__main__":
    app.run()

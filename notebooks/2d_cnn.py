import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
# Convolutional Neural Networks

This is the first lecture where we move beyond the general-purpose neural network and study a specific architecture designed for a specific kind of data. Everything you've learned so far — feedforward networks, backpropagation, gradient-based optimization, regularization — still applies. But now we add structure. And adding the right structure is arguably the most important decision in all of deep learning.

CNNs are the architecture that brought deep learning back from the dead. They won ImageNet in 2012, and that single result triggered the entire deep learning revolution we're living through. Understanding them deeply isn't optional — even if you never work on computer vision, the design principles behind CNNs (parameter sharing, locality, hierarchical feature extraction) appear everywhere in modern architectures, including transformers.

> **Reading:** [DLBook Ch 9](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) is the primary reference for this entire lecture. Read it alongside these notes.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 1. The Problem with Fully Connected Layers for Images

Before we build anything new, let's understand why fully connected networks fail on images. This isn't a matter of taste — it's a matter of mathematics.

Consider a modest $256 \times 256$ RGB image. Each pixel has 3 color channels, so the input is a vector of $256 \times 256 \times 3 = 196{,}608$ values. If the first hidden layer has, say, 1000 neurons (small by modern standards), the weight matrix for just that one layer has $196{,}608 \times 1{,}000 \approx 200$ million parameters. A second layer of the same size adds another billion. This is absurd — you'd need astronomical amounts of data to train this without catastrophic overfitting, and the memory requirements alone are prohibitive.

But the parameter count isn't even the real problem. The deeper issue is that a fully connected layer **treats every pixel as independent and unrelated to every other pixel**. A neuron connected to all 196K inputs has to learn from scratch that pixel (100, 50) is near pixel (101, 50) and far from pixel (200, 200). It has to learn from scratch that an edge in the top-left corner is the same kind of thing as an edge in the bottom-right corner. It ignores everything we know about the structure of images.

Images have two critical properties that we should exploit:

1. **Locality:** Meaningful patterns (edges, textures, object parts) are spatially local. A pixel is much more related to its neighbors than to pixels far away.
2. **Translation invariance:** An edge is an edge regardless of where it appears in the image. A cat ear in the top-left should be detected by the same mechanism as a cat ear in the bottom-right.

The convolutional neural network is what you get when you bake these two priors directly into the architecture. Instead of learning fully general weight matrices, we constrain the network to use small, local filters that are shared across all spatial positions. This is not just a computational convenience — it's an inductive bias that encodes real knowledge about the structure of visual data.

> **Reading:** [DLBook §9.1-9.2](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) motivates convolution from the perspective of sparse interactions, parameter sharing, and equivariant representations.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 2. The Convolution Operation

### Starting with 1D

Let's build intuition in one dimension first. Suppose you have a 1D input signal $x = [x_1, x_2, \ldots, x_n]$ and a small kernel (also called a filter) $k = [k_1, k_2, k_3]$ of length 3. The convolution slides the kernel across the input, computing a weighted sum at each position:

$$(x * k)[i] = \sum_{m} x[i + m] \cdot k[m]$$

Concretely, at position $i$, you take the three input values $x[i], x[i+1], x[i+2]$, multiply them element-wise by $k[1], k[2], k[3]$, and sum. Then you slide one step to the right and repeat. The output is a new sequence — shorter than the input (we'll fix that with padding shortly).

What does this do? It depends entirely on the kernel values. If $k = [-1, 0, 1]$, the output at position $i$ is $x[i+2] - x[i]$ — a discrete derivative, detecting changes in the signal. If $k = [1/3, 1/3, 1/3]$, you get a moving average, smoothing the signal. The kernel determines what pattern the convolution detects.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Extending to 2D

For images, the input is a 2D grid (or 3D, counting channels), and the kernel is a small 2D matrix, typically $3 \times 3$ or $5 \times 5$. The kernel slides across both height and width:

$$(I * K)[i, j] = \sum_{m=0}^{h-1} \sum_{n=0}^{w-1} I[i+m,\; j+n] \cdot K[m, n]$$

where $K$ is an $h \times w$ kernel. At each spatial position $(i, j)$, you place the kernel over that region of the image, multiply element-wise, and sum. This produces one scalar output value for that position. Doing this for all valid positions produces a 2D **feature map** (or activation map).

### A Technical Note: Cross-Correlation vs. Convolution

Mathematically, true convolution requires flipping the kernel before sliding it (rotating it 180 degrees). What I just described — and what deep learning frameworks actually implement — is **cross-correlation**. The entire field calls it "convolution" anyway. This is harmless because the kernel weights are learned: if the true optimal kernel is the flipped version, the network will just learn the flipped weights. But you should know the terminology discrepancy exists, especially if you read signal processing literature.

> **Reading:** [DLBook §9.1](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) covers the mathematical definition rigorously, including the distinction between convolution and cross-correlation.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Convolution Sliding Animation

The following animation shows how a kernel slides across a 2D input to produce a feature map:
""")
    return


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/ConvolutionSliding.gif")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### What Filters Detect

A $3 \times 3$ kernel like:

$$K = \begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix}$$

is a horizontal edge detector — it responds strongly wherever the image transitions from dark (top) to bright (bottom). Rotate it 90 degrees and you get a vertical edge detector. A Gaussian kernel blurs the image. In a CNN, **we don't hand-design these filters — the network learns them through backpropagation**. This is the whole point: the network discovers which local patterns are useful for the task at hand.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 3. CNN Building Blocks

### Filters and Feature Maps

A convolutional layer applies $F$ different filters to the input, producing $F$ feature maps. Each filter is a small 3D tensor of learnable weights — small in height and width (say $3 \times 3$) but extending through the full depth of the input. For an RGB input with 3 channels, each filter has shape $3 \times 3 \times 3 = 27$ weights plus one bias.

Each filter specializes in detecting a particular pattern. One might respond to horizontal edges, another to vertical edges, another to specific color gradients. The collection of all $F$ feature maps becomes the input to the next layer — now with spatial dimensions slightly reduced and depth equal to $F$.

The parameter savings are enormous. A convolutional layer with 32 filters of size $3 \times 3$ on an RGB input has $32 \times (3 \times 3 \times 3 + 1) = 896$ parameters. Compare that to the 200 million we computed for the fully connected approach. Three orders of magnitude less, and it works better.

### Stride

The **stride** controls how many pixels the filter moves between applications. Stride 1 means the filter moves one pixel at a time — you get a feature map almost the same size as the input. Stride 2 means the filter jumps two pixels, halving the spatial dimensions. Stride is a powerful tool for downsampling.

### Padding

Without padding, convolution shrinks the spatial dimensions: a $3 \times 3$ filter on a $32 \times 32$ input produces a $30 \times 30$ output. After several layers, your feature maps shrink to nothing.

**Valid padding** (no padding): output is smaller than input. Pixels at the border are underrepresented.

**Same padding**: pad the input with zeros so that the output has the same spatial dimensions as the input. For a $3 \times 3$ kernel, this means adding 1 pixel of zeros on each side. This is the most common choice.

### Receptive Field

Each neuron in a convolutional layer only "sees" a small patch of the input — the $3 \times 3$ or $5 \times 5$ region its filter covers. This is its **receptive field**. But here's the crucial insight: a neuron in the *second* convolutional layer sees a $3 \times 3$ patch of the first layer's feature map, which itself covers $3 \times 3$ patches of the input. So the second-layer neuron effectively sees a $5 \times 5$ patch of the original input. The receptive field grows with depth. By stacking many convolutional layers, neurons in deeper layers respond to progressively larger regions of the input — from edges to textures to entire objects. This is how CNNs build a spatial hierarchy without ever using fully connected layers.

> **Reading:** [DLBook §9.3-9.5](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) covers stride, padding, and the mechanics of parameter sharing.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 4. Pooling Layers

Pooling layers downsample feature maps, reducing spatial dimensions while retaining the most important information.

**Max pooling** with a $2 \times 2$ window and stride 2: divide the feature map into non-overlapping $2 \times 2$ blocks, and keep the maximum value in each block. The spatial dimensions are halved. The intuition is that if a feature was detected somewhere in a local region, we don't care exactly where — just that it was detected.

**Average pooling** takes the mean instead of the maximum. Less common in hidden layers but important at the end of the network (see below).

Why pool at all?

1. **Dimensionality reduction:** Halving height and width cuts the number of values by 4x, reducing computation in subsequent layers.
2. **Approximate translation invariance:** If a feature shifts by one pixel, the max in a $2 \times 2$ window often stays the same. The network becomes less sensitive to small spatial shifts.
3. **Increasing effective receptive field:** By shrinking the spatial dimensions, subsequent convolutional layers cover proportionally more of the original image.

### Modern Alternatives to Pooling

**Strided convolution:** Instead of a separate pooling layer, use a convolutional layer with stride 2. This learns how to downsample rather than using a fixed rule. Many modern architectures prefer this approach — it's more flexible, and the extra parameters are minimal.

**Global Average Pooling (GAP):** At the very end of the network, instead of flattening the feature map into a huge vector and feeding it to a fully connected layer, take the spatial average of each feature map. If the last convolutional layer produces 512 feature maps of size $7 \times 7$, GAP gives you a vector of 512 values. This dramatically reduces the parameter count of the classifier head and acts as a structural regularizer.

> **Reading:** [DLBook §9.3](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) discusses pooling in detail, including its role as an invariance mechanism.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 5. The Standard CNN Architecture Pattern

Now we assemble the pieces. The classic CNN follows a repeating pattern:

$$\textbf{Conv} \to \textbf{ReLU} \to \textbf{Pool} \to \textbf{Conv} \to \textbf{ReLU} \to \textbf{Pool} \to \cdots \to \textbf{Flatten/GAP} \to \textbf{FC} \to \textbf{Output}$$

Two things happen systematically as you go deeper:

1. **Spatial dimensions decrease.** Each pooling (or strided convolution) halves height and width. An input of $224 \times 224$ might become $112 \to 56 \to 28 \to 14 \to 7$.
2. **Depth (channels) increases.** Typical progression: $3 \to 64 \to 128 \to 256 \to 512$. More channels means the network can represent more types of features.

This creates a **hierarchy of features:**

- **Early layers** detect low-level features: edges, corners, color gradients. These are universal — the same edges appear in photos of cats, cars, and X-rays.
- **Middle layers** compose these into mid-level features: textures, parts of objects, repeated patterns.
- **Deep layers** detect high-level, semantic features: faces, wheels, handwritten digits. These are task-specific.

This hierarchical decomposition is not something we engineered — it emerges from training. We constrain the architecture to be local and hierarchical, and the optimization process discovers the right features at each level.

Here's a minimal CNN in PyTorch to make this concrete:
""")
    return


@app.cell
def _():
    import torch
    import torch.nn as nn

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1: 3 channels -> 32 feature maps
                nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32x32 -> 32x32
                nn.ReLU(),
                nn.MaxPool2d(2),                              # 32x32 -> 16x16

                # Block 2: 32 -> 64 feature maps
                nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16x16 -> 16x16
                nn.ReLU(),
                nn.MaxPool2d(2),                              # 16x16 -> 8x8

                # Block 3: 64 -> 128 feature maps
                nn.Conv2d(64, 128, kernel_size=3, padding=1), # 8x8 -> 8x8
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),                      # 8x8 -> 1x1 (GAP)
            )
            self.classifier = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.features(x)      # [B, 128, 1, 1]
            x = x.view(x.size(0), -1) # [B, 128]
            x = self.classifier(x)    # [B, num_classes]
            return x

    model = SimpleCNN(num_classes=10)
    # Count parameters:
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")  # ~111K — compare to 200M for FC!
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
Notice the progression: spatial dimensions shrink ($32 \to 16 \to 8 \to 1$), channel depth grows ($3 \to 32 \to 64 \to 128$), and global average pooling at the end collapses the spatial dimensions entirely. The final linear layer maps from 128 features to the number of classes. The entire network has roughly 111K parameters — five orders of magnitude fewer than the naive fully connected approach.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 6. Landmark Architectures

Each of these architectures taught the field something fundamental. Understanding them in sequence gives you a sense of how the field discovered what works and why.

### LeNet-5 (LeCun et al., 1998)

The original CNN. Yann LeCun designed it for handwritten digit recognition (the MNIST dataset), and it was deployed commercially for reading checks. The architecture was modest: two convolutional layers, two pooling layers, three fully connected layers, about 60K parameters. But the core idea — learnable local filters with shared weights — was revolutionary. LeNet proved that the convolutional structure could learn useful features from raw pixels.

### AlexNet (Krizhevsky, Sutskever, Hinton, 2012)

This is the paper that changed everything. AlexNet won the 2012 ImageNet Large Scale Visual Recognition Challenge, cutting the top-5 error rate from 26% to 15.3% — a margin of victory so large it stunned the computer vision community. The architecture itself was straightforward (five conv layers, three FC layers), but it introduced several now-standard techniques:

- **ReLU activations** instead of sigmoid/tanh — faster training, reduced vanishing gradients
- **Dropout** for regularization
- **GPU training** — AlexNet was trained on two GTX 580 GPUs, demonstrating that deep learning was a GPU-driven enterprise
- **Data augmentation** (random crops, horizontal flips) to reduce overfitting

AlexNet didn't introduce fundamentally new theory. It showed that the old ideas (CNNs, backpropagation) worked spectacularly well when combined with more data, more compute, and a few practical tricks. This was the "ImageNet moment" that launched the deep learning era.

### VGGNet (Simonyan & Zisserman, 2014)

VGG's contribution was a systematic study of depth. The key finding: **use only $3 \times 3$ filters, and stack many of them**. Two stacked $3 \times 3$ conv layers have the same receptive field as a single $5 \times 5$ layer, but with fewer parameters and more nonlinearities (two ReLUs instead of one). VGG-16 (16 weight layers) and VGG-19 (19 layers) achieved excellent results on ImageNet, and VGG's clean, regular architecture made it a popular feature extractor for years.

The lesson: depth matters, and small filters stacked deep are better than large filters stacked shallow.

### GoogLeNet / Inception (Szegedy et al., 2014)

Inception asked: why choose one filter size when you can use several in parallel? The **Inception module** applies $1 \times 1$, $3 \times 3$, and $5 \times 5$ convolutions simultaneously and concatenates the results. The $1 \times 1$ convolutions serve as bottleneck layers, reducing channel depth before the expensive $3 \times 3$ and $5 \times 5$ operations.

GoogLeNet was 22 layers deep but used far fewer parameters than VGG thanks to these bottlenecks and the use of global average pooling instead of fully connected layers at the end. It showed that architectural ingenuity could achieve better results with fewer parameters.

### ResNet (He et al., 2015)

This is the most important architecture on this list. You need to understand it deeply, because residual connections appear in virtually every modern architecture, including transformers.

**The degradation problem:** You might expect that stacking more layers always helps — a deeper network can represent everything a shallower one can, plus more. But in practice, networks deeper than about 20 layers performed *worse* than shallower ones, not just on test data (overfitting) but on *training data*. This isn't overfitting — it's an optimization problem. Very deep networks are harder to train.

**The residual learning idea:** Instead of learning a desired mapping $h(x)$ directly, learn the *residual* $f(x) = h(x) - x$. The output of a residual block is:

$$\text{output} = f(x) + x$$

where $f(x)$ is two or three convolutional layers. This is the **skip connection** (or shortcut connection).

Why does this work? Consider the identity mapping: if the optimal transformation at some layer is to do nothing (just pass the input through), a standard network has to learn weights that implement the identity — which is surprisingly hard to optimize. A residual network just has to learn $f(x) = 0$, which is trivial (just set all weights to zero). The skip connection makes identity the *default behavior*, and the convolutional layers only need to learn the *deviation* from identity.

This seemingly small change enabled training networks with 100, 152, even 1000+ layers. ResNet-152 won ImageNet 2015 with a top-5 error of 3.57% — surpassing human performance on the benchmark.

The residual connection has become a universal design principle. Transformers use them. Dense prediction networks use them. Diffusion models use them. Whenever you see $\text{output} = \text{layer}(x) + x$ in a modern architecture, that's the ResNet idea.
""")
    return


@app.cell
def _():
    import torch as _torch
    import torch.nn as _nn

    class ResidualBlock(_nn.Module):
        """A basic residual block with two conv layers."""
        def __init__(self, channels):
            super().__init__()
            self.conv1 = _nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = _nn.BatchNorm2d(channels)
            self.conv2 = _nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = _nn.BatchNorm2d(channels)

        def forward(self, x):
            residual = x                        # save the input
            out = _torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + residual                 # skip connection
            out = _torch.relu(out)
            return out

    block = ResidualBlock(64)
    print(f"ResidualBlock parameters: {sum(p.numel() for p in block.parameters()):,}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Brief Mentions

- **DenseNet (2017):** Takes skip connections further — every layer is connected to every subsequent layer. Feature reuse is maximized, but memory usage is high.
- **EfficientNet (2019):** Introduced compound scaling — systematically scaling depth, width, and resolution together. Showed that balanced scaling beats simply making one dimension larger.
- **ConvNeXt (2022):** "Modernized" the ResNet architecture using design lessons learned from transformers (larger kernels, LayerNorm, fewer activation functions). Showed that pure ConvNets can compete with vision transformers when designed carefully.

> **Reading:** [DLBook §9.10](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) discusses architectural design considerations. For the original papers, the ResNet paper (He et al., "Deep Residual Learning for Image Recognition," 2015) is the most important single paper to read.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 7. Transfer Learning

This is the most important practical technique in this entire lecture. If you apply CNNs in the real world, you will almost certainly use transfer learning rather than training from scratch.

The idea is simple: take a CNN that has been trained on a large dataset (typically ImageNet, with 1.2 million labeled images across 1000 categories), remove the final classification layer, and use the rest of the network as a **feature extractor** for your task. The early and middle layers have learned general-purpose visual features (edges, textures, parts) that transfer to virtually any visual task.

### Two Strategies

**Feature extraction:** Freeze all the pretrained layers. Add a new classification head (one or two fully connected layers) on top. Train only the new head on your data. This is fast, works with small datasets, and is your default starting point.
""")
    return


@app.cell
def _():
    import torch.nn as _nn2
    import torchvision.models as models

    # Load pretrained ResNet-18
    model_pretrained = models.resnet18(weights='IMAGENET1K_V1')

    # Freeze all layers
    for param in model_pretrained.parameters():
        param.requires_grad = False

    # Replace the final FC layer (originally 512 -> 1000)
    num_your_classes = 10
    model_pretrained.fc = _nn2.Linear(512, num_your_classes)
    # Only model.fc will be trained

    print("Trainable parameters:", sum(p.numel() for p in model_pretrained.parameters() if p.requires_grad))
    print("Total parameters:", sum(p.numel() for p in model_pretrained.parameters()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
**Fine-tuning:** Start from pretrained weights, but unfreeze some or all layers and train the entire network with a small learning rate. The pretrained weights provide a good initialization, and fine-tuning adapts them to your specific domain.

### When to Use Which

The decision depends on two factors: **dataset size** and **domain similarity**.

| | Similar to ImageNet | Different from ImageNet |
|---|---|---|
| **Large dataset** | Fine-tune the whole network | Fine-tune the whole network (may need to fine-tune more aggressively) |
| **Small dataset** | Feature extraction (or fine-tune only later layers) | Feature extraction from early layers; this is the hardest case |

Medical imaging, satellite imagery, and microscopy are examples of domains different from ImageNet. Even in these cases, transfer learning from ImageNet features almost always outperforms training from scratch — the low-level edge and texture detectors are universal enough to help.

Transfer learning is why CNNs are practical. Without it, you'd need millions of labeled images for every new task. With it, you can get good results from hundreds or thousands of examples.

> **Reading:** [Murphy PML2, Ch 19](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) covers transfer learning and fine-tuning in detail. [DLBook §15.2](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) discusses the theoretical basis of transfer and multi-task learning.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 8. Visualizing What CNNs Learn

One common criticism of deep learning is that the models are "black boxes." For CNNs, this is less true than people think. Several visualization techniques give us direct insight into what these networks learn.

### Filter Visualization

The first convolutional layer operates directly on pixel values, so its filters are interpretable as small images. When you visualize the learned $3 \times 3 \times 3$ filters of the first layer, you invariably find edge detectors at various orientations, color contrast detectors, and simple texture detectors. This holds across datasets and tasks — the first layer learns a bank of Gabor-like filters, matching what neuroscientists have found in the primary visual cortex (area V1).

### Feature Map Visualization

Feed an image through the network and examine the activations (feature maps) at each layer. Early layers produce feature maps that are recognizably edge-like. Middle layers respond to textures and patterns. Deep layers produce feature maps that are semantically meaningful but spatially coarse — a "face detector" feature map will light up wherever faces appear.

### Gradient-Based Saliency Maps

Compute the gradient of the output class score with respect to the input pixels: $\frac{\partial y_c}{\partial x}$. The magnitude of this gradient tells you which pixels, if changed, would most affect the network's prediction. This produces a "saliency map" highlighting the regions the network is paying attention to.

### Grad-CAM (Gradient-weighted Class Activation Maps)

A more refined technique: compute the gradient of the class score with respect to the feature maps of the *last* convolutional layer, use these gradients as weights, and take a weighted combination of the feature maps. The result is a coarse heatmap overlaid on the input image showing which regions were most important for the prediction. Grad-CAM is widely used for model interpretability and debugging — if the network is classifying "dog" based on the background grass rather than the actual dog, Grad-CAM will reveal that.
""")
    return


@app.cell
def _():
    # Conceptual Grad-CAM sketch (not production-ready)
    # 1. Forward pass, recording final conv layer activations
    # 2. Backward pass for target class
    # 3. Global average pool the gradients over spatial dims
    # 4. Weight the feature maps by these pooled gradients
    # 5. ReLU the result (we only want positive contributions)
    # 6. Upsample to input resolution and overlay on image
    print("Grad-CAM steps:")
    print("1. Forward pass -> record final conv layer activations")
    print("2. Backward pass for target class")
    print("3. Global average pool the gradients over spatial dims")
    print("4. Weight feature maps by pooled gradients")
    print("5. ReLU (keep only positive contributions)")
    print("6. Upsample to input resolution and overlay")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
> **Reading:** [DLBook §9.11](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) discusses representational interpretations. For Grad-CAM specifically, the original paper (Selvaraju et al., 2017) is accessible and well-written.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 9. Beyond Image Classification

CNNs were developed for classification, but the convolutional backbone is the foundation for almost every computer vision task. Here's a brief tour of what's downstream.

### Object Detection

Classification tells you *what* is in the image. Detection tells you *what* and *where* — producing bounding boxes around each object along with class labels.

Two paradigms:
- **Two-stage detectors** (R-CNN family): First propose regions that might contain objects, then classify each region. More accurate, slower.
- **Single-stage detectors** (YOLO, SSD): Predict bounding boxes and classes in one forward pass. Faster, suitable for real-time applications.

Both use CNN backbones (often pretrained on ImageNet) as their feature extractors.

### Semantic Segmentation

Segmentation assigns a class label to *every pixel* in the image. Instead of one label per image, you get a dense prediction map.

The **U-Net** architecture (Ronneberger et al., 2015) is the most influential design here. It has an **encoder** (standard CNN that downsamples) and a **decoder** (upsampling path that recovers spatial resolution), with skip connections between corresponding encoder and decoder layers. The skip connections allow the decoder to recover fine spatial details that would otherwise be lost during downsampling. U-Net was originally designed for medical image segmentation and remains the go-to architecture for dense prediction tasks.

### Image Generation

CNNs also appear in generative models — running "in reverse" to produce images from noise or latent codes. We'll cover this properly in a later module, but know that the convolutional structure is just as important for generation as it is for recognition.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 10. Putting It All Together: A Complete Training Example

Here's a complete, runnable example that trains a CNN on CIFAR-10 using transfer learning. This captures the workflow you'll use in practice:
""")
    return


@app.cell
def _():
    import torch as _t
    import torch.nn as _n
    import torch.optim as _o
    from torchvision import datasets, transforms, models as _models

    # ---- Data pipeline ----
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(224),           # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),   # ImageNet stats
                             (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    # Note: In practice you would create DataLoaders and run the training loop:
    # train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # test_set  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    # test_loader  = DataLoader(test_set, batch_size=64, shuffle=False)

    # ---- Model: fine-tune a pretrained ResNet-18 ----
    _model_ft = _models.resnet18(weights='IMAGENET1K_V1')
    _model_ft.fc = _n.Linear(_model_ft.fc.in_features, 10)  # 10 CIFAR classes

    print("Model ready for fine-tuning on CIFAR-10")
    print(f"Total parameters: {sum(p.numel() for p in _model_ft.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in _model_ft.parameters() if p.requires_grad):,}")

    # Training loop would be:
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # for epoch in range(5):
    #     model.train()
    #     for images, labels in train_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
Notice the key components: data augmentation (random crops and flips — a form of regularization), ImageNet normalization statistics, pretrained weights, and a replaced classification head. Even with just 5 epochs, you should see test accuracy above 90% on CIFAR-10 — far better than training from scratch with the same budget.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## Practice

### Conceptual Questions

1. Compute the number of parameters in a convolutional layer with 64 input channels, 128 output filters, $3 \times 3$ kernel size, and bias terms. Compare this to a fully connected layer connecting $64 \times 32 \times 32$ inputs to 128 outputs.

2. What is the receptive field of a neuron in the third convolutional layer if all three layers use $3 \times 3$ kernels with stride 1 and no pooling? What if there's a $2 \times 2$ max pool after the first layer?

3. Explain why a residual block can represent the identity mapping more easily than a plain block. Why does this matter for very deep networks?

4. You have 500 labeled images of a rare skin condition. You want to build a classifier. Describe your strategy, justifying each choice (architecture, pretrained weights, frozen vs. fine-tuned layers, data augmentation).

### Implementation Exercises

5. Build a CNN from scratch (no pretrained weights) for CIFAR-10. Start with the `SimpleCNN` architecture from Section 5. Experiment with: (a) adding batch normalization, (b) replacing max pooling with stride-2 convolution, (c) adding residual connections. Report test accuracy for each variant.

6. Implement Grad-CAM for a pretrained ResNet-18. Feed in several ImageNet images and visualize the class activation maps. Do the highlighted regions make sense?

7. Compare training from scratch vs. fine-tuning a pretrained ResNet on a small dataset (use a subset of CIFAR-10 with only 100 images per class). Plot training and test accuracy curves for both. What do you observe about convergence speed and final accuracy?

> **Primary reference for all topics in this lecture:** [DLBook, Chapter 9: Convolutional Networks](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf). For the practical side, [Murphy PML1 §14.1-14.3](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) provides a clear, modern treatment with additional architectural details.
""")
    return


if __name__ == "__main__":
    app.run()

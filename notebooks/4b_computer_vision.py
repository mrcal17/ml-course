import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Path B --- Computer Vision

    Computer vision asks machines to derive meaningful information from images and video. It is one of the oldest subfields of AI and the one where deep learning first proved its dominance --- AlexNet's victory in the 2012 ImageNet challenge is often cited as the moment that reignited the entire field. This guide maps the modern landscape, from classical detection and segmentation to Vision Transformers and multimodal models.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. The CV Landscape: Three Eras

    **Hand-crafted features (1960s--2012).** Edge detectors (Sobel, Canny), SIFT, HOG, SURF --- all designed by humans to capture local image structure. Classifiers like SVMs sat on top of these features. Performance was limited and domain-specific.

    **CNNs (2012--2020).** AlexNet (2012) showed that end-to-end learned features crush hand-crafted ones on ImageNet. VGGNet went deeper. GoogLeNet introduced inception modules. ResNet (2015) solved the degradation problem with skip connections, enabling networks with 100+ layers. For about eight years, CNNs were the undisputed backbone of all vision systems. See [Goodfellow Ch. 9: Convolutional Networks](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the theoretical foundations.

    **Vision Transformers and hybrids (2020--present).** ViT (Dosovitskiy et al., 2020) demonstrated that pure transformers, applied to sequences of image patches, can match or exceed CNN performance --- given enough data. The field has since oscillated between transformer-based and CNN-based architectures, with each camp borrowing ideas from the other.

    Understanding all three eras matters. Many production systems still use CNNs (they are fast, well-understood, and data-efficient). Transformers dominate research benchmarks. The best practitioners can pick the right tool for the constraint set.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Code Demo: Hand-Crafted Edge Detection (Sobel Filter)

    The Sobel operator computes image gradients using two 3x3 kernels --- one for horizontal edges ($G_x$), one for vertical edges ($G_y$). The gradient magnitude is $G = \sqrt{G_x^2 + G_y^2}$.
    """)
    return


@app.cell
def _(np):
    # Sobel edge detection --- the "Era 1" workhorse
    # Sobel kernels approximate the image gradient in x and y
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)

    # Simulate a small 6x6 grayscale image with a vertical edge
    image = np.zeros((6, 6), dtype=np.float32)
    image[:, 3:] = 1.0  # bright right half

    print("Image:\n", image)
    print("\nSobel X kernel (detects vertical edges):\n", sobel_x)
    return (image, sobel_x, sobel_y)


@app.cell
def _(image, np, sobel_x, sobel_y):
    # Apply Sobel via manual 2D convolution (valid mode)
    def convolve2d(img, kernel):
        """Naive 2D convolution --- no padding."""
        kh, kw = kernel.shape
        oh, ow = img.shape[0] - kh + 1, img.shape[1] - kw + 1
        out = np.zeros((oh, ow))
        for i in range(oh):
            for j in range(ow):
                # Element-wise multiply patch by kernel, sum
                out[i, j] = np.sum(img[i:i+kh, j:j+kw] * kernel)
        return out

    gx = convolve2d(image, sobel_x)  # horizontal gradient
    gy = convolve2d(image, sobel_y)  # vertical gradient
    # Gradient magnitude: G = sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(gx**2 + gy**2)

    print("Gradient magnitude (edges):\n", np.round(magnitude, 2))
    return (convolve2d,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Object Detection

    Object detection = classification + localization. The model must identify *what* objects are in an image and *where* they are (bounding boxes).

    ### The Evolution

    **R-CNN** (Girshick et al., 2014): Use selective search to propose ~2000 region candidates. Run each through a CNN. Classify each region with SVMs. Painfully slow --- each image took ~47 seconds.

    **Fast R-CNN** (Girshick, 2015): Run the CNN *once* on the entire image to produce a feature map. Project region proposals onto this feature map. Use RoI pooling to extract fixed-size features per region. End-to-end training with a multi-task loss (classification + bounding box regression). ~20x faster.

    **Faster R-CNN** (Ren et al., 2015): Replace selective search with a *Region Proposal Network* (RPN) --- a small CNN that proposes regions directly from the feature map. Now the entire pipeline is a single neural network. This architecture introduced **anchor boxes**: pre-defined boxes at each spatial location with different aspect ratios and scales that serve as reference templates.

    **YOLO** (Redmon et al., 2016): "You Only Look Once." A completely different philosophy --- divide the image into a grid, predict bounding boxes and class probabilities directly from each cell in a single forward pass. Real-time detection (45+ FPS on a GPU). The YOLO family has gone through many iterations (YOLOv2 through YOLOv8+, now maintained by Ultralytics), each improving speed and accuracy. YOLO trades some accuracy for massive speed gains and remains the default choice for real-time applications.

    **DETR** (Carion et al., 2020): Detection Transformer. Treats object detection as a set prediction problem. Uses a CNN backbone to extract features, then a transformer encoder-decoder to predict a fixed set of detections. Hungarian matching loss for training. Eliminates the need for anchor boxes, NMS, and most hand-designed components. Elegant but initially slow to converge --- follow-up work (Deformable DETR, DINO) addressed this.

    ### Key Concepts

    - **Intersection over Union (IoU):** The ratio of the area of overlap between predicted and ground-truth boxes to the area of their union. An IoU threshold (commonly 0.5) determines whether a detection is a true positive.
    - **Non-Maximum Suppression (NMS):** When multiple overlapping boxes detect the same object, NMS keeps only the highest-confidence one. Simple but effective. DETR eliminates the need for this.
    - **Anchor boxes:** Pre-defined reference boxes at multiple scales and aspect ratios placed at each spatial location. The model predicts offsets relative to these anchors rather than absolute coordinates. Used by Faster R-CNN, SSD, and YOLO (through v5). Anchor-free methods (FCOS, CenterNet) predict locations directly and have become increasingly popular.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code Demo: IoU and Non-Maximum Suppression

    IoU between two boxes $A$ and $B$:

    $$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
    """)
    return


@app.cell
def _(np):
    def compute_iou(box_a, box_b):
        """IoU for two boxes [x1, y1, x2, y2]."""
        # Intersection coordinates
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        # Union = area_A + area_B - intersection
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    # Two overlapping boxes
    pred_box = [10, 10, 50, 50]
    gt_box   = [20, 20, 60, 60]
    print(f"IoU = {compute_iou(pred_box, gt_box):.4f}")  # expect ~0.189
    return (compute_iou,)


@app.cell
def _(compute_iou, np):
    def nms(boxes, scores, iou_threshold=0.5):
        """Non-Maximum Suppression: keep highest-confidence, discard overlaps."""
        order = np.argsort(scores)[::-1]  # sort by confidence (descending)
        keep = []
        while len(order) > 0:
            idx = order[0]
            keep.append(idx)
            # Compute IoU of this box with all remaining boxes
            remaining = order[1:]
            ious = np.array([compute_iou(boxes[idx], boxes[j]) for j in remaining])
            # Keep only boxes with IoU below threshold
            order = remaining[ious < iou_threshold]
        return keep

    # Example: 4 overlapping detections with different confidences
    boxes  = np.array([[10,10,50,50],[12,12,52,52],[14,14,54,54],[100,100,140,140]])
    scores = np.array([0.9, 0.75, 0.8, 0.6])
    kept = nms(boxes, scores, iou_threshold=0.5)
    print(f"Kept box indices after NMS: {kept}")  # should keep box 0 and box 3
    return (nms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code Demo: Anchor Box Generation

    Anchor boxes tile across spatial locations with multiple scales and aspect ratios.
    """)
    return


@app.cell
def _(np):
    def generate_anchors(feature_h, feature_w, scales, ratios, stride=16):
        """Generate anchor boxes for a feature map grid."""
        anchors = []
        for i in range(feature_h):
            for j in range(feature_w):
                cx, cy = j * stride + stride / 2, i * stride + stride / 2
                for s in scales:
                    for r in ratios:
                        # w * h = s^2, w/h = r  =>  w = s*sqrt(r), h = s/sqrt(r)
                        w = s * np.sqrt(r)
                        h = s / np.sqrt(r)
                        anchors.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        return np.array(anchors)

    # Small 3x3 feature map, 2 scales, 3 aspect ratios => 3*3*2*3 = 54 anchors
    anchors = generate_anchors(3, 3, scales=[64, 128], ratios=[0.5, 1.0, 2.0])
    print(f"Generated {len(anchors)} anchors for a 3x3 feature map")
    print(f"First anchor box: {anchors[0].astype(int)}")
    return (anchors,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Semantic Segmentation

    Semantic segmentation assigns a class label to *every pixel* in an image. Unlike detection, there are no bounding boxes --- the output is a dense label map the same size as the input.

    **Fully Convolutional Networks (FCN)** (Long et al., 2015): Replace the final fully connected layers of a classification CNN with convolutional layers. Upsample the low-resolution feature maps back to the input resolution using transposed convolutions. First end-to-end approach.

    **U-Net** (Ronneberger et al., 2015): An encoder-decoder architecture with **skip connections** between corresponding encoder and decoder layers. The encoder contracts (downsamples), the decoder expands (upsamples), and skip connections pass high-resolution features from early layers to later layers, preserving fine-grained spatial detail. Originally designed for biomedical image segmentation but has become a universal architecture --- it also forms the backbone of most diffusion models (callback to 3B --- Generative Models).

    **DeepLab** (Chen et al., 2017--2018): Introduced **dilated (atrous) convolutions** --- convolutions with gaps between kernel elements that expand the receptive field without reducing spatial resolution or increasing parameters. DeepLabv3+ combines atrous spatial pyramid pooling (ASPP) at multiple dilation rates with an encoder-decoder structure. Excels at capturing multi-scale context.

    **Key concepts:**
    - **Dilated convolutions:** A standard 3x3 convolution with dilation rate *d* has an effective receptive field of $(2d+1) \times (2d+1)$. This lets you see more context without downsampling.
    - **Skip connections in segmentation:** Unlike ResNet's skip connections (which add features), U-Net's skip connections *concatenate* encoder features with decoder features, giving the decoder access to both high-level semantics and low-level spatial details.
    - **Pixel-wise cross-entropy:** The standard loss function --- just cross-entropy computed independently at each pixel, then averaged. Class imbalance is common (background dominates), so weighted cross-entropy or focal loss is often used.

    See [Goodfellow Ch. 9](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the convolution operations underlying these architectures.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code Demo: Dilated (Atrous) Convolution

    A dilation rate $d$ inserts $d-1$ zeros between kernel elements, expanding the receptive field from $3 \times 3$ to $(2d+1) \times (2d+1)$ without adding parameters.
    """)
    return


@app.cell
def _(np):
    def dilated_conv2d(img, kernel, dilation=1):
        """2D convolution with dilation --- expands receptive field without more params."""
        kh, kw = kernel.shape
        # Effective kernel size with dilation
        eff_kh = kh + (kh - 1) * (dilation - 1)
        eff_kw = kw + (kw - 1) * (dilation - 1)
        oh, ow = img.shape[0] - eff_kh + 1, img.shape[1] - eff_kw + 1
        out = np.zeros((oh, ow))
        for i in range(oh):
            for j in range(ow):
                val = 0.0
                for ki in range(kh):
                    for kj in range(kw):
                        # Sample with dilation stride
                        val += img[i + ki * dilation, j + kj * dilation] * kernel[ki, kj]
                out[i, j] = val
        return out

    # Compare receptive fields: dilation=1 vs dilation=2
    test_img = rng.standard_normal((8, 8))
    kernel = np.ones((3, 3)) / 9  # averaging kernel
    out_d1 = dilated_conv2d(test_img, kernel, dilation=1)  # 3x3 receptive field
    out_d2 = dilated_conv2d(test_img, kernel, dilation=2)  # 5x5 receptive field
    print(f"Dilation=1 output shape: {out_d1.shape}")  # 6x6
    print(f"Dilation=2 output shape: {out_d2.shape}")  # 4x4 (larger RF, smaller output)
    return (dilated_conv2d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code Demo: Pixel-Wise Cross-Entropy Loss

    For segmentation, we compute cross-entropy at every pixel independently, then average:

    $$\mathcal{L} = -\frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}\sum_{c=1}^{C} y_{ijc} \log(\hat{y}_{ijc})$$
    """)
    return


@app.cell
def _(np):
    def pixel_cross_entropy(pred_logits, target_labels, class_weights=None):
        """Pixel-wise cross-entropy loss for segmentation.
        pred_logits: (H, W, C) raw scores per class
        target_labels: (H, W) integer class labels
        """
        H, W, C = pred_logits.shape
        # Softmax over class dimension
        exp_logits = np.exp(pred_logits - pred_logits.max(axis=2, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=2, keepdims=True)
        # Gather the probability of the correct class at each pixel
        loss = 0.0
        for i in range(H):
            for j in range(W):
                c = target_labels[i, j]
                w = class_weights[c] if class_weights is not None else 1.0
                loss -= w * np.log(probs[i, j, c] + 1e-8)
        return loss / (H * W)

    # Tiny 4x4 segmentation map with 3 classes
    pred = rng.standard_normal((4, 4, 3))
    labels = rng.integers(0, 3, (4, 4))
    loss = pixel_cross_entropy(pred, labels)
    print(f"Pixel-wise cross-entropy loss: {loss:.4f}")

    # With class weights (upweight rare class 2)
    loss_w = pixel_cross_entropy(pred, labels, class_weights=[1.0, 1.0, 5.0])
    print(f"Weighted cross-entropy loss:   {loss_w:.4f}")
    return (pixel_cross_entropy,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Instance and Panoptic Segmentation

    **Semantic segmentation** labels every pixel with a class but does not distinguish between individual instances of the same class (two adjacent people are both labeled "person").

    **Instance segmentation** detects each object instance and produces a pixel-level mask for each. **Mask R-CNN** (He et al., 2017) extends Faster R-CNN by adding a mask prediction branch --- for each detected region, it predicts a binary mask in parallel with the class and box. Uses RoI Align (bilinear interpolation instead of quantized RoI pooling) for pixel-accurate alignment.

    **Panoptic segmentation** unifies semantic and instance segmentation: every pixel gets a class label (semantic) *and* an instance ID for countable objects (instance). This gives a complete scene understanding --- both "stuff" (sky, road, grass) and "things" (cars, people, dogs).

    **Segment Anything Model (SAM)** (Kirillov et al., 2023): A foundation model for segmentation. Trained on 1 billion masks from 11 million images. Given any prompt (point, box, text), it produces high-quality segmentation masks. SAM is a promptable segmentation model --- it generalizes to unseen objects and domains without fine-tuning. This is the "GPT-3 moment" for segmentation. SAM 2 extends this to video.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Code Demo: RoI Align vs RoI Pooling

    RoI Pooling quantizes coordinates (lossy). RoI Align uses bilinear interpolation for sub-pixel accuracy --- crucial for mask prediction.
    """)
    return


@app.cell
def _(np):
    def roi_pool(feature_map, roi, output_size=(2, 2)):
        """Simplified RoI Pooling: quantize, then max-pool each bin."""
        x1, y1, x2, y2 = [int(round(c)) for c in roi]  # quantize!
        roi_feat = feature_map[y1:y2, x1:x2]
        h, w = roi_feat.shape
        bh, bw = h // output_size[0], w // output_size[1]
        out = np.zeros(output_size)
        for i in range(output_size[0]):
            for j in range(output_size[1]):
                out[i, j] = roi_feat[i*bh:(i+1)*bh, j*bw:(j+1)*bw].max()
        return out

    def bilinear_sample(feat, y, x):
        """Bilinear interpolation at sub-pixel (y, x) --- key to RoI Align."""
        y0, x0 = int(np.floor(y)), int(np.floor(x))
        y1, x1 = min(y0 + 1, feat.shape[0] - 1), min(x0 + 1, feat.shape[1] - 1)
        dy, dx = y - y0, x - x0
        return (feat[y0, x0] * (1-dy)*(1-dx) + feat[y1, x0] * dy*(1-dx) +
                feat[y0, x1] * (1-dy)*dx + feat[y1, x1] * dy*dx)

    # Simulated 8x8 feature map
    feat = np.arange(64).reshape(8, 8).astype(float)
    roi = [1.3, 1.7, 5.8, 5.2]  # non-integer box coordinates
    pooled = roi_pool(feat, roi)
    # RoI Align would sample at exact sub-pixel locations using bilinear_sample
    print(f"RoI Pooled (quantized):\n{pooled}")
    print(f"\nBilinear sample at (1.7, 1.3): {bilinear_sample(feat, 1.7, 1.3):.2f}")
    return (bilinear_sample,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Vision Transformers (ViT)

    **ViT** (Dosovitskiy et al., 2020) applies the transformer architecture directly to images:

    1. Split the image into fixed-size patches (typically 16x16 pixels).
    2. Flatten each patch and project it through a linear layer to get a patch embedding.
    3. Add learnable position embeddings.
    4. Prepend a learnable `[CLS]` token.
    5. Feed the sequence of patch embeddings through a standard transformer encoder.
    6. Use the `[CLS]` token's output for classification.

    **Why ViTs need more data than CNNs:** CNNs have strong **inductive biases** built into their architecture --- translation equivariance (a filter detects the same feature anywhere in the image) and locality (each layer only looks at a small spatial neighborhood). ViTs lack these biases. Self-attention is global from the first layer, and there is no built-in notion that nearby pixels are more related than distant ones. This means ViTs must *learn* these spatial relationships from data, which requires much larger datasets (ImageNet-21k or JFT-300M, not just ImageNet-1k).

    ### Key Variants

    - **DeiT** (Touvron et al., 2021): "Data-efficient Image Transformers." Showed that with strong data augmentation, regularization, and knowledge distillation from a CNN teacher, ViTs can be trained effectively on ImageNet-1k alone. Made ViTs practical for researchers without Google-scale compute.
    - **Swin Transformer** (Liu et al., 2021): Introduces **shifted windows** --- self-attention is computed within local windows, and windows shift between layers to enable cross-window connections. This creates a hierarchical feature map (like a CNN) and reduces the quadratic cost of global self-attention to linear in image size. Became the dominant backbone for detection and segmentation.
    - **ConvNeXt** (Liu et al., 2022): "A ConvNet for the 2020s." Takes a standard ResNet and modernizes it with design choices borrowed from transformers --- larger kernels (7x7), fewer activation functions, LayerNorm instead of BatchNorm, inverted bottleneck blocks. The result matches or exceeds Swin Transformer performance. The takeaway: the architectural innovations of transformers matter more than self-attention itself.

    See [Goodfellow Ch. 9](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) and [Murphy PML2](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) for the theoretical backdrop connecting CNNs and attention mechanisms.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code Demo: ViT Patch Embedding

    Steps 1--4 of ViT: split an image into patches, flatten, project, add position embeddings and a [CLS] token.
    """)
    return


@app.cell
def _(np):
    def vit_patch_embed(image, patch_size=4, embed_dim=8):
        """ViT patch embedding: image (H, W, C) -> (N+1, D) token sequence."""
        H, W, C = image.shape
        num_patches = (H // patch_size) * (W // patch_size)
        # Step 1-2: Extract and flatten patches
        patches = []
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                patch = image[i:i+patch_size, j:j+patch_size, :].flatten()
                patches.append(patch)
        patches = np.array(patches)  # (N, patch_size^2 * C)
        # Step 2: Linear projection (random weights for demo)
        proj = rng.standard_normal((patches.shape[1], embed_dim)) * 0.02
        embeddings = patches @ proj  # (N, embed_dim)
        # Step 3: Add learnable position embeddings
        pos_embed = rng.standard_normal((num_patches + 1, embed_dim)) * 0.02
        # Step 4: Prepend [CLS] token
        cls_token = np.zeros((1, embed_dim))
        tokens = np.vstack([cls_token, embeddings])  # (N+1, embed_dim)
        tokens = tokens + pos_embed
        return tokens

    # 8x8 RGB image -> 4 patches of 4x4 -> 5 tokens (4 patches + CLS)
    dummy_img = rng.standard_normal((8, 8, 3))
    tokens = vit_patch_embed(dummy_img, patch_size=4, embed_dim=8)
    print(f"Image shape: {dummy_img.shape}")
    print(f"Token sequence shape: {tokens.shape}")  # (5, 8)
    print(f"[CLS] token (index 0): {tokens[0].round(3)}")
    return (vit_patch_embed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code Demo: Self-Attention on Patch Tokens

    After patch embedding, ViT applies standard multi-head self-attention. Here is single-head attention:

    $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
    """)
    return


@app.cell
def _(np):
    def self_attention(tokens, d_k=8):
        """Single-head self-attention over token sequence (N, D)."""
        N, D = tokens.shape
        # Learnable projection matrices (random for demo)
        W_q = rng.standard_normal((D, d_k)) * 0.1
        W_k = rng.standard_normal((D, d_k)) * 0.1
        W_v = rng.standard_normal((D, d_k)) * 0.1
        Q, K, V = tokens @ W_q, tokens @ W_k, tokens @ W_v
        # Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V
        scores = (Q @ K.T) / np.sqrt(d_k)
        attn_weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn_weights /= attn_weights.sum(axis=-1, keepdims=True)
        output = attn_weights @ V  # (N, d_k)
        return output, attn_weights

    demo_tokens = rng.standard_normal((5, 8))  # 5 tokens, dim 8
    out, weights = self_attention(demo_tokens, d_k=8)
    print(f"Attention output shape: {out.shape}")
    print(f"CLS token attends to patches with weights: {weights[0].round(3)}")
    return (self_attention,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. Multimodal Vision-Language Models

    The most exciting recent development in CV is the convergence with NLP through multimodal models:

    **CLIP** (Radford et al., 2021): Contrastive Language-Image Pre-training. Trains an image encoder and a text encoder jointly on 400M image-text pairs scraped from the internet. The training objective is simple: maximize the cosine similarity between matching image-text pairs and minimize it for non-matching pairs. The result is a shared embedding space where images and text can be directly compared. CLIP enables zero-shot image classification --- describe any category in text, and CLIP matches images to it without any task-specific training.

    **DALL-E, Stable Diffusion, and text-to-image generation:** These models generate images from text descriptions. Stable Diffusion (Rombach et al., 2022) uses a U-Net in a latent diffusion framework, conditioned on CLIP text embeddings. This is covered in depth in 3B --- Generative Models if you have taken that path.

    **GPT-4V, LLaVA, and vision-language LLMs:** Multimodal LLMs that can reason about images. Typically, a vision encoder (often a ViT variant) produces image embeddings that are projected into the LLM's embedding space, allowing the language model to "see" and reason about visual content alongside text.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code Demo: CLIP-Style Contrastive Loss

    CLIP trains by maximizing cosine similarity of matching (image, text) pairs along the diagonal and minimizing off-diagonal pairs. The loss is a symmetric cross-entropy over the similarity matrix.
    """)
    return


@app.cell
def _(np):
    def clip_contrastive_loss(image_embeds, text_embeds, temperature=0.07):
        """CLIP contrastive loss on a batch of (image, text) pairs."""
        # L2 normalize embeddings
        image_embeds = image_embeds / np.linalg.norm(image_embeds, axis=1, keepdims=True)
        text_embeds = text_embeds / np.linalg.norm(text_embeds, axis=1, keepdims=True)
        # Cosine similarity matrix scaled by temperature
        logits = (image_embeds @ text_embeds.T) / temperature  # (B, B)
        # Labels: matching pairs are on the diagonal
        B = logits.shape[0]
        labels = np.arange(B)
        # Symmetric cross-entropy: image->text + text->image
        def cross_entropy(logits, labels):
            exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)
            return -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-8))
        loss = (cross_entropy(logits, labels) + cross_entropy(logits.T, labels)) / 2
        return loss, logits * temperature  # return similarity matrix too

    # Batch of 4 image-text pairs (embed_dim=16)
    img_emb = rng.standard_normal((4, 16))
    txt_emb = img_emb + rng.standard_normal((4, 16)) * 0.1  # text similar to matching image
    loss, sim = clip_contrastive_loss(img_emb, txt_emb)
    print(f"CLIP loss: {loss:.4f}")
    print(f"Similarity matrix (should be high on diagonal):\n{sim.round(2)}")
    return (clip_contrastive_loss,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. 3D Vision (Brief)

    A rapidly growing area that extends CV beyond 2D images:

    - **Neural Radiance Fields (NeRFs)** (Mildenhall et al., 2020): Represent a 3D scene as a continuous function mapping 3D coordinates + viewing direction to color + density. Trained from a set of 2D images with known camera poses. Produces stunning novel view synthesis. Subsequent work (Instant-NGP, 3D Gaussian Splatting) dramatically improved training and rendering speed.
    - **Monocular depth estimation:** Predict a depth map from a single RGB image. Models like MiDaS and Depth Anything use large-scale pre-training to achieve impressive zero-shot depth estimation.
    - **Point clouds:** 3D data represented as unordered sets of points. PointNet (Qi et al., 2017) processes point clouds directly using shared MLPs and global pooling, respecting the permutation invariance of point sets. Important for autonomous driving (LiDAR data) and robotics.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code Demo: NeRF-Style Positional Encoding

    NeRF maps low-dimensional coordinates to a higher-dimensional space using sinusoidal positional encoding before passing to the MLP. This helps the network learn high-frequency detail:

    $$\gamma(p) = \left[\sin(2^0 \pi p),\; \cos(2^0 \pi p),\; \ldots,\; \sin(2^{L-1} \pi p),\; \cos(2^{L-1} \pi p)\right]$$
    """)
    return


@app.cell
def _(np):
    def nerf_positional_encoding(coords, num_frequencies=4):
        """NeRF positional encoding: maps (N, D) -> (N, D * 2 * L)."""
        encodings = []
        for freq in range(num_frequencies):
            # sin(2^freq * pi * x), cos(2^freq * pi * x)
            encodings.append(np.sin((2.0 ** freq) * np.pi * coords))
            encodings.append(np.cos((2.0 ** freq) * np.pi * coords))
        return np.concatenate(encodings, axis=-1)

    # Encode 3D coordinates (x, y, z) with 4 frequency bands
    points_3d = np.array([[0.1, 0.5, 0.9], [-0.3, 0.0, 0.7]])
    encoded = nerf_positional_encoding(points_3d, num_frequencies=4)
    print(f"Input shape: {points_3d.shape}")        # (2, 3)
    print(f"Encoded shape: {encoded.shape}")          # (2, 24) = 3 * 2 * 4
    return (nerf_positional_encoding,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 8. Video Understanding (Brief)

    Video adds a temporal dimension to vision:

    - **Temporal modeling approaches:** 3D convolutions (C3D, I3D) extend 2D convolutions along the time axis. Two-stream networks process appearance (RGB) and motion (optical flow) separately. Video transformers (TimeSformer, ViViT) apply attention across spatial and temporal dimensions.
    - **Action recognition:** Classify what action is being performed in a video clip. Kinetics-400/700 is the standard benchmark. Modern approaches use video foundation models pre-trained on large-scale data.
    - **Video object segmentation and tracking:** Follow objects across frames. SAM 2 extends the Segment Anything model to video with memory-based temporal propagation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Code Demo: 3D Convolution for Video

    A 3D convolution extends the 2D kernel along the temporal axis, processing spatial *and* temporal neighborhoods simultaneously.
    """)
    return


@app.cell
def _(np):
    def conv3d_naive(video, kernel):
        """Naive 3D convolution over a (T, H, W) video volume."""
        kt, kh, kw = kernel.shape
        T, H, W = video.shape
        ot, oh, ow = T - kt + 1, H - kh + 1, W - kw + 1
        out = np.zeros((ot, oh, ow))
        for t in range(ot):
            for i in range(oh):
                for j in range(ow):
                    out[t, i, j] = np.sum(
                        video[t:t+kt, i:i+kh, j:j+kw] * kernel
                    )
        return out

    # 5-frame, 6x6 video with a moving bright spot
    video = np.zeros((5, 6, 6))
    for t in range(5):
        video[t, 2, t] = 1.0  # spot moves right over time
    kernel_3d = np.ones((3, 3, 3)) / 27  # 3x3x3 averaging kernel

    result = conv3d_naive(video, kernel_3d)
    print(f"Video shape: {video.shape}, Kernel: {kernel_3d.shape}")
    print(f"Output shape: {result.shape}")  # (3, 4, 4)
    return (conv3d_naive,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 9. Evaluation Metrics

    Different tasks require different metrics:

    **Object Detection:**
    - **mAP (mean Average Precision):** The standard metric. Compute the precision-recall curve for each class at a given IoU threshold (mAP@0.5 uses IoU > 0.5). Average the area under this curve across all classes. COCO uses mAP averaged over multiple IoU thresholds from 0.5 to 0.95 in steps of 0.05 (denoted mAP@[0.5:0.95]).

    **Segmentation:**
    - **mIoU (mean Intersection over Union):** For each class, compute the IoU between predicted and ground-truth pixel masks, then average across classes. The standard metric for semantic segmentation.
    - **AP (Average Precision) for masks:** Instance segmentation uses the same mAP framework as detection, but IoU is computed on masks rather than bounding boxes.

    **Image Generation:**
    - **FID (Frechet Inception Distance):** Measures the distance between the distribution of generated images and real images in the feature space of an Inception network. Lower is better. The most widely used metric for generative models, though it has known limitations (insensitive to certain artifacts, dependent on sample size).
    - **IS (Inception Score):** Measures both quality (confident classifications) and diversity (uniform class distribution) of generated images. Less reliable than FID.

    See [ISLR Ch. 4](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) for the fundamentals of classification evaluation that underpin these metrics.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Code Demo: mIoU for Segmentation Evaluation

    For each class $c$, compute IoU between predicted and ground-truth pixel masks, then average:

    $$\text{mIoU} = \frac{1}{C}\sum_{c=1}^{C} \frac{|P_c \cap G_c|}{|P_c \cup G_c|}$$
    """)
    return


@app.cell
def _(np):
    def mean_iou(pred_mask, gt_mask, num_classes):
        """Compute mIoU for semantic segmentation evaluation."""
        ious = []
        for c in range(num_classes):
            pred_c = (pred_mask == c)
            gt_c = (gt_mask == c)
            intersection = np.sum(pred_c & gt_c)
            union = np.sum(pred_c | gt_c)
            if union == 0:
                continue  # skip classes not present
            ious.append(intersection / union)
        return np.mean(ious)

    # Simulated 8x8 segmentation predictions (3 classes)
    gt = np.array([[0,0,1,1,2,2,2,2],
                   [0,0,1,1,2,2,2,2],
                   [0,0,1,1,1,2,2,2],
                   [0,0,0,1,1,1,2,2]] * 2)
    pred = gt.copy()
    pred[0, 2] = 0  # a few mispredictions
    pred[1, 4] = 1
    print(f"mIoU: {mean_iou(pred, gt, num_classes=3):.4f}")
    return (mean_iou,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Code Demo: Image Augmentation

    Data augmentation is critical for training vision models --- especially ViTs which lack CNN inductive biases. Here are common augmentations implemented from scratch.
    """)
    return


@app.cell
def _(np):
    def random_horizontal_flip(img, p=0.5):
        """Flip image left-right with probability p."""
        if rng.random() < p:
            return img[:, ::-1].copy()
        return img

    def random_crop(img, crop_h, crop_w):
        """Random crop of size (crop_h, crop_w) from image."""
        H, W = img.shape[:2]
        top = rng.integers(0, H - crop_h + 1)
        left = rng.integers(0, W - crop_w + 1)
        return img[top:top+crop_h, left:left+crop_w]

    def cutout(img, mask_size=4):
        """Cutout: randomly zero out a square region (regularization)."""
        out = img.copy()
        H, W = img.shape[:2]
        cy, cx = rng.integers(0, H), rng.integers(0, W)
        y1, y2 = max(0, cy - mask_size//2), min(H, cy + mask_size//2)
        x1, x2 = max(0, cx - mask_size//2), min(W, cx + mask_size//2)
        out[y1:y2, x1:x2] = 0
        return out

    # Demo on a small 8x8 grayscale image
    demo_img = np.arange(64).reshape(8, 8).astype(float)
    print(f"Original shape:    {demo_img.shape}")
    print(f"After crop(6,6):   {random_crop(demo_img, 6, 6).shape}")
    print(f"Cutout applied:\n{cutout(demo_img, mask_size=4)}")
    return (cutout, random_crop, random_horizontal_flip)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 10. Recommended Reading and Resources

    **Textbooks:**
    - [Goodfellow et al., Deep Learning --- Ch. 9: Convolutional Networks](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf)
    - [Murphy, Probabilistic Machine Learning: Advanced Topics](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) --- vision models, attention, generative models
    - [Bishop, PRML](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) --- foundational probabilistic perspective

    **Courses:**
    - **Stanford CS231n: CNNs for Visual Recognition.** The definitive computer vision course. Lectures on YouTube, comprehensive notes at [cs231n.stanford.edu](https://cs231n.stanford.edu).
    - **Michigan EECS 498-007/598-005 (Justin Johnson):** A more modern version of CS231n. Excellent lectures freely available.

    **Key Papers (read in this order):**

    1. Krizhevsky et al., "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet), 2012.
    2. He et al., "Deep Residual Learning for Image Recognition" (ResNet), 2015. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
    3. Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," 2015. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)
    4. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
    5. Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection" (YOLO), 2016. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
    6. He et al., "Mask R-CNN," 2017. [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
    7. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT), 2020. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
    8. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
    9. Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," 2021. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
    10. Liu et al., "A ConvNet for the 2020s" (ConvNeXt), 2022. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
    11. Kirillov et al., "Segment Anything" (SAM), 2023. [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
    12. Carion et al., "End-to-End Object Detection with Transformers" (DETR), 2020. [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 11. Project Ideas

    **Beginner--Intermediate:**
    - **Object detection system.** Train YOLOv8 on a custom dataset (collect and annotate images using Roboflow or CVAT). Deploy the model and measure mAP and inference latency.
    - **Image segmentation.** Train a U-Net on a medical imaging dataset (cell segmentation, retinal vessel segmentation). Evaluate with mIoU and visualize predictions overlaid on inputs.

    **Intermediate:**
    - **Style transfer.** Implement neural style transfer (Gatys et al., 2015) from scratch using a pre-trained VGG network. Extend to fast style transfer with a feed-forward network.
    - **Visual search engine.** Use a pre-trained CLIP model to encode a dataset of images. Build a search interface where users type text queries and retrieve the most relevant images. Compare CLIP-based retrieval with CNN feature-based retrieval.

    **Advanced:**
    - **Train a small Vision Transformer.** Implement ViT from scratch in PyTorch. Train on CIFAR-100 with and without data augmentation. Compare to a ResNet baseline. Investigate how performance scales with dataset size.
    - **Multimodal image captioning.** Fine-tune a vision-language model (e.g., BLIP-2, LLaVA) on a custom captioning dataset. Evaluate with BLEU, CIDEr, and human judgment. Analyze failure modes.
    - **Real-time panoptic segmentation.** Build a system that takes webcam input and produces real-time panoptic segmentation. Combine a detection model with a segmentation model, or use a unified architecture like Mask2Former.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 12. Code It: Implementation Exercises

    Practice translating CV concepts into working code. Each exercise gives you a skeleton --- fill in the `TODO` sections.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 1: Implement 2D Max Pooling

    Max pooling downsamples a feature map by taking the maximum value in each non-overlapping window. Given an input of shape $(H, W)$ and a pool size $k$, the output has shape $(H/k, W/k)$.
    """)
    return


@app.cell
def _(np):
    def max_pool2d(feature_map, pool_size=2):
        """Downsample feature_map by taking max in each (pool_size x pool_size) window.

        Args:
            feature_map: numpy array of shape (H, W)
            pool_size: size of the pooling window

        Returns:
            pooled: numpy array of shape (H // pool_size, W // pool_size)
        """
        H, W = feature_map.shape
        out_h, out_w = H // pool_size, W // pool_size
        pooled = np.zeros((out_h, out_w))

        # TODO: Loop over output positions and compute max in each window
        # for i in range(out_h):
        #     for j in range(out_w):
        #         window = ...  # extract the (pool_size x pool_size) region
        #         pooled[i, j] = ...  # take the max

        return pooled

    # Test: should downsample 4x4 -> 2x2
    test_feat = np.array([[1, 3, 2, 4],
                          [5, 6, 7, 8],
                          [9, 2, 1, 0],
                          [3, 4, 5, 6]], dtype=float)
    # Expected: [[6, 8], [9, 6]]
    print("Input:\n", test_feat)
    print("Your max_pool2d output:\n", max_pool2d(test_feat, pool_size=2))
    return (max_pool2d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2: Build a Simple Feature Pyramid

    Feature Pyramid Networks (FPN) combine features at multiple scales. Given a list of feature maps at decreasing resolutions, upsample each to the largest resolution and concatenate. Use nearest-neighbor upsampling.
    """)
    return


@app.cell
def _(np):
    def build_feature_pyramid(feature_maps):
        """Upsample all feature maps to the largest resolution and concatenate.

        Args:
            feature_maps: list of numpy arrays [(H1,W1), (H2,W2), ...] where
                          H1 >= H2 >= ... (decreasing resolution)

        Returns:
            combined: numpy array of shape (H1, W1, num_maps)
        """
        target_h, target_w = feature_maps[0].shape
        upsampled = []

        for fm in feature_maps:
            # TODO: Upsample fm to (target_h, target_w) using nearest-neighbor
            # Hint: use np.repeat along each axis with the appropriate scale factor
            # scale_h = target_h // fm.shape[0]
            # scale_w = target_w // fm.shape[1]
            # up = ...
            up = fm  # placeholder --- replace with upsampled version
            upsampled.append(up)

        # TODO: Stack into (H1, W1, num_maps)
        # combined = np.stack(upsampled, axis=-1)
        combined = None  # replace
        return combined

    # Test: 3 feature maps at decreasing resolution
    fm1 = rng.standard_normal((8, 8))   # full resolution
    fm2 = rng.standard_normal((4, 4))   # 2x downsampled
    fm3 = rng.standard_normal((2, 2))   # 4x downsampled
    result = build_feature_pyramid([fm1, fm2, fm3])
    print(f"Expected output shape: (8, 8, 3), Got: {result.shape if result is not None else 'None'}")
    return (build_feature_pyramid,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 3: Implement Precision-Recall for Detection

    Given a list of detections (sorted by confidence) and their ground-truth matches, compute precision and recall at each detection threshold, then compute Average Precision (AP) as the area under the precision-recall curve.
    """)
    return


@app.cell
def _(np):
    def compute_ap(is_tp, num_gt):
        """Compute Average Precision from a list of true-positive flags.

        Args:
            is_tp: list/array of booleans, sorted by descending confidence.
                   True if that detection matched a ground-truth box (IoU > threshold).
            num_gt: total number of ground-truth objects for this class.

        Returns:
            ap: float, area under the precision-recall curve
            precisions: array of precision values
            recalls: array of recall values
        """
        is_tp = np.array(is_tp, dtype=float)
        n = len(is_tp)

        # TODO: Compute cumulative TP and FP counts
        # cum_tp = ...
        # cum_fp = ...

        # TODO: Compute precision = cum_tp / (cum_tp + cum_fp)
        # precisions = ...

        # TODO: Compute recall = cum_tp / num_gt
        # recalls = ...

        # TODO: Compute AP as area under PR curve (use trapezoidal rule or
        # the 11-point interpolation: np.trapz(precisions, recalls))
        # ap = ...

        precisions = np.zeros(n)  # placeholder
        recalls = np.zeros(n)     # placeholder
        ap = 0.0                  # placeholder
        return ap, precisions, recalls

    # Test: 6 detections, 4 ground-truth objects
    # Sorted by confidence: [TP, FP, TP, TP, FP, TP]
    is_tp = [True, False, True, True, False, True]
    ap, prec, rec = compute_ap(is_tp, num_gt=4)
    print(f"AP: {ap:.4f}")
    print(f"Precisions: {prec.round(3)}")
    print(f"Recalls:    {rec.round(3)}")
    return (compute_ap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 4: Multi-Head Self-Attention

    Extend single-head attention to multi-head: split the embedding into $h$ heads, run attention independently on each, then concatenate and project.
    """)
    return


@app.cell
def _(np):
    def multi_head_attention(tokens, num_heads=2):
        """Multi-head self-attention over (N, D) token sequence.

        Args:
            tokens: numpy array of shape (N, D)
            num_heads: number of attention heads (must divide D)

        Returns:
            output: numpy array of shape (N, D)
        """
        N, D = tokens.shape
        assert D % num_heads == 0
        d_k = D // num_heads

        # TODO: For each head h in range(num_heads):
        #   1. Create W_q, W_k, W_v of shape (D, d_k) (random init for demo)
        #   2. Compute Q, K, V = tokens @ W_q, tokens @ W_k, tokens @ W_v
        #   3. Compute scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V
        #   4. Collect the (N, d_k) output for this head

        # TODO: Concatenate all head outputs -> (N, D)
        # TODO: Apply output projection W_o of shape (D, D)

        output = np.zeros_like(tokens)  # placeholder
        return output

    demo = rng.standard_normal((5, 8))
    mha_out = multi_head_attention(demo, num_heads=2)
    print(f"Input shape: {demo.shape}, Output shape: {mha_out.shape}")
    return (multi_head_attention,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 5: YOLO-Style Grid Predictions

    YOLO divides the image into an $S \times S$ grid. Each cell predicts $B$ bounding boxes (center offset, width, height, confidence) plus $C$ class probabilities. Implement the prediction decoding step.
    """)
    return


@app.cell
def _(np):
    def decode_yolo_grid(predictions, S=3, B=2, C=3, img_size=96):
        """Decode YOLO grid predictions into absolute bounding boxes.

        Args:
            predictions: (S, S, B*5 + C) array. For each cell and each box:
                         [tx, ty, tw, th, conf, ...] + C class probs
                         tx, ty in [0,1] are offsets within the cell
                         tw, th in [0,1] are relative to image size
            S: grid size
            B: boxes per cell
            C: number of classes
            img_size: input image dimension (assume square)

        Returns:
            boxes: list of (x1, y1, x2, y2, confidence, class_id) tuples
        """
        cell_size = img_size / S
        boxes = []

        for i in range(S):
            for j in range(S):
                cell_pred = predictions[i, j]
                class_probs = cell_pred[B * 5:]  # last C values

                for b in range(B):
                    offset = b * 5
                    tx, ty, tw, th, conf = cell_pred[offset:offset + 5]

                    # TODO: Convert cell-relative (tx, ty) to absolute center (cx, cy)
                    # cx = (j + tx) * cell_size
                    # cy = (i + ty) * cell_size

                    # TODO: Convert (tw, th) to absolute width and height
                    # w = tw * img_size
                    # h = th * img_size

                    # TODO: Convert center (cx, cy, w, h) to corner (x1, y1, x2, y2)
                    # TODO: Get class_id = argmax of class_probs
                    # TODO: Append (x1, y1, x2, y2, conf, class_id) to boxes
                    pass

        return boxes

    # Test with random predictions
    S, B, C = 3, 2, 3
    preds = rng.random((S, S, B * 5 + C))
    decoded = decode_yolo_grid(preds, S=S, B=B, C=C)
    print(f"Decoded {len(decoded)} boxes from {S}x{S} grid with {B} boxes/cell")
    return (decode_yolo_grid,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    *Next steps: If you have not covered 2D --- Convolutional Neural Networks, start there for the foundational mechanics. For the generative side of vision (diffusion models, GANs for images), see 3B --- Generative Models. To understand how vision and language converge, explore Path A --- NLP for the language side of multimodal models.*
    """)
    return


if __name__ == "__main__":
    app.run()

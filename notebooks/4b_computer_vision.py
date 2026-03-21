import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Path B --- Computer Vision

    Computer vision asks machines to derive meaningful information from images and video. It is one of the oldest subfields of AI and the one where deep learning first proved its dominance --- AlexNet's victory in the 2012 ImageNet challenge is often cited as the moment that reignited the entire field. This guide maps the modern landscape, from classical detection and segmentation to Vision Transformers and multimodal models.
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
    ---

    ## 4. Instance and Panoptic Segmentation

    **Semantic segmentation** labels every pixel with a class but does not distinguish between individual instances of the same class (two adjacent people are both labeled "person").

    **Instance segmentation** detects each object instance and produces a pixel-level mask for each. **Mask R-CNN** (He et al., 2017) extends Faster R-CNN by adding a mask prediction branch --- for each detected region, it predicts a binary mask in parallel with the class and box. Uses RoI Align (bilinear interpolation instead of quantized RoI pooling) for pixel-accurate alignment.

    **Panoptic segmentation** unifies semantic and instance segmentation: every pixel gets a class label (semantic) *and* an instance ID for countable objects (instance). This gives a complete scene understanding --- both "stuff" (sky, road, grass) and "things" (cars, people, dogs).

    **Segment Anything Model (SAM)** (Kirillov et al., 2023): A foundation model for segmentation. Trained on 1 billion masks from 11 million images. Given any prompt (point, box, text), it produces high-quality segmentation masks. SAM is a promptable segmentation model --- it generalizes to unseen objects and domains without fine-tuning. This is the "GPT-3 moment" for segmentation. SAM 2 extends this to video.
    """)
    return


@app.cell(hide_code=True)
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
    ---

    ## 8. Video Understanding (Brief)

    Video adds a temporal dimension to vision:

    - **Temporal modeling approaches:** 3D convolutions (C3D, I3D) extend 2D convolutions along the time axis. Two-stream networks process appearance (RGB) and motion (optical flow) separately. Video transformers (TimeSformer, ViViT) apply attention across spatial and temporal dimensions.
    - **Action recognition:** Classify what action is being performed in a video clip. Kinetics-400/700 is the standard benchmark. Modern approaches use video foundation models pre-trained on large-scale data.
    - **Video object segmentation and tracking:** Follow objects across frames. SAM 2 extends the Segment Anything model to video with memory-based temporal propagation.
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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

    ---

    *Next steps: If you have not covered 2D --- Convolutional Neural Networks, start there for the foundational mechanics. For the generative side of vision (diffusion models, GANs for images), see 3B --- Generative Models. To understand how vision and language converge, explore Path A --- NLP for the language side of multimodal models.*
    """)
    return


if __name__ == "__main__":
    app.run()

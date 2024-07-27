
![rt-detr banner](imgs/rt-detr/rt-detr-banner.png)

## Abstract

DETRs have been improved a lot w.r.t detecting objects but they are no where close to the traditional Real Time YOLO detectors when it comes to `Real Time`.

The authors introduced the first transformer based detector which beats YOLO for most of the benchmarks but the impressive part is that it is even faster than them!

- Eliminate NMS (introduced by DETRs)
	- But actually reap its benefits
- Efficient Hybrid Encoder
	- Multi-Scale Feature Enhancement 
	- Decoupling Multi-Scale Feature Interactions
		- Intra-Scale & Cross-Scale
- Enhanced Query Selection Mechanism
- Flexible tuning without retraining:
	- Negligible Accuracy Loss
	- Impressive Latency Benefits
- 2 Step Development Process
	- Maintain accuracy and improve latency
	- Then maintain the improved latency and improve the accuracy


## Motivation & Challenges Tackled

### Issue with Traditional Detectors

**Negative Effect Of NMS**

The Speed and Accuracy of YOLOs are negatively affected by the NMS. Moreover, considering that different scenarios place different
emphasis on recall and accuracy, it is necessary to carefully
select the appropriate NMS thresholds, which hinders the
development of real-time detectors.

### Issue with Existing DETRs

Existing DETR models are NMS free but due to the high computational cost they are not able to reap the benefits of being NMS free. 
- Inefficient and sub-optimal multi-scale interaction methodology
- Poor query selection for the decoder

**Poor Multi-Scale Interaction**

Multi-scale features is beneficial in accelerating the training convergence, it leads to a significant increase in the length of the sequence feed into the encoder. Concatenating the multi-scale features and passing as an input!

Moreover, is it even required for the high level features to interact with the low level ones? Answer is No.

**Let The Detector Do Its Work**

Current query selection directly incorporates classification scores for selection. In fact, the query selection should only select the query, its important for the detector to model the location & classification of the objects. Hindering this process, restricts the detector to model to its full potential resulting in poor features with low localisation confidence.

> The authors consider query initialization as a breakthrough to further improve performance.


## Nice Recap For Existing Related Work

- `Yolov1` : First CNN based One stage OD achieving true Real Time.
- 2 Types of YOLO detectors
	- Anchor Based
	- Anchor Free
- End-2-End Object Detectors
	- Intro to DETR
	- Eliminates NMS, employs bipartite matching 
		- Predicting one-to-one object set.
	- Issues with DETR,
		- Slow training convergence
		- High computational cost
		- hard-to-optimize queries
	- `Deformable DETR`
		- Multi-Scale features (accelerating convergence)
		- Deformable Attention
	- `DAB DETR` & `DN DETR`
		- Iterative refinement scheme
		- Denoising training
	- `Group DETR`
		- Group wise one-to-many assignment
	- `Efficient DETR` & `Sparse DETR`
		- Reduction in computation cost
		- Achieved by reducing no. of encoder/decoder layers or the no of updated queries
	- `Lite DETR`
		- Improvement in Encoder Efficiency
			- Reducing update frequency of low lvl features in an interleaved way.
	- `Conditional DETR` & `Anchor DETR`
		- Decrease the optimization difficulty of queries


## NMS Analysis

The execution time of NMS primarily depends on the number of boxes and 2 thresholds, IOU threshold & Confidence threshold.

![No. Of boxes (YOLOv5 vs YOLOv8 | Anchor based vs Anchor free)](imgs/rt-detr/bbox-count-yolov5-yolov8.png)

**Effect Of IOU & confidence threshold on accuracy and NMS execution time**

![iou-conf-bench](imgs/rt-detr/iou-conf-thres-latency-acc.png)

The usual benchmark only reports the model inference time which excluded the NMS time. The benchmark shown in the above table is purely for NMS:

The authors used `Tensor RT's efficientNMSPlugin`. This involves multiple kernels such as `EfficientNMSFilter, RadixSort, EfficientNMS` but they report the execution time of only the `EfficientNMS` kernel.

Execution time increases as the confidence threshold decreases or the IOU threshold increases. The reason is that the high confidence threshold directly filters out more prediction boxes, whereas the high IoU threshold filters out fewer prediction boxes in each round of screening.

> Anchor-free detectors outperform anchor-based detectors with equivalent accuracy for YOLO detectors because the former require less NMS time than the latter.


## The Real Time DETR

### Model Overview

![model arch](imgs/rt-detr/rt-detr-overview-architecture.png)

- Backbone
- Efficient Hybrid Encoder
- Transformer Decoder
- Auxiliary Prediction Heads

`AIFI` - Attention based Intra-scale Feature Interaction

`CCFF` - CNN based Cross Scale Feature Fusion

**Fusion Block**

![Fusion Block](imgs/rt-detr/rt-detr-ccff.png)

> S3, S4, S5 are the features from the last three stage of the Backbone

The efficient encoder converts the multi-scale feature into a sequence of image features through intra-scale feature interaction and cross-scale feature fusion. 

The uncertainty-minimal query selection is used to select a fixed number of encoder features to serve as initial object queries for the decoder. The decoder then iteratively optimizes the object queries to generate categories and boxes. 


### Efficient Hybrid Encoder

#### Computational Bottleneck Analysis

Deformable DETR, introduced multi-scale feature reduces the computation cost using deformable attention but the increase in the sequence length results in a bottleneck at the encoder.

> Acc. to Lin et al, encoder accounts for 49% of the GFLOPs but contributes only 11% of the AP in Deformable-DETR

Intuitively, high-level features that contain rich semantic information
about objects are extracted from low-level features, making it redundant to perform feature interaction on the concatenated multi-scale features.

#### Hybrid Design

- Attention Based Intra-Scale Feature Interaction (AIFI)
- CNN Based Cross-Scale Feature Fusion (CCFF)

AIFI further reduces the computational cost by performing the intra-scale interaction only on S5.

The reason is that applying the self-attention operation to high-level features with richer semantic concepts captures the connection between conceptual entities, which facilitates the localization and recognition of objects by subsequent modules.

The role of the fusion block is to fuse two queries for the decoder. Specifically, the feature uncertainty adjacent scale features into a new feature.
The Hybrid Encoder's calculation is as follows:

![hybrid encoder calculations](imgs/rt-detr/rt-detr-encoder-calculations.png)

### Uncertainty-Minimal Query Selection

Traditional query selection schemes use confidence score to select top K features to be the initial object queries. The confidence scores represent the likelihood that the feature includes foreground objects.

> The performance score is a latent variable that is jointly correlated with both classification and localization. The detectors are required to model the category & localization for quality features.

The traditional scheme results is sub-optimal initialization which hinders the performance of the detector.

**Uncertainty Minimal Query Selection** - Explicitly constructs and optimizes the epistemic uncertainty to model the joint latent variable of encoder features, thereby providing high quality queries for the decoder.

> epistemic uncertainty - Uncertainty in the knowledge about the world due to limitations in understanding, data etc.

To minimize the uncertainty of the queries, the uncertainty is integrated to the loss function for gradient based optimization.

The uncertainty is defined as the discrepancy between the predicted distributions of localization & classification.

![uncertainty of the queries](imgs/rt-detr/rt-detr-uncertainty.png)

- y & yhat : prediction and ground truth
- Xhat : encoder features
- chat & bhat : category and bounding box

![uncertainty-vs-vanila](imgs/rt-detr/rt-detr-uncertainty-vanila-query.png)

The closer the dot to the top right of the figure, the higher the quality of the corresponding feature.

> The plot only has features with classification score > 0.5

- Purple dots are concentrated to the top right
- Green are concentrated to the bottom right
- This shows uncertainty produces high-quality encoder features.
- There are 138% more purple than green dots
	- Green has more dots with conf < 0.5
- 120% more purple than green dots with both scores greater than 0.5

Quantitative results further demonstrate that the uncertainty-minimal query selection provides more features with accurate classification and precise location for queries, thereby improving the accuracy of the detector.

### Scaled RT-DETR

RT-DETR supports flexible scaling.

**Hybrid Encoder** - we control the width by adjusting the embedding
dimension and the number of channels, and the depth by adjusting the number of Transformer layers and RepBlocks.

**Decoder** - width and depth of the decoder can be controlled by
manipulating the number of object queries and decoder layers.

The authors observe that removing a few decoder layers at the end has minimal effect on accuracy, but greatly enhances inference speed.

> RT-DETR has input shape of (640, 640) just like YOLO. Other DETRs use an input shape of (800, 1333).


### Ablation 

#### Hybrid Encoder

![Encoder Ablation](imgs/rt-detr/rt-detr-encoder-ablation.png)

**Variant B improves accuracy by 1.9% AP and increases the latency by 54%**
- This proves that the intra-scale feature interaction is significant, but the single-scale Transformer encoder is computationally expensive.

**Variant C delivers a 0.7% AP improvement over B and increases
the latency by 20%**
- This shows that the cross-scale feature fusion is also necessary but the multi-scale Transformer encoder requires higher computational cost.

**Variant D delivers a 0.8% AP improvement over C, but reduces latency by 8%**
- Decoupling intra-scale interaction and cross-scale fusion not only reduces computational cost but also improves accuracy.

**Compared to variant D, DS5 reduces the latency by 35% but delivers 0.4% AP improvement**
- Ds5 : variant of D where the intra-scale inetraction is performed on the S5 feature (high lvl feature).
-  Shows that intra-scale interactions of lower level features are not required.

![Table](imgs/rt-detr/rt-detr-encoder-ablation-table.png)

> Authors variant, E delivers 1.5% AP improvement over D. Despite a 20% increase in the number of parameters, the latency is reduced by 24%, making the encoder more efficient.

#### Query Selection

![Table Query](imgs/rt-detr/rt-detr-query-selection-ablation.png)

> The results show that the encoder features selected by uncertainty-minimal query selection not only increase the proportion of high classification scores (0.82% vs 0.35%) but also provide more high quality features (0.67% vs 0.30%).


#### Decoder

![Decoder Table](imgs/rt-detr/rt-detr-ablation-decoder.png)

Acc. to the table, the difference in accuracy between adjacent decoder layers gradually decreases as the index of the decoder layer increases.

RT-DETR supports flexible speed tuning by adjusting the number of decoder layers without retraining, thus improving its practicality.

### Limitations

Although the proposed RT-DETR outperforms the state-of-the-art real-time detectors and end-to-end detectors with similar size in both speed and accuracy.

It shares the same limitations as other DETRs:

> The performance on small object is still inferior than the strong real-time detectors (YOLOs).


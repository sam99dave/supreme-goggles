
This is not a details review of the paper, its more like notes:
For detailed info checkout the [paper](https://arxiv.org/abs/2212.02623)

![udop-banner](imgs/udop/banner.png)

## Abstract

The authors are proposing an architectures:
- Unifies text, image and layout modalities together
- Introducing layout induced architecture
- Homogeneous vocabulary for texts and document layout
- Self-supervised and supervised pretraining
- Unifying multi-domain downstream tasks into a prompt-based sequence generation scheme.

## Challenges Tackled

### 2D Document Layout

For traditional vision-text data, the text modality is usually the high-level description of the corresponding image or task prompt .

When comparing document images with other images used for join embedding models and other classic vision language research, we get to know that in case of document images the text is structurally embedded in the image itself along with other information such as style, figures etc.

For Document AI, the cross modality (text & visual) interactions are much stronger than regular vision language data as the text modality is visually situated in the image at specific layouts.

### Unifying Diverse Downstream Tasks

Diverse downstream task as in,
- Document QA
- Layout Detection
- Classification
- Information Extraction

Usually different heads are implemented for these different tasks rendering multiple models based on the task.

In short, these are the 2 challenges:
- How to utilize the strong correlation between image, text and layout modalities and unify them to model the document as a whole?
- How can the model efficiently and effectively learn diverse vision, text, and layout tasks across different domains?

## Solution Overview

Classic Encoding Architectures:
- Concatenation of text and visual tokens to a multi-modal transformer
- 2 tower / 3 tower architectures
	- Independent encoding for modalities
	- Projection heads / fusion networks on top to generate multi-modal representation
	- Using joint embedding models (e.g. CLIP) for mapping the modalities

The model architecture consists of the following components:
- A Unified Vision, Text and Layout Encoder (Modal Agnostic Encoder)
- Text-Layout Decoder
- Vision Decoder

### Modal Agnostic Encoder

![layout induced vision-text embedding](imgs/udop/unified-encoder.jpg)

`v` -> Document Image
`si` -> Word Token
`(x1, y1, x2, y2)i` -> Layout 
`P` -> Patch size
`PxPxC` -> Dimension for the patch size

Each patch is encoded with a `D-dim` vector, these vectors are then grouped as a sequence of vectors.

Each text tokens are also converted to numerical `D-dim` embeddings (through vocabulary look-up).

**Vision-Text Embeddings**
This embedding is a joint representation, its the sum of the text and image patch feature.

> si_updated = si + vj 

**Layout Induced Vision-Text Embeddings**
The authors define a  `layout indicator function` for image patch and token embeddings.
- 1 >> if the center of `si` falls within the image patch vj
- 0 >> otherwise

> si_updated = si + vj  |  layout indicator == 1
> vj_updated = vj  |  layout indicator == 0


## Performance Overview

## Limitations

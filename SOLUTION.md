# Solution Writeup

I would like to thank the organizers for providing a real-world test dataset
and for hosting the competition on Kaggle, which made it easy to iterate and
compare approaches with others. Before the competition, I assumed that
biomedical research was driven almost entirely by well-intentioned contributors,
with duplication or manipulation being rare exceptions. However, as I explored
the literature and worked with the data, it became clear that many such cases
likely remain undiscovered. This shift in perspective strongly motivated me to
build the most robust and effective solution possible, and I genuinely enjoyed
the process.

## Brief Description of the Solution

1. Detect **Blot** (western blot) and **Microscopy** (microscopy/macroscopy)
   regions using a YOLO-based model.
2. Apply models tailored to each data type to identify duplicate patterns.
3. If no matches are found, use a segmentation model from the best available
   public kernel, initially based on @ravaghi and later replaced with
   @pankajiitr's code without any modifications, and constrain predictions with
   Blot/Microscopy masks. This consistently provided a ~0.005-0.01 improvement,
   so I included it in both final submissions.
4. If no evidence is detected in the previous steps, classify the sample as
   **authentic**.

I started by reviewing related publications to better understand the new domain
and build intuition about the data. This analysis suggested that roughly 50% of
reported cases involve western blots, while microscopy and macroscopy images
make up a substantial portion of the remainder. For simplicity, I merged
microscopy and macroscopy into a single class. Based on this, I focused on these
categories and deprioritized the rest, as they were less likely to be prominent
in the test set.

I also experimented with a simple detector for identical
[XRD patterns](https://pubpeer.com/publications/739636EFE147869FB290A67B01E5C0),
but it had no measurable impact on the leaderboard, so I abandoned this
direction. Similarly, I chose not to invest time in inpainting detection, as it
appears relatively rare in real publications and I expected an existing public
kernel to handle such cases with some degree of success if needed; the same
reasoning applied to microscopy images with duplicated cells.

In parallel, I set up a pipeline to download biomedical papers from
[PubMed](https://pubmed.ncbi.nlm.nih.gov/). By mid-December, I had collected
~2 million `.tar.gz` archives, about 14 TB, under CC0 / CC BY licenses. I
extracted source images from the PDFs, applied YOLO-based panel detection, and
obtained around 6.6M western blot images and 16.3M microscopy/macroscopy images.
I then progressively built a deduplicated dataset, following a setup where each
"identity" was represented by a single image.

In practice, exact duplicates are unlikely in biomedical data; however, I still
identified hundreds of thousands of near-duplicate or even identical samples.
Some of these cases have also been reported on [PubPeer](https://pubpeer.com/),
although many likely remained unnoticed. This dataset later proved useful for
training embedding models.

In addition to the large training dataset, I created a reliable validation set.
It was initially based on BioFors, but I found it insufficient for proper
evaluation: many negative pairs were actually positives, and some positives were
incorrect. To address this, I reannotated all positive pairs and as many
negative pairs as possible. This proved critical. Without a clean validation
set, it is nearly impossible to iterate and improve the solution.

## Baseline

At the beginning of the competition, I established a pipeline skeleton that
remained largely unchanged throughout. It consisted of the organizers'
YOLO-based panel extractor, followed by candidate filtering using SigLIP2 for
microscopy and a binary "similar / not similar" classifier for western blots.
The selected candidates were then passed to a LightGlue-SuperPoint matcher for
precise localization:

- `max_keypoints = 2048`
- `inlier_threshold = 100`
- `match_score_threshold = 0.6`

This setup achieved a score of 0.327. When ensembled with a 0.319 public
kernel, it reached **0.341**, which was the top score at that stage, with the
second-best at 0.324.

Later, I improved the pipeline to 0.342 by replacing SigLIP2 with my binary
classifier and retraining YOLO on my own data. Combined with the public kernel,
this version reached approximately **0.353**. At that point, the best score was
~0.355 by @melgor, and his presence on the leaderboard motivated me to invest
more effort into embedding-based approaches.

I also started to observe signs of performance saturation, and as a result,
moved away from classification approaches. For the remainder of the competition,
I focused entirely on embeddings, effectively replacing classification scores
with similarity measures and tuning the corresponding thresholds.

## YOLO-Based Panel Detector

As a starting point for the detector, I used the competition organizers'
[detector](https://github.com/phillipecardenuto/upm/tree/main/panel-extractor).
However, I quickly encountered several issues, including **missed detections**,
**false positives**, and **inconsistent labels**.

![Panel detector examples](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F29607288%2F3917a8436b5c13528e6548a412459a9c%2F2026-04-27%20170820.png?generation=1777296062816155&alt=media)

To address this, I reannotated the dataset to correct inaccurate bounding boxes
and labeling errors, and to better align it with the competition requirements,
for example handling single-panel images. I also sampled images from a large
unlabeled biomedical dataset and annotated them. In addition, I analyzed failure
cases of the detector and incorporated those examples into the training set.

In total, this resulted in 11.3k training figures in version 9 of the dataset
and 12.3k in version 10. However, leaderboard performance with version 10 was
slightly worse, so I stopped further annotation at that point.

The model was trained using the Ultralytics YOLO library:

```bash
yolo detect train \
  model=yolo12x.pt \
  data=dataset_V9.yaml \
  epochs=100 \
  imgsz=640 \
  batch=80 \
  translate=0.2 \
  flipud=0.2 \
  mosaic=1 \
  cos_lr=True \
  warmup_epochs=0 \
  optimizer="radam" \
  lr0=0.001
```

![YOLO training result](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F29607288%2F477661ee5ca5c6f2db79fe84d9e1475d%2F2026-04-27%20173634.png?generation=1777297044745513&alt=media)

Panels corresponding to `{"Graphs", "Flow Cytometry", "Body Imaging"}` were
excluded in the best submission, which scored 0.550 on the private leaderboard.
Including **Body Imaging** led to a performance drop of approximately 0.009.

Towards the end of the competition, I trained a multiscale variant and used an
ensemble of these models for the final submission, which provided a +0.002-0.003
improvement on the leaderboard compared to the single model above:

```bash
yolo detect train \
  model=yolo12x.pt \
  data=dataset_V10.yaml \
  epochs=80 \
  imgsz=640 \
  batch=16 \
  translate=0.2 \
  flipud=0.2 \
  mosaic=1 \
  multi_scale=True
```

## Embeddings

Midway through the competition, I shifted my focus to learning embeddings that
maximize separation between similar and dissimilar samples. Here, "items" refer
to different types of images:

- **Duplicate**: images that are approximately 80-100% derived from the same
  source.
- **Overlap**: images that share roughly 5-100% common origin, with duplicates
  being a subset of this category.

![Embedding examples](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F29607288%2Fd6cd9c1c639121bf64eefd1eb9695445%2F2026-04-27%20215210.png?generation=1777312401814733&alt=media)

I used deduplicated datasets for western blot and microscopy/macroscopy images,
investing significant effort to ensure high data quality, under the assumption
that each image appears only once. I started with
[InfoNCE loss](https://arxiv.org/pdf/1807.03748), but then switched to
[SupCon](https://arxiv.org/pdf/2004.11362).

For each batch, I sampled N images and generated one augmented version per image,
resulting in 2N samples. This produced N positive pairs, while all remaining
pairs within the batch were treated as non-similar. I experimented with various
alternative metric learning losses, but they consistently underperformed, so I
ultimately adopted `SupCon` as the primary training objective, using a
temperature in the range of 0.07-0.09 and the largest batch size permitted by GPU
memory. I used
[nextvit_small](https://huggingface.co/timm/nextvit_small.bd_ssld_6m_in1k) as
the backbone for each model toward the end of the competition.

For western blot "Duplicate" cases, I applied augmentations inspired by
real-world variations. I used
[albumentations](https://github.com/albumentations-team/albumentations) for:

- geometric transformations: shifts, scaling, small rotations, perspective
  transforms, and grid distortions
- flips and rotations: horizontal/vertical flips and 180-degree rotation
- color transformations: inversion, CLAHE, and grayscale conversion
- image quality degradation: blur and compression

![Western blot augmentations](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F29607288%2F4e975dfb57582f524f7943e6c3ccda27%2F2026-04-27%20222419.png?generation=1777314283147218&alt=media)

For the western blot "Overlap" case, I initially relied on simple heuristics,
but later replaced them with bounding boxes from a YOLO-based lane detector. The
goal was to identify two views of an image that share at least one common black
"lane." This could correspond either to a simple overlap or to a zoom scenario,
where one image is nested within another. The cropping process was then
restricted to the green-highlighted regions shown below, corresponding to the
gaps between black lanes in the western blot image:

![Western blot overlap crops](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F29607288%2Fefe2cfa0c7604b4cf3e14fee65ce2a07%2F2026-04-27%20220956.png?generation=1777313425171714&alt=media)

For the microscopy class, I designed custom augmentations to simulate common
artifacts observed in real data. These included adding noisy elements such as
letters, bounding boxes, zoom callouts, arrows, and lines, along with
CoarseDropout, inspired by real inpainting and cell copy-paste examples that my
model struggled to handle correctly. Since not all images were suitable for
generating overlap or zoom crops, I clustered the dataset and manually labeled
the clusters into two categories: suitable for cropping and not suitable.

![Microscopy augmentations](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F29607288%2F40b20a9c68f077fffaf41823e94f00af%2F2026-04-27%20225153.png?generation=1777315934812421&alt=media)

Switching from classification to embedding models for western blots improved
the score by +0.017, from 0.353 to 0.370. Further gains came from improving the
western blot overlap embedder by incorporating augmentations based on bounding
boxes from the lane detector described below, adding approximately +0.010.

The microscopy embedder was more challenging to train due to higher input
resolution and larger data volume, so I used a cloud machine, either A100 or RTX
6000, instead of my local RTX 4090 setup. This provided an additional boost
toward the end of the competition, improving the score by +0.015, from 0.431 to
0.446.

## The Idea That Moved the Leaderboard

By mid-December, I decided to follow a simple idea: while western blot images
are generally unique, each individual lane within a blot should be unique as
well. Without clear evidence that this would work, I spent about a week manually
annotating bounding boxes for lane detection. In total, I labeled 3,732 images
to train a YOLO-based detector for dark regions in western blot images. I then
applied this model to the full dataset of ~6M blots, producing approximately 27M
bounding boxes:

![Lane detector examples](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F29607288%2Fcdcfd828a7dad58c9fad610d652f6076%2F2026-04-27%20232637.png?generation=1777318021984840&alt=media)

I reused the pipeline from the western blot "Duplicate" case, modifying only the
input resolution and the strength of augmentations. The first checkpoint already
**improved the leaderboard score from 0.375 to 0.397** without any threshold
tuning, and further adjustment of the similarity threshold increased it to
**0.401**. That is how I ended up spending New Year's Eve.

After cleaning the dataset and tuning the training hyperparameters, I reached a
score of **0.416**. The model started capturing these cases:

![Western blot lane matches](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F29607288%2F9bf008b292e573f701ea75201b213a88%2F2026-04-27%20234225.png?generation=1777318975369083&alt=media)

Finally, applying a simple heuristic provided another boost to **0.431**:

> Count the number of lanes in each panel involved in matches. If more than 50%
> of lanes in a panel are matched, use the entire panel bounding box in the mask
> instead of individual segment bounding boxes.

![Full-panel mask heuristic](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F29607288%2F881b367125632bfa5a78e326060e6651%2F2026-04-27%20234754.png?generation=1777319300391967&alt=media)

The improvement from 0.431 to 0.458 was primarily driven by a stronger
microscopy embedding model. Additional gains came from applying TTA, light
ensembling, and tuning the inference thresholds for similarity and inliers.

## Matching

Candidates from the embedding stage were passed to a matching module, which
filtered out non-relevant microscopy pairs using inlier count and inlier score
thresholds. For western blots, all candidates were retained for mask generation,
even when no keypoints were matched.

For microscopy images, SIFT performed surprisingly well, and despite multiple
attempts, I was unable to replace it with more modern alternatives. However,
SIFT struggled on western blots, often failing to extract reliable keypoints. To
address this, I trained a neural matching pipeline using
[Glue Factory](https://github.com/cvg/glue-factory), specifically ALIKED-n16 +
LightGlue. Compared to using SIFT on western blots, this significantly
accelerated inference from ~2.5 hours to ~20 minutes, while providing only
marginal improvement in leaderboard score, +0.005 or less.

Final setup:

- Microscopy: LightGlue with SIFT features and RANSAC-based geometric
  verification
- Western blots: LightGlue with ALIKED features and MAGSAC-based geometric
  verification

![Matching examples](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F29607288%2Fb93678952e53b2ad7e71305b00f4984a%2F2026-04-28%20144014.png?generation=1777372849011274&alt=media)

## Code

This repository contains a simplified version of the solution:

- one YOLO model for panel detection
- one embedding model per major matching task
- LightGlue-based matching utilities
- no segmentation model

The checkpoint files are distributed separately through GitHub Releases and are
downloaded on demand by the package.

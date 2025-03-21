# <p align='center'>Recent Progress of Implicit Neural Representations:<br> Architectural Improvements and Cross-domain Applicationss</p>
----
This repository collects recent advances in the field of Implicit Neural Representations (INR) along with a curated paper list.
We have organized these papers into five main categories:
- Theoretical Understanding
- Architectural Improvement
- Data Representation
- Capability Exploration
- Downstream Applications
We are committed to keeping this repository updated to the best of our ability. Feel free to submit pull requests to contribute to this collection!

If you find this repository useful for your research, we welcome you to follow our work and consider citing our previous publications:
- (CVPR'25) EVOS: Efficient Implicit Neural Training via EVOlutionary Selector [[paper]](https://arxiv.org/abs/2412.10153) [[code]](hhttps://github.com/zwx-open/EVOS-INR)
- (AAAI'25) Enhancing Implicit Neural Representations via Symmetric Power Transformation [[paper]](https://arxiv.org/abs/2412.09213) [[code]](https://github.com/zwx-open/Symmetric-Power-Transformation-INR)


# News
- [2025.03.21] Repository Initialization.

# Links
- [Awesome Implicit Neural Representation](https://github.com/vsitzmann/awesome-implicit-representations) (*Last Updated in 4 years ago*)
- [Awesome Implicit Neural Representations in Medical Imaging](https://github.com/xmindflow/Awesome-Implicit-Neural-Representations-in-Medical-imaging) (*Last Updated in 2 years ago*)
- [Awesome Implicit NeRF Robotics](https://github.com/zubair-irshad/Awesome-Implicit-NeRF-Robotics)
- [NeRF and Beyond Docs](https://github.com/yangjiheng/nerf_and_beyond_docs) 

# Paper list
<summary>Table of Content</summary>
- [Recent Publications](#recent-publications)
- [Survey, Benchmark and Important Papers](#survey-benchmark-and-important-papers)
- Topical Organization
  - [Theoretical Understanding](#theoretical-understanding)
  - [Architectural Improvement](#architectural-improvement)
  - [Data Representation](#data-representation)
  - [Capability Exploration](#capability-exploration)
  - [Downstream Applications](#downstream-applications)

## Recent Publications
<details span>
<summary><b>CVPR 2025</b></summary>
<ul>
  <li></li>
  <li></li>
  <li></li>
</ul>
</details>

<details span>
<summary><b>ICLR 2025</b></summary>
<ul>
  <li></li>
  <li></li>
  <li></li>
</ul>
</details>

<details span>
<summary><b>AAAI 2025</b></summary>
<ul>
  <li></li>
  <li></li>
  <li></li>
</ul>
</details>

<details span>
<summary><b>NeurIPS 2024</b></summary>
<ul>
  <li></li>
  <li></li>
  <li></li>
</ul>
</details>

<details span>
<summary><b>MM 2024</b></summary>
<ul>
  <li></li>
  <li></li>
  <li></li>
</ul>
</details>


<details span>
<summary><b>ECCV 2024</b></summary>
<ul>
  <li></li>
  <li></li>
  <li></li>
</ul>
</details>

<details span>
<summary><b>ICML 2024</b></summary>
<ul>
  <li></li>
  <li></li>
  <li></li>
</ul>
</details>


<details span>
<summary><b>IJCAI 2024</b></summary>
<ul>
  <li></li>
  <li></li>
  <li></li>
</ul>
</details>


## Survey, Benchmark and Important Papers
---
### Survey
*(ICCV'2023)* **Implicit Neural Representation in Medical Imaging: A Comparative Survey**<br>
[[paper]](https://arxiv.org/abs/2111.05849) [[code]](https://github.com/xmindflow/Awesome-Implicit-Neural-Representations-in-Medical-imaging)<br>


### Important Papers
*(NeurIPS'2020)* **(SIREN) Implicit Neural Representations with Periodic Activation Functions**<br>
[[paper]](https://arxiv.org/abs/2006.09661) [[code]](https://github.com/vsitzmann/siren)<br>

*(ECCV'2020)* **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**<br>
[[paper]](https://arxiv.org/abs/2003.08934) [[code]](https://github.com/bmild/nerf)[[website]](https://www.matthewtancik.com/nerf)<br>


## Theoretical Understanding


## Architectural Improvement
### Basic Architecture & Activation Functions
*(NeurIPS'2020)* **(Position Encoding) Fourier features let networks learn high frequency functions in low dimensional domains**<br>
[[paper]](https://arxiv.org/abs/2006.10739) [[code]](https://github.com/MingyuKim87/fourier_feature_torch)<br>

*(NeurIPS'2020)* **(SIREN) Implicit Neural Representations with Periodic Activation Functions**<br>
[[paper]](https://arxiv.org/abs/2006.09661) [[code]](https://github.com/vsitzmann/siren)<br>

*(ICLR'2021)* **(MFN) Multiplicative Filter Networks**<br>
[[paper]](https://openreview.net/pdf?id=OmtmcPkkhT) [[code]](https://github.com/boschresearch/multiplicative-filter-networks)<br>

*(ECCV'2022)* **(GAUSS) Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate MLPs**<br>
[[paper]](https://arxiv.org/abs/2111.15135)<br>

*(CVPR'2023)* **(WIRE) Wavelet Implicit Neural Representations**<br>
[[paper]](https://arxiv.org/abs/2301.05187) [[code]](https://github.com/vishwa91/wire)<br>

*(CVPR'2024)* **FINER: Flexible spectral-bias tuning in Implicit NEural Representation by Variable-periodic Activation Functions**<br>
[[paper]](https://arxiv.org/abs/2312.02434) [[code]](https://github.com/liuzhen0212/FINER)<br>

*(ICML'2024)* **ReLUs Are Sufficient for Learning Implicit Neural Representations**<br>
[[paper]](https://arxiv.org/abs/2406.02529) [[code]](https://github.com/joeshenouda/relu-inrs)<br>

### Initialization
*(CVPR'2021)* **Learned initializations for optimizing coordinate-based neural representations**<br>
[[paper]](https://arxiv.org/abs/2012.02189) [[code]](https://github.com/sanowar-raihan/nerf-meta)<br>

*(ECCV 2022)* **Transformers as Meta-learners for Implicit Neural Representation**<br>
[[paper]](https://arxiv.org/abs/2208.02801) [[code]](https://github.com/yinboc/trans-inr)<br>

*(ICLR 2025)* **Fast training of sinusoidal neural fields via scaling initialization**<br>
[[paper]](https://arxiv.org/abs/2410.04779)<br>


### Partition-based Acceleration
*(TOG 2021)* **ACORN: Adaptive coordinate networks for neural representation**<br>
[[paper]](https://arxiv.org/abs/2105.02788)[[code]](https://github.com/computational-imaging/ACORN?tab=readme-ov-file)[[website]](http://www.computationalimaging.org/publications/acorn/)<br>


### Sampling
*(CVPR 2024)* **Accelerating Neural Field Training via Soft Mining**<br>
[[paper]](https://arxiv.org/abs/2312.00075)[[code]](https://github.com/computational-imaging/ACORN)[[website]](https://ubc-vision.github.io/nf-soft-mining/)<br>


*(ICML 2024)* **Nonparametric Teaching of Implicit Neural Representations**<br>
[[paper]](https://arxiv.org/pdf/2405.10531)[[code]](https://github.com/chen2hang/INT_NonparametricTeaching)[[website]](https://chen2hang.github.io/_publications/nonparametric_teaching_of_implicit_neural_representations/int.html)<br>

*(CVPR 2025)* **EVOS: Efficient Implicit Neural Training via EVOlutionary Selector**<br>
[[paper]](https://arxiv.org/abs/2412.10153) [[code]](hhttps://github.com/zwx-open/EVOS-INR) <br>



### Data Transformation
*(CVPR 2023)* **DINER: Disorder-Invariant Implicit Neural Representation**<br>
[[paper]](https://arxiv.org/abs/2211.07871)[[code]](https://github.com/Ezio77/DINER)[[website]](https://ezio77.github.io/DINER-website/)<br>

*(CVPR 2024)* **In Search of a Data Transformation That Accelerates Neural Field Training**<br>
[[paper]](https://arxiv.org/abs/2311.17094)[[code]](https://github.com/effl-lab/DT4Neural-Field)<br>

*(CVPR 2024)* **Batch Normalization Alleviates the Spectral Bias in Coordinate Networks**<br>
[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Cai_Batch_Normalization_Alleviates_the_Spectral_Bias_in_Coordinate_Networks_CVPR_2024_paper.pdf)[[code]](https://github.com/Aiolus-X/Norm-INR/tree/main/image%20fitting)<br>

*(AAAI 2025)* **Enhancing Implicit Neural Representations via Symmetric Power Transformation**<br>
[[paper]](https://arxiv.org/abs/2412.09213) [[code]](https://github.com/zwx-open/Symmetric-Power-Transformation-INR)<br>


### Level Of Details

### Regularization


### Alternatives

## Data Representation
### Time Series
### Audio
### Video
### Shape
### Radiance Field
### Alternatives

## Capability Exploration

## Downstream Applications
### Compression
### Image Enhancement




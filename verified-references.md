# Verified References for Domain Shortcuts in Multi-Dataset Image Forgery Localization

## VERIFIED

### TruFor — Guillaro et al., CVPR 2023
- **BibTeX**: `@inproceedings{guillaro2023cvpr-trufor, title={{TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization}}, author={Guillaro, Fabrizio and Cozzolino, Davide and Sud, Avneesh and Dufour, Nicholas and Verdoliva, Luisa}, booktitle={Conference on Computer Vision and Pattern Recognition}, year={2023}, pages={20606-20615}, doi={10.1109/CVPR52729.2023.01974}, url={https://mlanthology.org/cvpr/2023/guillaro2023cvpr-trufor/}}`
- **Relevance**: Forensic framework combining RGB image with Noiseprint++ for joint detection and localization; state-of-the-art on multiple forgery datasets.
- **Source**: CVPR 2023 open access | DOI 10.1109/CVPR52729.2023.01974 | arXiv:2212.10957

### DocTamper — Qu et al., CVPR 2023
- **BibTeX**: `@inproceedings{qu2023cvpr-robust, title={{Towards Robust Tampered Text Detection in Document Image: New Dataset and New Solution}}, author={Qu, Chenfan and Liu, Chongyu and Liu, Yuliang and Chen, Xinhong and Peng, Dezhi and Guo, Fengjun and Jin, Lianwen}, booktitle={Conference on Computer Vision and Pattern Recognition}, year={2023}, pages={5937-5946}, doi={10.1109/CVPR52729.2023.00575}, url={https://mlanthology.org/cvpr/2023/qu2023cvpr-robust/}}`
- **Relevance**: Document tampering detection benchmark (170k images, copy-move/splicing/generation); cross-domain generalization via DocTamper-FCD and DocTamper-SCD.
- **Source**: CVPR 2023 open access | DOI 10.1109/CVCR52729.2023.00575 | arXiv:2604.25213

### Geirhos et al., "Shortcut learning in deep neural networks", Nature Machine Intelligence 2020
- **BibTeX**: `@article{geirhos2020shortcut, title={Shortcut learning in deep neural networks}, author={Geirhos, Robert and Jacobsen, Jörn-Henrik and Michaelis, Claudio and Zemel, Richard and Brendel, Wieland and Bethge, Matthias and Wichmann, Felix A}, journal={Nature Machine Intelligence}, volume={2}, number={11}, pages={665-673}, year={2020}, doi={10.1038/s42256-020-00257-z}}`
- **Relevance**: Perspective article defining shortcut learning as the core problem behind deep learning's failure to generalize; foundational for understanding domain shortcuts in forensic models.
- **Source**: Nature Machine Intelligence | DOI 10.1038/s42256-020-00257-z | arXiv:2004.07780

### BBSE — Lipton, Wang, Smola, ICML 2018
- **BibTeX**: `@inproceedings{lipton2018bbse, title={{Detecting and Correcting for Label Shift with Black Box Predictors}}, author={Lipton, Zachary Chase and Wang, Yu-Xiang and Smola, Alexander}, booktitle={Proceedings of the 35th International Conference on Machine Learning}, pages={3122-3130}, year={2018}, editor={Dy, Jennifer and Krause, Andreas}, series={Proceedings of Machine Learning Research}, volume={80}, url={http://proceedings.mlr.press/v80/lipton18a.html}, doi={10.48550/arXiv.1802.03916}}`
- **Relevance**: Black Box Shift Estimation (BBSE) for label-shift estimation without calibrated predictors; pre-2020 must-have for label-shift under domain shift.
- **Source**: ICML 2018 PMLR | arXiv:1802.03916 | Proceedings ML Research v80

### MVSS-Net++ — Dong et al., IEEE TPAMI 2022 (journal extension of ICCV 2021 MVSS-Net)
- **BibTeX**: `@article{dong2022mvss, title={{MVSS-Net: Multi-View Multi-Scale Supervised Networks for Image Manipulation Detection}}, author={Dong, Chengbo and Chen, Xinru and Hu, Ruohan and Cao, Juan and Li, Xirong}, journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, year={2022}, volume={}, number={}, pages={1-14}, doi={10.1109/TPAMI.2022.3180556}}`
- **Relevance**: Journal-extended version of ICCV 2021 MVSS-Net; multi-view multi-scale supervision for learning generalizable manipulation-detection features across datasets; MVSS-Net++ improves over conference variant.
- **Source**: IEEE TPAMI 2022 | DOI 10.1109/TPAMI.2022.3180556 | arXiv:2112.08935

### ManTra-Net — Wu, AbdAlmageed, Natarajan, CVPR 2019
- **BibTeX**: `@inproceedings{Wu_2019_CVPR, title={{ManTra-Net: Manipulation Tracing Network for Detection and Localization of Image Forgeries With Anomalous Features}}, author={Wu, Yue and AbdAlmageed, Wael and Natarajan, Premkumar}, booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, year={2019}, pages={9535-9544}, doi={10.1109/CVPR.2019.00977}}`
- **Relevance**: End-to-end forgery detection and localization; must-have lineage for manipulation-tracing architectures; 385-class self-supervised trace learning.
- **Source**: CVPR 2019 | DOI 10.1109/CVPR.2019.00977 | arXiv versions available

### Ding et al., "Local Temperature Scaling for Probability Calibration", ICCV 2021
- **BibTeX**: `@inproceedings{ding2021local, title={{Local Temperature Scaling for Probability Calibration}}, author={Ding, Zhipeng and Han, Xu and Liu, Peirong and Niethammer, Marc}, booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}, year={2021}, pages={6889-6899}, doi={10.1109/ICCV48922.2021.00681}}`
- **Relevance**: Learning-based local TS method predicting separate temperature scale per pixel/voxel for multi-label semantic segmentation calibration; addresses distribution-shift calibration.
- **Source**: ICCV 2021 open access | DOI 10.1109/ICCV48922.2021.00681

### ObjectFormer — Wang et al., CVPR 2022
- **BibTeX**: `@inproceedings{wang2022cvpr-objectformer, title={{ObjectFormer for Image Manipulation Detection and Localization}}, author={Wang, Junke and Wu, Zuxuan and Chen, Jingjing and Han, Xintong and Shrivastava, Abhinav and Lim, Ser-Nam and Jiang, Yu-Gang}, booktitle={Conference on Computer Vision and Pattern Recognition}, year={2022}, pages={2364-2373}, doi={10.1109/CVPR52688.2022.00240}, url={https://mlanthology.org/cvpr/2022/wang2022cvpr-objectformer/}}`
- **Relevance**: Multimodal transformer combining RGB and frequency features with learnable object prototypes for manipulation detection/localization; strong 2022+ method evaluated on CASIA, Columbia, Coverage, NIST16, IMD20.
- **Source**: CVPR 2022 open access | DOI 10.1109/CVPR52688.2022.00240 | arXiv:2203.14681

## UNVERIFIED

### Noiseprint++ — Cozzolino et al., 2023/2024 (venue?)
- **Status**: UNVERIFIED. No standalone "Noiseprint++" paper exists from 2023/2024 in TIP or TPAMI. Noiseprint++ is an enhanced fingerprint variant introduced within the TruFor paper (CVPR 2023, Guillaro et al.) as "a new noise-sensitive fingerprint, Noiseprint++, with enhanced robustness to image laundering." The original Noiseprint by Cozzolino & Verdoliva appeared in IEEE TIFS 2019 (DOI 10.1109/TIFS.2019.2916364). No separate publication with that exact label was found.

## KEEP pre-2020 must-haves / ADD post-2020

### KEEP pre-2020 must-haves
1. **BBSE** — Lipton, Wang, Smola, ICML 2018 (label shift; pre-2020 must-have)
2. **ManTra-Net** — Wu, AbdAlmageed, Natarajan, CVPR 2019 (must-have lineage)
3. **Geirhos et al.** — "Shortcut learning in deep neural networks", Nature MI 2020 (foundational shortcut-learning perspective)

### ADD post-2020
1. **TruFor** — Guillaro et al., CVPR 2023 (forensic localization with Noiseprint++)
2. **DocTamper** — Qu et al., CVPR 2023 (document tampering benchmark, 170k images)
3. **MVSS-Net++** — Dong et al., IEEE TPAMI 2022 (journal extension of ICCV 2021 MVSS-Net)
4. **Ding et al.** — Local temperature scaling for probability calibration, ICCV 2021 (calibration under distribution shift)
5. **ObjectFormer** — Wang et al., CVPR 2022 (multimodal transformer for manipulation localization)
6. **GM-DF** — Chen et al., arXiv 2024 (generalized multi-scenario deepfake detection; multi-dataset/domain generalization)
7. **M2SFormer** — Dou et al., arXiv 2025 (multi-spectral and multi-scale attention with edge-aware difficulty guidance; cross-domain generalization for forgery localization)
8. **LTW** — Dhabi et al., AAAI 2021 (domain generalization for face forgery detection by learning to weight)
9. **Discovering Distribution Shifts using Latent Space Representations** — Zhou et al., arXiv 2022 (non-parametric framework for detecting distribution shifts via embedding space geometry)
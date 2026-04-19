Local map net>

<img src="projects/local_map_history.png">

- Local map Construction Methods with SD map: A Novel Survey and Taxonomy : https://arxiv.org/html/2409.02415v2
- P-MapNet: Far-seeing Map ConstructorEnhanced by both SDMap and HDMap Priors : https://jike5.github.io/P-MapNet/
- SMERF: Augmenting Lane Perception and Topology Understanding with Standard Definition Navigation Maps : https://github.com/NVlabs/SMERF
- MAPTR: [Vectorized map ouput] STRUCTURED MODELING AND LEARNING FOR ONLINE VECTORIZED HD MAP CONSTRUCTION : https://arxiv.org/pdf/2208.14437
    - polyline query like map geometry, bipartite matching by hungarian algo
- HRMapNet : [Vectorized map ouput/Rasterized map input] Enhancing Vectorized Map Perception with Historical Rasterized Maps : https://github.com/HXMap/HRMapNet
- GlobalMapNet: [Temporal] An Online Framework for Vectorized Global HD Map Construction : https://arxiv.org/pdf/2409.10063
- VectorMapNet: [Vectorized map ouput] End-to-end Vectorized HD Map Learning : https://arxiv.org/pdf/2206.08920
    - Autoregressive Mechanism: It predicts map elements (lanes, boundaries, etc.) one by one. The key feature is that the input to the Transformer decoder for predicting the $i$-th element includes the encoded representations of the previously predicted $i-1$ elements.
- MAPQR : [Vectorized map ouput/Point query]Leveraging Enhanced Queries of Point Sets for Vectorized Map Construction : https://github.com/HXMap/MapQR?tab=readme-ov-file
- SEPT: [Vector and Raster SD map input] Standard-Definition Map Enhanced Scene Perception and Topology Reasoning for Autonomous Driving : https://arxiv.org/pdf/2505.12246
- BezierFormer: [Vision] A Unified Architecture for 2D and 3D Lane Detection : https://arxiv.org/pdf/2404.16304
- DETR transformer : [Vision] End-to-End Object Detection with Transformers : https://arxiv.org/abs/2005.12872
- Deformable DETR: [Vision] Deformable Transformers for End-to-End Object Detection : https://arxiv.org/abs/2010.04159
- CurveFormer++: [Curve/Temporal] 3D Lane Detection by Curve Propagation with Temporal Curve Queries and Attention
- PolyNet : PolyNet: Polynomial Neural Network for 3D Shape Recognition with PolyShape Representation https://arxiv.org/abs/2110.07882 https://github.com/myavartanoo/PolyNet_PyTorch
- MapTR2 : 
    - Inter/intra sperated branch to attend for fast convergence, auxiliary BEV/PV segmentation loss/ auxiliary one to many set prediction loss
- LaneSegNet : MAP LEARNING WITH LANE SEGMENT PERCEPTION FOR AUTONOMOUS DRIVING : Lanelet based https://github.com/OpenDriveLab/LaneSegNet
- TopoLogic: An Interpretable Pipeline for Lane Topology Reasoning on Driving Scenes : https://github.com/Franpin/TopoLogic?tab=readme-ov-file
- SDTagNet : 
- OpenVLA : 
- prioMapNet : use reference points to query for decoder of mapping
- BevFormer : BEV feature for temporal/spatial data

- BERT : Language model

E2E Archi>

<img src="projects/e2e_archi.png">
<img src="projects/e2e_archtectures.png">

- Uni-AD : https://arxiv.org/pdf/2212.10156
- PARA-Drive: Parallelized Architecture for Real-time Autonomous Driving : https://openaccess.thecvf.com/content/CVPR2024/papers/Weng_PARA-Drive_Parallelized_Architecture_for_Real-time_Autonomous_Driving_CVPR_2024_paper.pdf

model design>
- model flow for training stages
<img src="projects\eagle\model_design.svg">


<dataset>
- nagative dataset : do not contain target class like parking lot for road driving model
- hard dataset : 
    - hard positive : in class, but ambigous blurry image
    - hard negative : not in class, still wrong prediction
Symmetric Cross Entropy (SCE) is a specialized loss function primarily used in machine learning when dealing with noisy labels (label errors) in the training data, particularly in large datasets where manual cleaning is impractical.


<augmentation>
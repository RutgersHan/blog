title: "Notes for <Part-based R-CNNs for Fine-grained Category Detection>"
date: 2015-04-13 22:11:56
categories: [Research]
tags: [Notes, Deep Learning, Object Detection]
---
[paper](http://www.cs.berkeley.edu/~nzhang/papers/eccv14_part.pdf)
[论文笔记](http://zhangliliang.com/2014/11/10/paper-note-part-rcnn/)

###Overview:
This paper is a nextention to R-CNN and it is mostly for Fine-grained Category Detection. It learns both object and part detectors and it also enforces learned geometric constraints between them.
* Region Proposals: selective search(same as R-CNN)
* find the bounding box for both the objects and parts, add the geometic constraints to them

![System Overview](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-14 at 8.27.18 PM.png)
<!--more-->

###Part-based R-CNNs

* Deal with subtle appearance differences for Fine-grained Category Detection.
* In training , assume to have the bounding box for the full object and the semantic parts(where regions with ≥ 0.7 overlap with a ground truth object or part bounding box are labeled as positives for that object or part, and regions with ≤ 0.3 overlap with any ground truth region are labeled as negatives)
* In testing, for each region proposal window we compute scores from all root and part SVMs, and then add geomatric constraints

训练一个有part的RCNN，满足了两个条件：

* proposal里面覆盖了95%的part
* part有标记信息，可进行有监督学习

但是其实具体来说，跟RCNN框架还是有点差别的：
都用ImageNet的模型做pretrain，但这里finetuning一个200的分类器（这里对应了200种鸟）。而RCNN是tuning200类+1背景的

####Geometric constraints
![Geometric constraints](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-14 at 8.45.00 PM.png)
The $\Delta X$ is a constraint function, $d(x_i)$ is the score for each SVM
Two kinds of constrant function
*  Box constraints:  all the part windows inside the object window, at most ten pixels can be outside (首先给出第一个约束是part的bbox应该要几乎都在root的bbox里面（最多只有10个像素能在外面）).
![Box constraints](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-14 at 9.06.23 PM.png)
*  Geometric constraints:给出一个更强的约束是，part相对于root应该是有一个“默认”的位置的（比如鸟头应该在上方等），于是有基于第一个约束有了第二个约束
![Geomatric constrints](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-14 at 9.05.51 PM.png)
其中$\delta i$ 代表对part的位置的某种建模方式，文中提到了两种，分别是基于多高斯和最近邻的
![part modeling](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-14 at 9.05.57 PM.png)
The following is the near neighbour constraint:
![](http://7xikhz.com1.z0.glb.clouddn.com/next.png)

####Fine-grained categorization
* Using ImageNet pre-trained CNN, fined tune using CUB images(change the 1000way to 200 bird classes)
* fine-tuning learning rates: initializing the global rate to a tenth of the initial ImageNet learning rate and dropping it by a factor of 10 throughout training, but with a learning rate in the new fc8 layer of 10 times the global learning
rate.
* For the whole object bounding box and each of the part bounding boxes, we independently finetune the ImageNet pre-trained CNN for classification on ground truth crops of each region warped to the 227 × 227 network input size, always with 16 pixels on each edge of the input serving as context as in RCNN
* For a new test image, we apply the whole and part detectors with the geometric scoring function to get detected part locations and
use the features (concatenate part and whole features)for prediction. If part i is not detected, the corresponding feature is set to 0. 


####Evaluation

* Data: Caltech-UCSD bird dataset(CUB200-2011), 1w+ images,200 bird classes(in average,for each class, it is about 30 images). It has the bounding box for whole object and 15 parts. But the author only uses head and body parts. 
* Toolbox used: Caffe
* Use fc6 to train R-CNN object and part detectors as well as image representation for classification. For $\delta^{NP}$ , nearest neighbors are computed using pool5 and cosine distance metri

Detailed can be seen [here](http://zhangliliang.com/2014/11/10/paper-note-part-rcnn/)(with images)
> * 在给出bbox时候，是state-of-the-art（其中Oracle82%是因为测试阶段也用了bbox还有part标注。 在不给出bbox的时候， 因为太难基本没有其他人做， 这个方法依然是state-of-the-art。  
> * 去掉part特征时候的结果，也就是只利用空间信息来定位object，依然有提高，说明了空间约束有助于提高结果。
> * selective search给出的proposal的召回率。在ol>0.5时候，基本都能够召回。在ol>0.7时候，召回率会大幅度下降 所以作者认为目前方法的bottleneck在于proposal方法.
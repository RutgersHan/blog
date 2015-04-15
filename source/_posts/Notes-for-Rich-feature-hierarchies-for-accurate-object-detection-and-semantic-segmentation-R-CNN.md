title: "Notes for <Rich feature hierarchies for accurate object detection and semantic segmentation(R-CNN)>"
date: 2015-04-14 11:36:49
categories: Research
tags: [Notes, Deep Learning, Object Detection]
---


###The resources
[Code](https://github.com/rbgirshick/rcnn)
[Slides](http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf)
[Paper](http://arxiv.org/abs/1311.2524)
[论文笔记1](http://zhangliliang.com/2014/07/23/paper-note-rcnn/)

![system overview](http://7xikhz.com1.z0.glb.clouddn.com/owzgVal.png)

<!--more-->
###核心框架：
* selective search(region proposal) to replace sliding window， 2K region proposal
* For each region, warp the using the fc7(Alex Net, caffe implementation) as the feature
* Warp the bounding box and some image context, to fixed $227 * 227$ (match Alex Net)
* Use the fc7 feature to train K one-verse-all linear SVM for each class, and then using non-maxmum suppression for each region
* add bounding-box regression to better locate the objects


### Feature Extraction

也就是使用CNN，具体来说是AlexNet来提取特征，摘掉了最后一层softmax，利用前面5个卷积层和2个全连接层来提取特征，得到一个4096维的特征。一个值得注意的细节是如何将region缩放到CNN需要的 [227,227]  ,作者是直接忽略aspect ratio之间缩放到 [227*227]  (含一个16宽度的边框), 这样的好处是稍微扩大region，将背景也包括进来来提供先验信息.

### Training
* Supervised pre-training:  Imagenet 120w classification data (No bounding box, since each image itself only contains one object)
* Domain-specific fine-tuning: 将上面训练出来的模型用到new task(dection)和new domain(warped region proposals)上，作者将最后一个softmax从1000路输出替换成了N+1路输出（N个类别+1背景）。然后将IoU大于50%的region当成正样本，否则是负样本。将fine-tuning学习率设置成pre-train模型中的1/10（目的是为了既能学到新东西但是不会完全否定旧的东西）。batch为128，其中正负样本比例是1:3,因为正样本本来就少
* Object category classifiers: 选择SVM对每一类都做一个二分类，在选择样本的时候，区分正负样本的IoU取多少很重要，取IoU=0.5时候，mAP下降5%，取IoU=0，mAP下降4%，作者最后取了0.3. 用hard negative mining得到负样本

###Visualzation, ablation, and modes of error

* Visualzing learned feature: 核心思想是在pool5中一个神经元对应回去原图的[227,227]中的[195, 195]个像素
可视化的方法是将10M的region在训练好的网络中FP，然后看某个pool5中特定的神经元的激活程度并且给一个rank

* Performance layer-by-layer, without fine tuning: pool5，fc6，fc7的特征做SVM分类， 出来的效果都差不多。 作者得到的结论是： CNN的特征表达能力大部分是在卷积层
* Performance lyaer-by-layer, with fine tuning: pool5经过finetuning之后，mAP的提高不明显，所以卷积层提取出来的特征是具有普遍性的，而fc7经过finetuning后得到很大的提升，说明finetuning的效果主要是在全连接层上。
* Comparision to recent feature learning methods: 这里主要说明CNN的特征学习能力比其他方法要好

###Network architectures, Detection error analysis, Bounding-box regression
* Network architectures: Other achitectures, like OxfordNet can boost the recognition rate, but take more time to train
* Bounding-box Regression: To reduce localization errors, train a linear regression model to predict a new detection window given the pool5 features for a selective search region proposal
* Detection error analysis: 用了一个工具来分析错误

####Semantic segmentation
* worked within O2P(leading semantic segmentation system) to  compare
* CNN features for segmentions: 
* three strategies:
 * bouding box
 * foregound, backgound is mean(which would 0 after normilizaiton)
 * bounding box & foreground.



### Appendix

####Positive vs. Negative examples and softmax
**Why having differernt definition for CNN and SVM?** 
Fine tune data is limited, so by define different, we can have more fine tune data.
**Why not using softmax, but a extra SVM?**
The location for fine tune data is not very accurate. 


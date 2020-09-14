---
layout:     post
title:      快且省钱的style_transfer算法
subtitle:   Simple and fast Style Transfer Algorithm
date:       2020-09-14
author:     SC
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - Vision
---

## What is Style Transfer

Style Transfer, 风格转移，即把图片A变成图片B的风格，但保持图片A的内容不变，举个栗子，假设下图左是你自己的作品，中图是梵高的星空，右图则是风格转移算法的结果。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913123707105.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NjcmlwdGVkZHJlYW1z,size_16,color_FFFFFF,t_70#pic_center)

## How to do it?

风格转移的实现方法很多，复杂的算法使用到GANs，其结果会非常的精美，但会需要（十分）强大的算力（和钱）。一些改进的方法能够显著将GANs类模型的计算成本控制在合理范围，例如将一张图片切割成很多小的方块，对每个小块进行风格转移，最后再将它们拼起来。

但Gatys et al在"A Neural Algorithm of Artistic Style"这篇文章中提出了一种更简单更快速的方法——不需要用到the full GANs、只需要借助pre-trained image classification CNN，即可完成风格转换。理解其原理之前，必须先简单提一下CNN是如何完成image classification的，我们知道CNN classifier是由很多很多的CNN神经网络组成的，不同的CNN的kernel size不一样，较浅的layers的kernel size都比较小，这样可以抓取一些细微的特征，例如动物的毛发、金属表面的质感等等，较深的layers的kernel size会逐渐增大，这样可以抓取一些更完整的特征，例如眼睛、尾巴、车轮等等。可以发现，刚刚描述的较浅的layers描述的接近于一张图片的styles——油画画笔的texture、色块的分布等等，而较深的layers则描述的更多的是一张图片的内容——眼睛、尾巴、车轮等等。于是一个简单的风格转移模型就是利用CNN classifier不同的layer抓取的内容不一这个特点来实现快速、低成本的风格转移。

## 具体实现方法
需要利用的工具：VGG19(in Tensorflow), Python
（我们甚至不需要GPU）

### Flow
1. 将input图片feed进一个pre-trained image architecture, like VGG or ResNet.
2. 计算**Loss**：

	1）Content：把content image的content layer, <img src="https://render.githubusercontent.com/render/math?math=F^{l} \in \mathcal{R}^{m ,n}">提取出来，将content layer变平成一个向量<img src="https://render.githubusercontent.com/render/math?math=\mathbf{f}^{l} \in \mathcal{R}^{m*n,1}">；将生成的图片<img src="https://render.githubusercontent.com/render/math?math=P^{l} \in \mathcal{R}^{m ,n}">也做同样的变平处理成一个向量<img src="https://render.githubusercontent.com/render/math?math=$\mathbf{p}^{l} \in \mathcal{R}^{m*n,1}">，那么content loss就是f和p这两个向量的Euclidean Norm：
	
	<img src="https://render.githubusercontent.com/render/math?math=L_{content}(\mathbf{p},\mathbf{f},l)=\frac{1}{2}\sum_{i,j}(F_{i,j}^l-P_{i,j}^l)^2">
	
	2）Style Loss：两个向量的点乘可以表现这两个向量有多相似（即同方向），当我们把两个flattened feature vector点乘时，这个乘积也代表了某个feature vector在某个方向上是否相似，需要注意的是，由于图形这个张量被flatten成一个向量，故点乘并不能展示spatial信息，而只能描述更加细微的texture。
	
	<img src="https://render.githubusercontent.com/render/math?math=L_{style}=G^l_{i,j}=\sum_k F^l_{i,k}F^l_{j,k}">
	其中G代表Gram matrix，即两个向量的outer product组成的矩阵
	
	3）A somewhat intuitive explaination w.r.t. why use difference in content loss and dot product in style loss：The content feature extracted from VGG is like greyscaled sketches of the content image. 即F_{content}可以想象成黑白的勾勒content的线条，所以当我们想比较生成的图片是否具备F_{content}所代表的content，我们只需要检查某个pixel上，是否存在一个相似的pixel的值。而style的话是一种local texture，可以想象在一副油画中，笔刷刷出来的质感，或者像梵高的星空这幅画，你会看到大面积的螺旋状的gradient，所以比起是否或高或低的像素值，我更在意这些像素它们变化的方向是否和style image一致，而这种方向可以很好的被dot product给capture。
	
3. 计算**Gradients w.r.t. input image pixels P**。注意这个gradients不会被back propagate到VGG的weights上，而是back propagate给input图片，VGG的weights全程保持不变。

### Implementation
首先我们load content image and style image，注意这里用的VGG，VGG的input是224X224，所以需要把它们都裁成224X224。
```python
content_image = content #load your content image here

style_image = style #load your style image here
```

我们load VGG19 model from Keras
```python
import tensorflow as tf
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
```
可以看一下VGG19里面有哪些layers
```python
print([layer.name for layer in vgg.layers])

block1_conv1
block1_conv2
block1_pool
block2_conv1
block2_conv2
block2_pool
block3_conv1
block3_conv2
block3_conv3
block3_conv4
block3_pool
block4_conv1
block4_conv2
block4_conv3
block4_conv4
block4_pool
block5_conv1
block5_conv2
block5_conv3
block5_conv4
block5_pool
```
虽然看起来很普通，但这一步就是奇迹发生的时刻，我们从VGG19里pick了content layer和style layer。（try picking different layers to represent content and loss, and see what you get）
```python
content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
		'block2_conv1',
		'block3_conv1',
		'block4_conv1',
		'block5_conv1']
num_content_layers = len(conten_layers)
num_style_layers = len(style_layers)
```
写一个function把Layers给wrap up一下
```python
def vgg_layers(layer_names):
	"""creates a vgg model that returns a list of intermediate output values"""
	vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
	#锁住VGG的参数不变，因为我们想训练的不是参数，而是input
	
	vgg.trainable = False 
	#vgg.get_layer(name).output 是一个tensor placeholder，下面vgg.input同理，因为VGG19的input必须是224X224，所以vgg.input也是这个size的tensor placeholder
	
	outputs = [vgg.get_layer(name).output for layer in layer_names] 
	model = tf.keras.Model([vgg.input],outputs)
	return model
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)
```
利用Gram matrix计算style loss的function，这个就是前文提到的feature vector的dot product。我们选取了5个CNN block的第一层CNN作为style feature vector，计算这些feature vector和其它feature vectors（包括它们自己）的dot product，并组成一个(num_of_feature_vector *num_of_feature_vector)的矩阵，这个过程可以用gram_matrix来实现：

<img src="https://render.githubusercontent.com/render/math?math=G^l_{cd}=\frac{\sum_{ij}F^l_{ijc}(x)F^l_{ijd}(x)}{IJ}">

```python
def gram_matrix(input_tensor):
	#(b,i,j,c)=(batch_size, ith row, jth col, cth color channel)
	
	result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
	input_shape = tf.shape(input_tensor)
	# 第一个dim[0]是batch_size，所以IJ=dim[1]*dim[2]
	
	num_locations = tf.cast(input_shape[1]*input_shape[2],tf.float32)
	return result/num_locations
```

Wrap loss into the model:

```python
class StyleContentModel(tf.keras.models.Model):
	def __init__(self, style_layers, content_layers):
		super(StyleContentModel, self).__init__()
		self.vgg = vgg_layers(style_layers+content_layers)
		self.style_layers = style_layers
		self.content_layers = content_layers
		self.num_style_layers = len(style_layers)
		self.vgg.trainable = False
	
	def call(self, inputs):
		"""float input form [0,1]"""
		inputs = input*255.0
		preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
		outputs = self.vgg(preprocessed_input)
		style_outputs, content_outputs = (outputs[:self.num_style_layers],
									outputs[self.num_style_layers:])
		style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
		content_dict = {content_name:value for content_name,value in zip(self.content_layers, content_outputs)}
		style_dict = {style_name:value for style_name,value in zip(self.style_layers, style_outputs)}
		return {'content':content_dict, 'style':style_dict}

extractor = StyleContentModel(style_layers, content_layers)
```
计算gradient，开始backpropagate：
```python
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# initialize a placeholder tensor, with the same dimension of content image
image = tf.Variable(content_image)

#输入进模型的数据都*255了，所以这里还原成[0,1]

def clip_0_1(image):
	return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

#定义一个optimizer
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# 定义style和content的loss在总loss各占多少比重

style_weight, content_weight=1e-2, 1e4
def style_content_loss(outputs):
	style_outputs = outputs['style']
	content_outputs = outputs['content']
	style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
	style_loss *= style_weight/num_style_layers
	content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
	content_loss *= content_weight/num_content_layers
	return style_loss + content_loss
```
到这里就是全部的setup了，后续就是一个tf.session开始训练，感兴趣的可以去链接2继续看看，这里就不继续copy&paste了（不然都没办法tag成原创了hhh），所以就到此为止了，感谢阅读。


参考：

https://arxiv.org/abs/1508.06576

https://www.tensorflow.org/tutorials/generative/style_transfer

---
layout:     post
title:      RNN中BPTT和TBPTT的笔记和理解
subtitle:   All about backpropagation through time
date:       2020-09-22
author:     SC
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - NLP
---

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200922214400315.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NjcmlwdGVkZHJlYW1z,size_16,color_FFFFFF,t_70#pic_center)
![](https://img-blog.csdnimg.cn/20200922214414955.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NjcmlwdGVkZHJlYW1z,size_16,color_FFFFFF,t_70#pic_center)
![](https://img-blog.csdnimg.cn/20200922214429707.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NjcmlwdGVkZHJlYW1z,size_16,color_FFFFFF,t_70#pic_center)
![](https://img-blog.csdnimg.cn/20200922214440459.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NjcmlwdGVkZHJlYW1z,size_16,color_FFFFFF,t_70#pic_center)

参考：
https://www.youtube.com/watch?v=_M-nDb0MIa4&list=PLyqSpQzTE6M-SISTunGRBRiZk7opYBf_K

---
layout:     post
title:      百度统计trackingID
subtitle:   获取追踪网址的trackingID
date:       2020-09-06
author:     SC
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - Analytics
---

## 获取百度统计Baidu Analytics的Tracking ID的方法

最近在尝试做自己的portfolio/blog主页，用的是这个[Tutorial](https://github.com/qiubaiying/qiubaiying.github.io)。其中在编辑```_config.yml```文件的```Analytics```的section时，可以填写百度统计（Baidu Analytics）的tracking ID，作者一句话的描述比较简略，对于像我一样的初学者来说会感到困惑，这里是一个小小的步骤说明。

### 具体步骤

1. 在Baidu Analytics注册账号，并进入到这个页面
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200906195548418.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NjcmlwdGVkZHJlYW1z,size_16,color_FFFFFF,t_70#pic_center)
2. 点击“新增网站”，填写你的blog域名，注意粘贴的域名要完整地带上协议（如https://gnehc-uhs.github.io/），点击确定。
3. 若出现如下code block，便可以直接截取```hm.src```这个参数的```js?```后面这串ID，即为你帐户下该网址的Tracking ID啦，Voilà!
```javascript
<script>
var _hmt = _hmt || [];
(function() {
  var hm = document.createElement("script");
  hm.src = "https://hm.baidu.com/hm.js?d472767XXXXXX7173fXXXXXX5be49d94";
  var s = document.getElementsByTagName("script")[0]; 
  s.parentNode.insertBefore(hm, s);
})();
</script>
```

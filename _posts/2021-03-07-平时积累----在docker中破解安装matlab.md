---
layout:     post
title:      平时积累
subtitle:   docker中破解安装matlab
date:       2021-03-07
author:     Shawn
header-img: img/post-bg-miui6.jpg
catalog: true
tags:
    - 平时积累










---

# docker中破解安装matlab

## 一、拉取ubuntu镜像并启动容器

`docker pull ubuntu`

启动docker时要使用特权模式，否则后续无法挂载matlab镜像

`docker run -itd -r ~/matlab-workspace:/workspace --privileged=true ubuntu /bin/bash`

## 二、准备安装包及后续安装破解流程

安装包资源如下，matlab版本为R2017b

链接：[https://pan.baidu.com/s/1wijZCXIWsNXgz0yYYBXHnQ](https://pan.baidu.com/s/1wijZCXIWsNXgz0yYYBXHnQ)
密码：e8b2

后续安装及破解步骤参考[https://blog.csdn.net/sjjbsj/article/details/102583432](https://blog.csdn.net/sjjbsj/article/details/102583432)

## 三、遇到的问题

- 按照上述教程安装破解后，此时启动matlab会提示：

`error while loading shared libraries: libX11.so.6: cannot open shared object file: No such file or directory.`

执行以下操作：

`apt-get install gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates fonts-liberation libappindicator1 libnss3 lsb-release xdg-utils wget`

- 此外还有一个小问题便是umount镜像文件时，会提示：

`umount: /data/software/matlab2017b/temp: target is busy.`

安装fuser即可解决：升级apt-get后，执行`apt-get install psmisc`

然后执行 `fuser -mk '%挂载路径'`，之后即可正常umount






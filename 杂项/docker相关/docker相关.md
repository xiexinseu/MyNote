# docker相关
[TOC]
[docker相关](docker_practice.pdf)

```
docker run -v /data:/data -t -i ubuntu:18.04 /bin/bash
```
可以将本地的/data目录映射到/data
`cd /data`cd到data目录
`mkdir xiexin`在data文件夹下新建xiexin目录
`chmod 777 xiexin`修改xiexin这个目录的权限
这样以后就有xiexin这个目录的读写权限了

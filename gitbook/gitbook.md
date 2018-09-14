# gitbook

[TOC]



## gitbook安装：

[gitbook安装：](https://blog.csdn.net/feosun/article/details/72806825)


## gitbook插件：

* 显示或隐藏toc插件

[显示或隐藏toc插件](https://cnodejs.org/topic/575229332420978970d4a5f0)

在book.js或book.json
```json
{
    "plugins": ["toc2"],
    "pluginsConfig": {
        "toc2": {
            "addClass": true,
            "className": "toc"
        }
    }
}
```


\$ gitbook install
在markdown文件加入 \<!-- toc -->，当编译的时候回自动在这个位置增加toc

打开页面，默认会显示toc，回车，显示toc，再回车，隐藏toc。当然你也可以直接h隐藏


* gitbook 常用插件

[插件网址如下：](http://www.css88.com/archives/6622)


```json
// book.json
{
  "title": "Webpack 中文指南",
  "description": "Webpack 是当下最热门的前端资源模块化管理和打包工具，本书大部分内容翻译自 Webpack 官网。",
  "language": "zh",
  "plugins": [
    "disqus",
    "github",
    "editlink",
    "prism",
    "-highlight",
    "baidu",
    "splitter",
    "sitemap"
  ],
  "pluginsConfig": {
    "disqus": {
      "shortName": "webpack-handbook"
    },
    "github": {
      "url": "https://github.com/zhaoda/webpack-handbook"
    },
    "editlink": {
      "base": "https://github.com/zhaoda/webpack-handbook/blob/master/content",
      "label": "编辑本页"
    },
    "baidu": {
        "token": "a9787f0ab45d5e237bab522431d0a7ec"
    },
    "sitemap": {
        "hostname": "http://zhaoda.net/"
    }
  }
}
```


* book-summary-scroll-position-saver

https://plugins.gitbook.com/plugin/book-summary-scroll-position-saver

## gitbook导出pdf

安装calibre，地址如下：

https://calibre-ebook.com/download_linux

linux使用`sudo -v && wget -nv -O- https://download.calibre-ebook.com/linux-installer.sh | sudo sh /dev/stdin`
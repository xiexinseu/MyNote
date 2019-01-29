# vscode调试c++显示图片

1. VsCode安装CodeLLDB插件
2. 系统安装lldb：sudo apt install lldb-6.0
3. 注意Python要通过pip安装numpy、matplotlib，并且注意lldb与系统中默认的python可能不是同一个，可以在下面的Python脚本中添加代码先查看python的版本，用pip给对应版本安装需要的库
4. 写C++代码：

```C++
// mandelbrot.cpp
#include <cstdio>
#include <complex>

void mandelbrot(int image[], int xdim, int ydim, int max_iter) {
    for (int y = 0; y < ydim; ++y) {
        for (int x = 0; x < xdim; ++x) { // <<<<< Breakpoint here
            std::complex<float> xy(-2.05 + x * 3.0 / xdim, -1.5 + y * 3.0 / ydim);
            std::complex<float> z(0, 0);
            int count = max_iter;
            for (int i = 0; i < max_iter; ++i) {
                z = z * z + xy;
                if (std::abs(z) >= 2) {
                    count = i;
                    break;
                }
            }
            image[y * xdim + x] = count;
        }
    }
}

int main() {
    const int xdim = 500;
    const int ydim = 500;
    const int max_iter = 100;
    int image[xdim * ydim] = {0};
    mandelbrot(image, xdim, ydim, max_iter);
    for (int y = 0; y < ydim; y += 10) {
        for (int x = 0; x < xdim; x += 5) {
            putchar(image[y * xdim + x] < max_iter ? '.' : '#');
        }
        putchar('\n');
    }
    return 0;
}
```

4. 编译

```bash
c++ -g mandelbrot.cpp -o mandelbrot
```

5. 写画图的python脚本（注意16行原github上的有bug）

```python
# debugvis.py
import io
import lldb
import debugger
import base64
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Use this function as a replacement for Matplotlib's show()
def show():
    image_bytes = io.BytesIO()
    plt.savefig(image_bytes, format='png', bbox_inches='tight')
    document = '<html><img src="data:data:image/png;base64,%s"></html>' % base64.b64encode(image_bytes.getvalue())
    debugger.display_html(html = document, title='Pretty Plot', position=2)

# First parameter is an array, or a pointer to an array, of (xdim * ydim) elements
def plot_image(image, xdim, ydim, cmap='nipy_spectral_r'):
    image = debugger.unwrap(image)
    if image.TypeIsPointerType():
        image_addr = image.GetValueAsUnsigned()
    else:
        image_addr = image.AddressOf().GetValueAsUnsigned()
    data = lldb.process.ReadMemory(image_addr, int(xdim * ydim) * 4, lldb.SBError())
    data = np.frombuffer(data, dtype=np.int32).reshape((ydim,xdim))
    plt.imshow(data, cmap=cmap, interpolation='nearest')
    show()

```

6. 设置debug环境

```json
// launch.json
    {
        "name": "Launch Mandelbrot",
        "type": "lldb",
        "request": "launch",
        "program": "${workspaceRoot}/mandelbrot",
        "initCommands": [
            "command script import ${workspaceRoot}/debugvis.py" // <<<<< This is the important bit
        ]
    }
```

7. 设置条件断点

```bash
/py debugvis.plot_image($image, $xdim, $ydim) if $y % 50 == 0 else False
```

8. F5运行，即可将内存中的数据显示为图像
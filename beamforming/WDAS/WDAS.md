# WDAS

<!-- toc -->

## 参考文献or网址
https://github.com/xanguera/BeamformIt.git

## 环境配置

1. 安装Libsndfile：`sudo apt-get install libsndfile1-dev`
2. make
`cmake .`
`make`
也可以直接`Ctrl+shift+p`->`CMake: Clean rebuild`
如果想要debug，注意修改 CMakeLists.txt中的编译选项
3. 运行
`bash ./do_beamforming.sh '/media/xiexin/98BC1B8BBC1B62D4/work/ProFromGitHub/BeamformIt/test_wav/test1' 'F02_011C021A_BUS.CH1'`
后面2个参数为指定音频文件夹和文件。
4. debug
* 可以`Ctrl+shift+p->cmake debug target`，有个问题无法接受参数输入。
* 也可以调试界面修改launch.json：
    修改`"program": "${workspaceFolder}/BeamformIt",`
    再F5启动调试。
    修改
    ```
    "args": [
        "--scroll_size" 250 \
        "--window_size" 500 \
        "-nbest_amount" 4 \
        "--do_noise_threshold" 1 \
        "--noise_percent" 10 \
        "--trans_weight_multi" 25 \
        "--trans_weight_nbest" 25 \
        "--print_features" 1 \
        "--do_avoid_bad_frames" 1 \
        "--do_compute_reference" 1 \
        "--do_use_uem_file" 0 \
        "--do_adapt_weights" 1 \
        "--do_write_sph_files" 1 \
        "--channels_file" "./output/F02_011C021A_BUS.CH1/channels_file" \
        "--show_id" "F02_011C021A_BUS.CH1" \
        "--result_dir" "./output/F02_011C021A_BUS.CH1"
    ],
    ````
    可以接受输入参数

5. 通过注释do_beamforming.sh的最后部分，不调用函数，得到函数的输入参数
```
# ./BeamformIt \
#     --scroll_size 250 \
#     --window_size 500 \
#     --nbest_amount 4 \
#     --do_noise_threshold 1 \
#     --noise_percent 10 \
#     --trans_weight_multi 25 \
#     --trans_weight_nbest 25 \
#     --print_features 1 \
#     --do_avoid_bad_frames 1 \
#     --do_compute_reference 1 \
#     --do_use_uem_file 0 \
#     --do_adapt_weights 1 \
#     --do_write_sph_files 1 \
#     --channels_file ./output/${outName}/channels_file \
#     --show_id ${outName} \
#     --result_dir ./output/${outName}
```
    通过echo返回${outName}需要的输入参数。

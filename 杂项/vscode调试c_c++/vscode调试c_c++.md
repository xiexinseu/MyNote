# vscode调试c_c++

注意需要安装clang： sudo apt-get install clang 

1. Ctrl+shift+p 选择CMake:Clean rebuild 

2.右下角 选择Quickstart a new CMake project 

3. 顶部输入工程名，假设是test_test 

4. 顶部选择Executable，会新建一个CMakeLists.txt 

5. 删除工程中的main.cpp（如果不是自己建的） 

6. CMakeLists中add_executable中把main.cpp修改为自己的源文件 

7. 如果想要debug，CMakeLists中需要添加set(CMAKE_CXX_FLAGS_DEBUG "-g") 

8. 重新执行1，则编译完成 

9. 终端输入./build/test即可运行 

10.  源代码中设置好断点，Ctrl+shift+p 选择CMake:Debug Target，即可进行调试 
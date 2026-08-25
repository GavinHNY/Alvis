# Alvis

A personal AI assistance using gemma 4 in native C++

# How to build
## cuda

```
cd external\llama.cpp
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release j
```
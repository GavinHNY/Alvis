# Alvis
A personal AI assistance using gemma 4 with llama.cpp in native C++

# Stucture
## Host
- Run model



# How to build
## cuda

```
cd external\llama.cpp
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release -j
```




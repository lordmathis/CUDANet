#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>


template <typename T>
std::vector<T> load_binary(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Cannot open fixture: " + path);

    auto byte_count = static_cast<std::size_t>(file.tellg());
    if (byte_count % sizeof(T) != 0)
        throw std::runtime_error("File size not a multiple of element size: " + path);

    file.seekg(0);
    std::vector<T> data(byte_count / sizeof(T));
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(byte_count));
    return data;
}


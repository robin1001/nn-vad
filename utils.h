// Created on 2017-06-07
// Author: Binbin Zhang


#ifndef UTILS_H_ 
#define UTILS_H_

#include <stdio.h>
#include <stdlib.h>

#define DISALLOW_COPY_AND_ASSIGN(Type) \
    Type(const Type &); \
    Type& operator=(const Type &)

#define LOG(format, ...) \
    do { \
        fprintf(stderr, "LOG (%s: %s(): ,%d" format "\n", \
            __FILE__, __func__, __LINE__, ##__VA_ARGS__); \
    } while (0)

#define ERROR(format, ...) \
    do { \
        fprintf(stderr, "ERROR (%s: %s(): ,%d) " format "\n", \
            __FILE__, __func__, __LINE__, ##__VA_ARGS__); \
         exit(-1); \
    } while (0)

#endif

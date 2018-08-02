// Copyright (c) 2018 Personal (Binbin Zhang)
// Created on 2016-08-08
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef UTILS_H_
#define UTILS_H_

#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#define DISALLOW_COPY_AND_ASSIGN(Type) \
  Type(const Type &); \
  Type& operator=(const Type &)

#define LOG(format, ...) \
  do { \
    fprintf(stderr, "LOG (%s: %s(): %d) " format "\n", \
            __FILE__, __func__, __LINE__, ##__VA_ARGS__); \
  } while (0)

#define ERROR(format, ...) \
  do { \
    fprintf(stderr, "ERROR (%s: %s(): %d) " format "\n", \
            __FILE__, __func__, __LINE__, ##__VA_ARGS__); \
    exit(-1); \
  } while (0)

#define CHECK(test) \
  do { \
    if (!(test)) { \
      fprintf(stderr, "CHECK (%s: %s(): %d) %s \n", \
              __FILE__, __func__, __LINE__, #test); \
      exit(-1); \
    } \
  } while (0)

#endif  // UTILS_H_

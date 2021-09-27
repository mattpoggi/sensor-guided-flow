#!/bin/bash
gcc -O3 -fopenmp -fPIC -shared -o libguide.so guide.c

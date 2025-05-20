#include <Python.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32 // Windows-specific includes and definitions
/*
 * Windows requires specific handling for file operations:
 * - io.h provides Windows-specific file I/O functions
 * - Windows prefixes standard POSIX functions with underscore (_)
 * - O_BINARY flag is required for proper binary file handling on Windows
 * - ssize_t type needs to be defined as it's not provided by Windows
 */
#include <io.h>
#define open _open
#define close _close
#define lseek _lseek
#define read _read
#define O_BINARY _O_BINARY
typedef long ssize_t;
#else // Unix-like includes and definitions
/*
 * Unix/Linux systems:
 * - unistd.h provides standard POSIX file I/O functions
 * - No distinction between binary and text files, but O_BINARY is defined
 *   as 0 (no-op) for cross-platform compatibility
 * - This ensures the same code works on both Windows and Unix without modification
 */
#include <unistd.h>
#ifndef O_BINARY
#define O_BINARY 0
#endif
#endif

/////////////////////////////////////////////////////////////////////////////////////

long long get_file_size(const char* fn) {
    struct stat st;
    if (stat(fn, &st) == 0) {
        return (long long)st.st_size;
    }
    return -1;
}

/////////////////////////////////////////////////////////////////////////////////////

int get_array_shape(long long num_clumps, char* fn, int offset_bytes){
    printf("[get_array_shape] Called with num_clumps=%lld, fn=%s, offset_bytes=%d\n", num_clumps, fn, offset_bytes);
    long long i = 0;
    char* bytes_read = calloc(256, 1);
    long long fsize = get_file_size(fn);
    printf("[get_array_shape] File size: %lld bytes\n", fsize);

    int fd = open(fn, O_RDONLY | O_BINARY);
    if (fd == -1) {
        perror("open");
        exit(errno);
    }
    if (lseek(fd, offset_bytes, SEEK_SET) == -1) {
        perror("lseek");
        exit(errno);
    }
    printf("[get_array_shape] Seeked to offset: %d\n", offset_bytes);

    while (i < num_clumps) {
        ssize_t num_bytes_read = read(fd, bytes_read, 32);
        if (num_bytes_read == -1) {
            perror("read");
            free(bytes_read);
            close(fd);
            exit(errno);
        }
        if (!num_bytes_read) {
            printf("[get_array_shape] EOF reached at i=%lld\n", i);
            break;
        }
        ++i;    
    }
    printf("[get_array_shape] Returning i=%lld\n", i);
    return i;
}

/////////////////////////////////////////////////////////////////////////////////////

int* get_partA(long long num_clumps, char* fn, int offset_bytes) {
    printf("[get_partA] Called with num_clumps=%lld, fn=%s, offset_bytes=%d\n", num_clumps, fn, offset_bytes);
    long long fsize = get_file_size(fn);
    printf("[get_partA] File size: %lld bytes\n", fsize);
    int fd = open(fn, O_RDONLY | O_BINARY);
    if (fd == -1) {
        perror("open");
        exit(errno);
    }
    if (lseek(fd, offset_bytes, SEEK_SET) == -1) {
        perror("lseek");
        exit(errno);
    }
    printf("[get_partA] Seeked to offset: %d\n", offset_bytes);
    printf("[get_partA] Processing data: 0%%\n");

    int* partA = calloc(12*num_clumps, sizeof(int));
    char* bytes_read = calloc(256, 1);

    int i = 0, x = 0, j = 0, bits_left = 0;
    int bit_offset = 4;
    unsigned char first, second, third, mask, temp;
    char* full_num = calloc(4, sizeof(char));
    int last_percentage = 0;
    while (j < num_clumps) {
        // Calculate and print progress
        int current_percentage = (j * 100) / num_clumps;
        if (current_percentage > last_percentage) {
            printf("[get_partA] Processing data: %d%%\r", current_percentage);
            fflush(stdout);
            last_percentage = current_percentage;
        }

        if (!j || i == 31) {
            assert(bit_offset == 4);
            bit_offset = 8;
            ssize_t num_bytes_read = read(fd, bytes_read, 32);
            if (num_bytes_read == -1) {
                perror("read");
                free(bytes_read);
                free(full_num);
                close(fd);
                exit(errno);
            }
            if (!num_bytes_read) {
                printf("[get_partA] EOF reached at column %i\n", j); 
                break;
            }
            ++j; i = 0;
        }
        else {
            mask = (~0)<<(8-bit_offset);
            first = (bytes_read[i] & mask) >> (8-bit_offset);
            first |= (bytes_read[i+1] << (bit_offset));
            second = bytes_read[i+1];
            second = second >> (8 - bit_offset);
            third = bytes_read[i+2];
            second |= (bytes_read[i+2] << (bit_offset));
            third = bytes_read[i+2];
            third = third >> (8 - bit_offset);

            bits_left = 5 - bit_offset;
            if (bits_left > 0) {
                ++i;
                mask = ~((~0)<<bits_left);
                temp = mask & bytes_read[i+2];
                third |= temp << bit_offset;
                bit_offset = 8 - bits_left;
            }
            else if (bits_left < 0) {
                mask = ~(((uint32_t)~0)<<5);
                third &= mask;
                bit_offset = -1 * bits_left;
            }
            else {
                ++i;
                bit_offset = 8;
            }
            full_num[0] = first; full_num[1] = second; full_num[2] = third;
            partA[x] = *((int*)full_num);
            ++x;
            i += 2;
        }
    }

    printf("[get_partA] Processing data: 100%%\n");
    printf("[get_partA] Finished, returning partA pointer\n");
    free(bytes_read);
    free(full_num);
    close(fd);

    return partA;
}

/////////////////////////////////////////////////////////////////////////////////////

static PyObject* py_get_partA(PyObject* self, PyObject* args) {
    printf("[py_get_partA] Called\n");
    long long num_clumps;
    const char* fn;
    int offset_bytes;

    if (!PyArg_ParseTuple(args, "Lsi", &num_clumps, &fn, &offset_bytes)) {
        return NULL;
    }

    int* partA = get_partA(num_clumps, (char*)fn, offset_bytes);
    PyObject* list = PyList_New(12 * num_clumps);
    for (int i = 0; i < 12 * num_clumps; i++) {
        PyList_SetItem(list, i, PyLong_FromLong(partA[i]));
    }
    free(partA);
    printf("[py_get_partA] Returning Python list\n");
    return list;
}

/////////////////////////////////////////////////////////////////////////////////////

int* get_partB(long long num_clumps, char* fn, int offset_bytes) {
    printf("[get_partB] Called with num_clumps=%lld, fn=%s, offset_bytes=%d\n", num_clumps, fn, offset_bytes);
    long long fsize = get_file_size(fn);
    printf("[get_partB] File size: %lld bytes\n", fsize);
    int fd = open(fn, O_RDONLY | O_BINARY);
    if (fd == -1) {
        perror("open");
        exit(errno);
    }
    if (lseek(fd, offset_bytes, SEEK_SET) == -1) {
        perror("lseek");
        exit(errno);
    }
    printf("[get_partB] Seeked to offset: %d\n", offset_bytes);
    printf("[get_partB] Processing data: 0%%\n");

    int* partB = calloc(num_clumps, sizeof(int));
    char* bytes_read = calloc(256, 1);

    int x = 0, j = 0;
    unsigned char cur_num, mask;
    char* full_num = calloc(4, sizeof(char));
    mask = ~((uint32_t)(~0)<<4);
    int last_percentage = 0;
    while (j < num_clumps) {
        // Calculate and print progress
        int current_percentage = (j * 100) / num_clumps;
        if (current_percentage > last_percentage) {
            printf("[get_partB] Processing data: %d%%\r", current_percentage);
            fflush(stdout);
            last_percentage = current_percentage;
        }

        ssize_t num_bytes_read = read(fd, bytes_read, 32);
        if (num_bytes_read == -1) {
            perror("read");
            free(bytes_read);
            free(full_num);
            close(fd);
            exit(errno);
        }
        if (!num_bytes_read) {
            printf("[get_partB] EOF reached at column %i\n", j); 
            break;
        }
        ++j;
        cur_num = bytes_read[0];
        cur_num &= mask;
        partB[x] = (int)cur_num;
        ++x;
    }

    printf("[get_partB] Processing data: 100%%\n");
    printf("[get_partB] Finished, returning partB pointer\n");
    free(bytes_read);
    free(full_num);
    close(fd);

    return partB;
}

/////////////////////////////////////////////////////////////////////////////////////

static PyObject* py_get_partB(PyObject* self, PyObject* args) {
    printf("[py_get_partB] Called\n");
    long long num_clumps;
    const char* fn;
    int offset_bytes;

    if (!PyArg_ParseTuple(args, "Lsi", &num_clumps, &fn, &offset_bytes)) {
        return NULL;
    }

    int* partB = get_partB(num_clumps, (char*)fn, offset_bytes);
    PyObject* list = PyList_New(num_clumps);
    for (int i = 0; i < num_clumps; i++) {
        PyList_SetItem(list, i, PyLong_FromLong(partB[i]));
    }
    free(partB);
    printf("[py_get_partB] Returning Python list\n");
    return list;
}

/////////////////////////////////////////////////////////////////////////////////////

static PyMethodDef MyMethods[] = {
    {"getPartA", py_get_partA, METH_VARARGS, "Get part A data from file"},
    {"getPartB", py_get_partB, METH_VARARGS, "Get part B data from file"},
    {NULL, NULL, 0, NULL}
};

/////////////////////////////////////////////////////////////////////////////////////

static struct PyModuleDef mymodule = {
    PyModuleDef_HEAD_INIT,
    "philipsRfParser",
    NULL,
    -1,
    MyMethods
};

/////////////////////////////////////////////////////////////////////////////////////

PyMODINIT_FUNC PyInit_philipsRfParser(void) {
    return PyModule_Create(&mymodule);
}

/////////////////////////////////////////////////////////////////////////////////////
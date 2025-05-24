// mylibrary.h
#ifndef MY_LIBRARY_H
#define MY_LIBRARY_H

#include <stdio.h> // Required for size_t and printf
#include <string.h>
#include <ctype.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Prints an integer to the console.
 * @param num The integer to print.
 */
void print_int(int num);

/**
 * @brief Prints a float to the console.
 * @param num The float to print.
 */
void print_float(float num);

/**
 * @brief Prints a string to the console without a newline.
 * @param str The string to print.
 */
void print(const char* str);

/**
 * @brief Prints a string to the console followed by a newline.
 * @param str The string to print.
 */
void println(const char* str);

/**
 * @brief Reads a string from the command line.
 * @param result The pointer to the resulting string.
 * @param max_input_size The maximum amount of characters read.
 */
void ptr_input(char* result, int max_input_size);

/**
 * @brief Parses a string to an integer.
 * @param str The pointer to the input string.
 * @param out The output integer.
 */
int str_to_int(const char *str, int *out);

#ifdef __cplusplus
}
#endif

#endif // MY_LIBRARY_H

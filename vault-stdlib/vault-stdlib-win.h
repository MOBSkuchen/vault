// mylibrary.h
#ifndef MY_LIBRARY_H
#define MY_LIBRARY_H

#include <stdio.h> // Required for size_t and printf
#include <string.h>
#include <stdlib.h>
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

/**
 * @brief Replaces all occurrences of a specified search string with a replacement string
 * within an original string. This function allocates new memory for the result.
 * The caller is responsible for freeing the returned memory using `free()`.
 *
 * @param original A pointer to the null-terminated original string.
 * @param search A pointer to the null-terminated substring to find.
 * @param replace A pointer to the null-terminated string to replace found occurrences with.
 * @return A pointer to the new dynamically allocated string with replacements, or NULL if
 * memory allocation fails or any input string is NULL.
 */
char* replaceAll(const char* original, const char* search, const char* replace);

/**
 * @brief Reads the entire content of a file into a dynamically allocated string.
 * The caller is responsible for freeing the returned memory using `free()`.
 *
 * @param filename The path to the file to read.
 * @return A pointer to the dynamically allocated string containing the file's content,
 * or NULL if the file cannot be opened, memory allocation fails, or the file is empty.
 */
char* readFileToString(const char* filename);

#ifdef __cplusplus
}
#endif

#endif // MY_LIBRARY_H

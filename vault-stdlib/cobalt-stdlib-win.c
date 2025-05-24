// mylibrary.c
#include "vault-stdlib-win.h" // Include our own header

/**
 * @brief Prints an integer to the console.
 * @param num The integer to print.
 */
void print_int(int num) {
    printf("%d", num);
}

/**
 * @brief Prints a string to the console without a newline.
 * @param str The string to print.
 */
void print(const char* str) {
    printf("%s", str);
}

/**
 * @brief Prints a string to the console followed by a newline.
 * @param str The string to print.
 */
void println(const char* str) {
    printf("%s\n", str);
}

/**
 * @brief Reads a string from the command line.
 * @param result The pointer to the resulting string.
 * @param max_input_size The maximum amount of characters read.
 */
void ptr_input(char* result, int max_input_size) {
    if (result == NULL) {
        return;
    }

    if (fgets(result, max_input_size, stdin)) {
        size_t len = strlen(result);
        if (len > 0 && result[len - 1] == '\n') {
            result[len - 1] = '\0';
        }
    } else {
        result[0] = '\0';
    }
}

/**
 * @brief Prints a float to the console.
 * @param num The float to print.
 */
void print_float(float num) {
    printf("%f", num);
}

/**
 * @brief Parses a string to an integer.
 * @param str The pointer to the input string.
 * @param out The output integer.
 */
int str_to_int(const char *str, int *out) {
    int result = 0;
    int sign = 1;
    int seen_digit = 0;

    // Skip whitespace
    while (isspace(*str)) {
        str++;
    }

    // Handle optional sign
    if (*str == '+' || *str == '-') {
        if (*str == '-') {
            sign = -1;
        }
        str++;
    }

    // Convert digits
    while (isdigit(*str)) {
        seen_digit = 1;
        result = result * 10 + (*str - '0');
        str++;
    }

    // Check if no digits were found or invalid trailing characters
    if (!seen_digit || (*str != '\0' && !isspace(*str))) {
        return 1; // Failure
    }

    *out = sign * result;
    return 0; // Success
}
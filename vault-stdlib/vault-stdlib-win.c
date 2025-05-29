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
char* replaceAll(const char* original, const char* search, const char* replace) {
    char* result;       // Pointer to the new string that will hold the result
    char* current_pos;  // Pointer to iterate through the original string
    char* found_pos;    // Pointer to the start of the found 'search' substring
    int count = 0;      // Counter for the number of times 'search' is found

    // Get lengths of the search and replace strings
    size_t len_search = strlen(search);
    size_t len_replace = strlen(replace);

    // --- Input Validation and Edge Cases ---
    if (original == NULL || search == NULL || replace == NULL) {
        fprintf(stderr, "Error: Input strings for replaceAll cannot be NULL.\n");
        return NULL;
    }
    // If the search string is empty, replacing it would lead to an infinite loop.
    // In this case, we just return a copy of the original string.
    if (len_search == 0) {
        return original;
    }

    // --- Step 1: Count Occurrences to Determine New String Length ---
    // Iterate through the original string to find all occurrences of 'search'.
    // strstr returns a pointer to the first occurrence of 'search' in 'current_pos', or NULL if not found.
    for (current_pos = (char*)original; (found_pos = strstr(current_pos, search)); current_pos = found_pos + len_search) {
        count++;
    }

    // --- Step 2: Calculate the Required Memory for the Result String ---
    // The length of the original string.
    size_t len_original = strlen(original);
    // The new length will be the original length plus the net change for each replacement.
    // (len_replace - len_search) is the change in length per replacement.
    size_t len_result = len_original + (len_replace - len_search) * count;

    // --- Step 3: Allocate Memory for the Result String ---
    // Allocate memory for the result string (+1 for the null terminator).
    result = (char*)malloc(len_result + 1);
    if (result == NULL) {
        perror("Error: Failed to allocate memory for replacement string");
        return NULL; // Return NULL to indicate failure
    }

    // --- Step 4: Perform the Replacements and Copy to Result String ---
    current_pos = (char*)original; // Reset current_pos to the beginning of the original string
    char* out_pos = result;        // Pointer to write into the result string

    // Loop through the original string, performing replacements
    while ((found_pos = strstr(current_pos, search)) != NULL) {
        // Calculate the length of the segment before the found 'search' string
        size_t len_prefix = found_pos - current_pos;

        // Copy the segment before the 'search' string into the result
        strncpy(out_pos, current_pos, len_prefix);
        out_pos += len_prefix; // Advance the output pointer

        // Copy the 'replace' string into the result
        strcpy(out_pos, replace);
        out_pos += len_replace; // Advance the output pointer

        // Move the current_pos in the original string past the found 'search' string
        current_pos = found_pos + len_search;
    }

    // Copy any remaining part of the original string (after the last replacement)
    strcpy(out_pos, current_pos);

    return result; // Return the pointer to the new string
}

/**
 * @brief Reads the entire content of a file into a dynamically allocated string.
 * The caller is responsible for freeing the returned memory using `free()`.
 *
 * @param filename The path to the file to read.
 * @return A pointer to the dynamically allocated string containing the file's content,
 * or NULL if the file cannot be opened, memory allocation fails, or the file is empty.
 */
char* readFileToString(const char* filename) {
    FILE* file = NULL;      // File pointer
    char* buffer = NULL;    // Buffer to store file content
    long file_size = 0;     // Size of the file

    // --- Input Validation ---
    if (filename == NULL) {
        fprintf(stderr, "Error: Filename for readFileToString cannot be NULL.\n");
        return NULL;
    }

    // --- Step 1: Open the file in binary read mode ('rb') ---
    // 'rb' is used for robust cross-platform compatibility, especially for file size calculation.
    file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file for reading");
        return NULL; // Return NULL if file cannot be opened
    }

    // --- Step 2: Determine file size ---
    // Seek to the end of the file
    if (fseek(file, 0, SEEK_END) != 0) {
        perror("Error seeking to end of file");
        fclose(file);
        return NULL;
    }

    // Get the current position, which is the file size
    file_size = ftell(file);
    if (file_size == -1L) { // Check for ftell error
        perror("Error getting file size");
        fclose(file);
        return NULL;
    }

    // Seek back to the beginning of the file
    if (fseek(file, 0, SEEK_SET) != 0) {
        perror("Error seeking to beginning of file");
        fclose(file);
        return NULL;
    }

    // --- Step 3: Allocate memory for the file content ---
    // Allocate enough memory for the file content plus one byte for the null terminator.
    buffer = (char*)malloc(file_size + 1);
    if (buffer == NULL) {
        perror("Error allocating memory for file content");
        fclose(file);
        return NULL; // Return NULL if memory allocation fails
    }

    // --- Step 4: Read file content into the buffer ---
    // fread returns the number of items successfully read.
    // Here, we read file_size number of 1-byte items.
    size_t bytes_read = fread(buffer, 1, file_size, file);
    if (bytes_read != file_size) {
        fprintf(stderr, "Warning: Mismatch in bytes read (%zu) vs file size (%ld).\n", bytes_read, file_size);
        // It's good practice to free the buffer even on partial reads if we can't guarantee completeness.
        free(buffer);
        fclose(file);
        return NULL;
    }

    // --- Step 5: Null-terminate the buffer ---
    // Ensure the buffer is null-terminated so it can be treated as a C string.
    buffer[file_size] = '\0';

    // --- Step 6: Close the file ---
    fclose(file);

    return buffer; // Return the pointer to the string containing file content
}
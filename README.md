# VAULT
Stands for Verified, Atomic, Uncomplicated, Low-level Toolkit

and is a C-like, Rust-inspired and written optimized low level programming language

## Does the world really need this? Another C?
Well kind of. C is at its core a way to write low level code, without fiddling with registers.
Stupid as we programmers are, we have taken it to all kinds of places it was not made for - everything which is not a computer in the 70s or a bare metal machine. I think, that what makes C great, is the control it gives over the hardware and its *do what you want approach*. Its flaws are obvious tho: No namespaces, no easy way of obtaining packages and other libraries, and most importantly: **no compiler, that is made for humans**.

VAULT aims to change that. (currently most of it is not yet implemented lol). It features LLVMs amazing optimizations, C's control and the rust compilers *smartness*.

### Why not yet another C compiler?
Creating a C-spec adherent compiler that fulfills those desires is impossible. Due to the nature of the language, anything past some better error handling is impossible.

That's why I changed the things that needed changing and along the way also made some sensible improvements, because this language does not yet need to be backwards compatible 50 years.

### VAULT's philosophy
The programmer knows what they are doing. Give them the right tools and they can achieve greatness.

## Installation
Currently, the only way of obtaining the compiler is by cloning the repo and building it. This also means building LLVM on your machine.

Notice: Make sure to compile LLVM in release mode and also compile this project in release mode (or debug if that's what you want).

## The Syntax
First program: *Print "Hello World"*
````
import stdlib

def export main(): void {
    stdlib::print("Hello World")
}
````
### Types
VAULT has the following types
- **i32**: 32-bit signed   integer
- **i64**: 64-bit signed   integer
- **u32**: 32-bit unsigned integer
- **u64**: 64-bit unsigned integer
- **u8**:  8-bit  unsigned integer
- **f32**: 32-bit          float
- **f64**: 64-bit          float
- **bool**: 1-bit unsigned integer
- **void**: Function type exclusive void
- **struct**: A struct. Not actually a type, but you create your own types via structs
- **function**: A type representing a function

### Assignments
Assignment are similar to the ones in rust. However, types must always be clear. Mutability is by default and there is no const.
```
// Automatically infer a name
let name = "John";
// Annotated type
let last_name: u8* = "Doe";
// Declaration (must include a type)
let age: i32;
// Reassignment
age = 23;
name = "Chuck";
```

### Functions def / dec
Function definitions and declarations are similar to Python and C.

Function modes:
- export : Expose externally
- extern : Declared externally
- private : WIP
- *default* : WIP (public)

Note : The *main* function is not mandatory and can return anything, e.g. **unchecked**
```
// Return type i32
def export main(): i32 {
    return 0
}

// Extern void function with param 'num' i32
def extern print_whatever(num: i32): void;
```

### Function calling
Calling a function is very simple and identical to many other languages.

Note: *void*-type functions can not be used as values. That will result in a compiler error

```
// `age`, `print_user_data` and `prompt_for_name` previously defined

let name = prompt_for_name("What's your name? ");
print_user_data(age, name);
```

### Malloc and free
Malloc and free are built in and have their own syntax.
Both will return the type `ptr` which is an untyped pointer. You might want to cast it or use type annotations.

Note: Free can be auto-detected for most things
```
// Allocate 256 bytes
let result: u8* = |> 256;
// Scanf to `result`
stdlib::ptr_input(result, 256);
```

Use `|> (amount)` for malloc and `|< (expr)` for free

### Structs
Struct definitions are a like the ones in rust, but simpler.

```
struct Point {
    x: i32,
    y: i32
}
```

#### Initialization
Use the `new` keyword to create a struct. The arguments must be in the order that the struct has.

```
// x = 0; y = 1
let point = new Point(0, 1);
```

To reassign a property, you must have a pointer to the property (like shown before)
```
&point~x = 160;
```

#### Properties
Literal and pointer access are possible. However, they must occur on a pointer to the struct.

```
// Access literal value of `Point->x`
let x: i32 = &point.x;
// Access pointer to value of `Point->x`
let x: i32* = &point~x;
```

To reassign a property, you must have a pointer to the property (like shown before)
```
&point~x = 160;
```

#### Member functions on structs
Declaring member functions for structs is also possible:
```
def draw_point(p: Point*) Point : void {
    ...
}
```
First param of the method must be of type *struct* pointer.

They are called like so:
```
// The value to call on must be a pointer to the struct
&point.draw_point();
```

### Conditionals
If-*elif*-else statements are possible and a combination of C and python syntax.
```
let error_code = do_something_dangerous();

if error_code != 0 {
    stdlib::print("An error has occured!")
} else {
    stdlib::print("Everything is fine")
};
// Note: ';' is required after the last block
```

### Conditional loops
While statements are possible and a combination of C and python syntax.
```
let a = 100;
while a > 10 {
    a = a - 1;
};
// Note: ';' is required after the block
```

### Casts
You may cast all primitive types (all types except structs and functions).
However, you must be careful as casts are **NOT** safe and purely change the *perceived type*.
```
// Ext cast.
let num: i32 = 0;
let other = num => i64;
// Trunc cast. (lossy)
num = other => i32;
...
// Very useful: Cast a typed pointer into an untyped one
let p = "abc" => ptr;
```

### Imports
You may import code from other files.
Once you import a file (aka module), you carry over all symbols to the MAV (Module Access Variant) named after the module. Import compiles the file in between the previous and following code.
```
// Import the file 'stdlib.vt' to 'stdlib'
import stdlib
...
// Import the file 'stdlib.vt' to 'std'
import stdlib => std
...
// Import the file 'stdlib-other.vt' to 'std'
import "stdlib-other" => std
...
// Invalid syntax! String imports require a module name
import "stdlib-other"
```

Then use as follows:
```
std::imported_function("this is neat");
```

### Directives
Directives are compile time instructions. They may enable you to link something for specific users or enable / disable creating a function.
```
// Windows only function
#(OS "windows")
def func();
```
Directives are declared with `#`. They follow the layout `( NAME <...> )`. They always return a boolean.
Putting them over a statement will disable / enable it depending on the return value of the directive.

Here is a list of directives:
- AND <a: directive> <b: directive>
- OR <a: directive> <b: directive>
- XOR <a: directive> <b: directive>
- NOT <a: directive>
- ALWAYS => Always returns *true*
- NEVER => Always returns *false*
- DEBUG => Returns whether debug mode is on
- LINK <path: string> => Link with a library; Always returns *true*
- OS <os: string> => Is on os
- ARCH <arch: string> => Is on arch
- IF <condition: directive> <exe: directive> => Runs and returns value of exe if condition is true
- IFI <condition: directive> <exe: directive> => Runs exe if condition is true; Always returns *true*
- IFE <condition: directive> <else: directive> <exe: directive> => Runs exe if condition is true, otherwise else; Returns the corresponding value
- IFEI <condition: directive> <else: directive> <exe: directive> => Runs exe if condition is true, otherwise else; Always returns *true*
- ERR <footer: string> => Creates a compiler error at the directive with a footer
- ERRN <footer: string> <note: string> => Creates a compiler error at the directive with a footer and a note
- WARN <footer: string> => Creates a compiler warning at the directive with a footer; Always returns *true*
- WARNF <footer: string> => Creates a compiler warning at the directive with footer; Always returns *false*

## Concepts
### 1. (Built-in) free means destruction
When using the built-in free method, the compiler will forget about the variable to protect against *use after free* exceptions
### 2. Data types should match their memory layout
There are no built-in arrays or hashmaps or such, because they abstract how data is stored in memory and how the CPU handles it
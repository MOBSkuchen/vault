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

### VAULT's approach
The programmer knows what they are doing. Give them the right tools and they can achieve greatness.

## Installation
Currently, the only way of obtaining the compiler is by cloning the repo and building it. This also means building LLVM on your machine.

Notice: Make sure to compile LLVM in release mode and also compile this project in release mode (or debug if that's what you want).

## The Syntax
Print "Hello World"
```
// As there are currently no imports, we need to create a proto type function ourselves
def extern print(s: u8*): void;

def export main(): void {
    print("Hello World")
}
```
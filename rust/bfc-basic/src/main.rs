// Traits -> `interface` feature of Rust
//  use `impl` keyword to implement the methods of trait
//
// Iterators in rust are different than usual C++ ones
//  Its a trait that may be implemented by many types
//  Iterator `produces` value in Rust, unlike C++ where an iterator
//  only points to existing values
//  Iterators are `lazy`, evaluation happens only when methods like
//  `.collect` are called

// Cmd line args
use std::{env, process::exit};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("No file provided");
        exit(1);
    }

    // https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html
    // Ownership model of Rust:
    //  If Rust Ownership rules are violated -- then the program won't
    //  compile :skull emoji:
    //
    //  Its all RAII?? (drop trait(method) after scope)
    //
    //  s2 = s1 -> this is not a shallow copy, instead its a move! s1
    //  is no longer a valid reference, if we try to use s1, we get
    //  compile time err!
    //  For deep copy, use `clone` method.
    //  !! This is valid only for heap objects and not stack memory !!
    //
    //  drop function is called whenever reference to memory becomes
    //  none (no ptr to allocated mem)
    //  ex: let mut s = String::from("hello");
    //      s = String::from("world");
    //      -> "hello" is `drop`ped as s now points
    //         to "world"
    //
    //  Passing values to different functions transfers the ownership
    //  to the called function.
    //  If the function does not return the value back to callee you
    //  cannot use the value in callee function!
    //
    //  NOTE: The ownership of a variable follows the same pattern
    //  every time: Assigning a value to another variable moves it.
    //
    // References & Borrowing:
    //  Like a pointer, passing by ref won't pass the ownership of var.
    //  Going outta scrop won't drop a referenced var!
    //  Action of creating a reference is `borrowing`
    //  References are also immutable, we cannot modify values of vars
    //  that are borrowed! (use &mut reference)
    //    NOTE: We cannot borrow mutable ref of a var more than once
    //    at a time (even if it already has an immutable ref)!
    let file = &args[1];

    println!("file name: {}", file);
}

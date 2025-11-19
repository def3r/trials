#![allow(dead_code)]

use std::mem;

// a var is immutable by default
fn main() {
    // Annotation is possible
    let var: bool = true;
    let c: char = ' ';
    // var = false; can be done (immutable by default </3)

    let mut i: i16 = 0;
    i = 420;

    let arr: [u16; 4] = [0, 1, 2, 3];
    let tup = (-32i16, 0f64, "ohno", '4');

    println!("{:?}", arr);
    println!("{:?}", tup);

    println!("Classic ptr size {}", size_of::<&char>());

    println!("{} bytes for bool", mem::size_of_val(&var));
    println!("{} bytes for char is insane", mem::size_of_val(&c));

    println!("w h y");
    let res = std::any::type_name_of_val(&String::from("Hello :D"));
    println!("{}", res);
    let res = std::any::type_name_of_val("Hello :D");
    println!("{}", res);
}

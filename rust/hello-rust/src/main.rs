// Launch docs: rustup doc
// Launch std lib docs: rustup doc std
// ref: https://doc.rust-lang.org/rust-by-example/

fn main() {
    let x = 10;
    let y = 20;

    // format! is a macro, works just like sprintf in c
    let s = format!("{x} + {y} = {}", x + y);

    // println! is also a macro, present in std::fmt
    println!("{s}\n");

    // positional args
    println!(
        "{0} is pos 0 and {1} is pos 1, here's 0 again: {0}",
        "Zero", 1
    );

    // named args also allowed
    println!("{key} = {val}", key = "name", val = "rust");
    // named + pos args
    println!("{key} {0}= {val}", ":", key = "name", val = "rust");

    // different formatting opts, after ':' char
    println!("Base 10: {}", 67);
    println!("Base 2: {:b}", 67);

    // right justify width
    // text widht = 6, var width = 2; => 4 spaces from beg
    println!("{:>6}", "so");

    // to use named arg to specify var width, use $
    println!("{number:0>width$}", number = 67, width = 10);
}

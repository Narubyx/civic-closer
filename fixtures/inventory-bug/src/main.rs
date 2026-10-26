fn main() {
    // An empty shelf: the report for it panics.
    let shelf: Vec<u32> = Vec::new();
    println!("average: {}", inventory::average_price_cents(&shelf));
}

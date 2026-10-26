//! Tiny synthetic inventory library with a seeded bug, used to try Panic-to-Patch.

/// Average price in cents of the given items.
pub fn average_price_cents(prices: &[u32]) -> u32 {
    let total: u32 = prices.iter().sum();
    total / prices.len() as u32
}

/// Total price in cents of the given items.
pub fn total_price_cents(prices: &[u32]) -> u32 {
    prices.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_sums_prices() {
        assert_eq!(total_price_cents(&[100, 250, 50]), 400);
    }

    #[test]
    fn average_of_three_items() {
        assert_eq!(average_price_cents(&[100, 200, 300]), 200);
    }
}

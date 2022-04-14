// Ternary addition domain.

use core::str::FromStr;
use std::iter::once;

use rand::Rng;

use super::{State, Action};

pub struct TernaryAddition {
    max_size: usize,
}

impl TernaryAddition {
    pub fn new(max_size: usize) -> TernaryAddition {
        TernaryAddition { max_size: max_size }
    }
}

#[derive(Copy, Clone)]
struct TernaryDigit {
    digit: u8, // Always 0, 1 or 2.
    power: u8, // The corresponding power of 3 that the digit multiplies.
}

impl FromStr for TernaryDigit {
    type Err = ();

    fn from_str(s: &str) -> Result<TernaryDigit, Self::Err> {
        if s.len() != 2 {
            return Err(());
        }

        Ok(TernaryDigit { digit: ((s.chars().nth(0).unwrap() as u32) - ('a' as u32)) as u8,
                          power: ((s.chars().nth(1).unwrap() as u32) - ('0' as u32)) as u8 })
    }
}

impl ToString for TernaryDigit {
    fn to_string(&self) -> String {
        format!("{}{}",
                (('a' as u8) + self.digit) as char,
                (('0' as u8) + self.power) as char)
    }
}

struct TernaryNumber {
    digits: Vec<TernaryDigit>
}

impl FromStr for TernaryNumber {
    type Err = ();

    fn from_str(s: &str) -> Result<TernaryNumber, Self::Err> {
        if !(s.starts_with("#(") && s.ends_with(")")) {
            return Err(());
        }

        let mut digits = Vec::new();

        if s.len() > 3 {
            for d in s[2..(s.len() - 1)].split(' ') {
                digits.push(TernaryDigit::from_str(&d)?);
            }
        }

        Ok(TernaryNumber{digits: digits})
    }
}

impl ToString for TernaryNumber {
    fn to_string(&self) -> String {
        format!("#({})", self.digits.iter().map(|d| d.to_string()).collect::<Vec<String>>().join(" "))
    }
}

impl TernaryNumber {
    fn is_reduced(&self) -> bool {
        for (i, d) in self.digits.iter().enumerate() {
            if d.digit == 0 {
                return false;
            }

            if i > 0 && d.power <= self.digits[i-1].power {
                return false;
            }
        }

        true
    }
}

impl super::Domain for TernaryAddition {
    fn name(&self) -> String {
        return String::from("ternary-addition");
    }

    fn generate(&self, seed: u64) -> State {
        let mut rng = super::new_rng(seed);
        let size = rng.gen_range(1..self.max_size);

        let mut digits = Vec::with_capacity(size);

        for _ in 0..size {
            digits.push(TernaryDigit {
                digit: rng.gen_range(0..3) as u8,
                power: rng.gen_range(0..6) as u8,
            });
        }

        (TernaryNumber{ digits: digits }).to_string()
    }

    fn step(&self, state: State) -> Option<Vec<Action>> {
        let n = TernaryNumber::from_str(state.as_str()).unwrap();

        if n.is_reduced() {
            return None;
        }

        let mut actions = Vec::new();

        for (i, d) in n.digits.iter().enumerate() {
            // Delete
            if d.digit == 0 {
                actions.push(Action {
                    next_state: TernaryNumber {
                        digits: n.digits[0..i].iter()
                            .chain(n.digits[(i+1)..n.digits.len()].iter())
                            .copied().collect(),
                    }.to_string(),
                    formal_description: format!("del {}, {}", i, d.to_string()),
                    human_description: format!("Erase {}", d.to_string()),
                });
            }

            // Swap
            if i + 1 < n.digits.len() {
                actions.push(Action {
                    next_state: TernaryNumber {
                        digits: n.digits[0..i].iter()
                            .chain(once(&n.digits[i+1]))
                            .chain(once(&n.digits[i]))
                            .chain(n.digits[(i+2)..n.digits.len()].iter())
                            .copied().collect(),
                    }.to_string(),
                    formal_description: format!("swap {}, {} {}", i, d.to_string(), n.digits[i+1].to_string()),
                    human_description: format!("Swap {} and {}", d.to_string(), n.digits[i+1].to_string()),
                });
            }

            // Combine
            if i + 1 < n.digits.len() && d.power == n.digits[i+1].power {
                let new_d1 = TernaryDigit { digit: (d.digit + n.digits[i+1].digit) % 3, power: d.power };
                let new_d2 = TernaryDigit { digit: (d.digit + n.digits[i+1].digit) / 3, power: d.power + 1 };
                actions.push(Action {
                    next_state: TernaryNumber {
                        digits: n.digits[0..i].iter()
                            .chain(once(&new_d1))
                            .chain(once(&new_d2))
                            .chain(n.digits[(i+2)..n.digits.len()].iter())
                            .copied().collect(),
                    }.to_string(),
                    formal_description: format!("comb {}, {} {}", i, d.to_string(), n.digits[i+1].to_string()),
                    human_description: format!("Combine {} and {}", d.to_string(), n.digits[i+1].to_string()),
                });
            }
        }

        Some(actions)
    }
}

#[cfg(test)]
mod test {
    use std::str::FromStr;

    #[test]
    fn test_parser() {
        let d = super::TernaryNumber::from_str("#()").unwrap();
        assert_eq!(d.digits.len(), 0);

        let d = super::TernaryNumber::from_str("#(a0)").unwrap();
        assert_eq!(d.digits.len(), 1);
    }
}

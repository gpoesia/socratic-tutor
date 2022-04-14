// Sorting domain.

use core::str::FromStr;
use std::iter::{repeat};

use rand::Rng;
use rand::seq::SliceRandom;

use super::{State, Action};

pub struct Sorting {
    max_size: usize,
}

impl Sorting {
    pub fn new(max_size: usize) -> Sorting {
        Sorting { max_size: max_size }
    }
}

struct SortingState {
    elements: Vec<usize>
}

impl FromStr for SortingState {
    type Err = ();

    fn from_str(s: &str) -> Result<SortingState, Self::Err> {
        if !(s.starts_with("[") && s.ends_with("]")) {
            return Err(());
        }

        let mut elems = Vec::new();
        let mut last: usize = 0;

        for c in s[1..s.len()].chars() {
            if c == '|' || c == ']' {
                elems.push(last);
                last = 0;
            } else {
                last += 1;
            }
        }

        Ok(SortingState{elements: elems})
    }
}

impl ToString for SortingState {
    fn to_string(&self) -> String {
        format!("[{}]",
                self.elements.iter().map(|d| repeat("=").take(*d).collect::<String>())
                .collect::<Vec<String>>().join("|"))
    }
}

impl SortingState {
    fn is_sorted(&self) -> bool {
        for i in 1..self.elements.len() {
            if self.elements[i] < self.elements[i-1] {
                return false;
            }
        }

        true
    }
}

impl super::Domain for Sorting {
    fn name(&self) -> String {
        return String::from("sorting");
    }

    fn generate(&self, seed: u64) -> State {
        let mut rng = super::new_rng(seed);
        let size = rng.gen_range(2..self.max_size);

        let mut elems = (1..size).collect::<Vec<usize>>();
        elems.shuffle(&mut rng);

        (SortingState{ elements: elems }).to_string()
    }

    fn step(&self, state: State) -> Option<Vec<Action>> {
        let s = SortingState::from_str(state.as_str()).unwrap();

        if s.is_sorted() {
            return None;
        }

        let mut actions = Vec::new();

        for i in 0..(s.elements.len() - 1) {
            actions.push(Action {
                    next_state: SortingState {
                        elements: swap_adj(&s.elements, i),
                    }.to_string(),
                    formal_description: format!("swap {}", i),
                    human_description: format!("Swap the {}-th element with the next.", i+1),
                });
        }

        actions.push(Action {
            next_state: SortingState {
                elements: reverse(&s.elements),
            }.to_string(),
            formal_description: format!("reverse"),
            human_description: format!("Reverse the list"),

        });

        Some(actions)
    }
}

fn swap_adj(v: &Vec<usize>, i: usize) -> Vec<usize> {
    let mut v2 = v.clone();
    let tmp = v2[i];
    v2[i] = v2[i+1];
    v2[i+1] = tmp;
    v2
}

fn reverse(v: &Vec<usize>) -> Vec<usize> {
    let mut v2 = v.clone();
    v2.reverse();
    v2
}

#[cfg(test)]
mod test {
    use std::str::FromStr;
    use crate::domain::Domain;

    #[test]
    fn test_parser() {
        let d = super::SortingState::from_str("[=|==]").unwrap();
        assert_eq!(d.elements.len(), 2);

        let d = super::SortingState::from_str("[===|=====|==|====|=]").unwrap();
        assert_eq!(d.elements.len(), 5);
    }

    #[test]
    fn test_step() {
        let s = super::SortingState::from_str("[====|=|==|===]").unwrap();
        let d = super::Sorting::new(10);
        let actions = d.step(s.to_string());
        assert_eq!(actions.unwrap().len(), 4);

        let sorted = super::SortingState::from_str("[=|==|===|====]").unwrap();
        let actions2 = d.step(sorted.to_string());
        assert!(actions2.is_none());
    }
}

// Definition of a domain

pub mod equations;
pub mod ternary;
pub mod sorting;
pub mod rubiks_cube;
pub mod fractions;
pub mod multiplication;
pub mod key_to_door;

use rand_pcg::Pcg64;

pub type State = String;

pub struct Action {
    pub next_state : State,
    pub formal_description : String,
    pub human_description : String,
}

pub trait Domain {
    fn name(&self) -> String;

    fn generate(&self, seed: u64) -> State;

    fn step(&self, state: State) -> Option<Vec<Action>>;
}

fn new_rng(seed: u64) -> Pcg64 {
    Pcg64::new((0xcafef00dd15ea5e5 + seed).into(),
               0xa02bdbf7bb3c0a7ac28fa16a64abf96)
}

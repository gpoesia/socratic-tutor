// Rubik's Cube domain.

use core::str::FromStr;
use std::convert::TryFrom;

use rand::Rng;
use rand::distributions::{Distribution, Uniform};

use super::{State, Action};

pub struct RubiksCube {
    max_shuffles: usize,
}

impl RubiksCube {
    pub fn new(max_shuffles: usize) -> RubiksCube {
        RubiksCube { max_shuffles: max_shuffles }
    }
}

const N_ELEMENTS: usize = 9*6; // 9 elements per each of the 6 faces.
const N_MOVES: usize = 12; // One rotation for each face + its reverse.
const N_FACES: usize = 6; // One rotation for each face + its reverse.
const FACES_BEGIN: [char;6] = ['A', 'B', 'C', 'D', 'E', 'F']; // One letter identifying the beginning of each face.
const FACES_END: [char;6] = ['a', 'b', 'c', 'd', 'e', 'f']; // One letter identifying the end of each face.

const MOVE_NAMES: &[&&str; 12] = &[&"U1", &"F1", &"B1", &"L1", &"R1", &"D1",
                                   &"U-1", &"F-1", &"B-1", &"L-1", &"R-1", &"D-1"];

const MOVE_SIZE: usize = 21;

const MOVE_IDX_NEW: [[usize;MOVE_SIZE]; N_MOVES] = [
    [ 0, 1, 2, 5, 8, 7, 6, 3, 0,20,23,26,47,50,53,29,32,35,38,41,44],
    [45,46,47,50,53,52,51,48,45, 0, 3, 6,24,25,26,17,14,11,29,28,27],
    [36,37,38,41,44,43,42,39,36, 2, 5, 8,35,34,33,15,12, 9,18,19,20],
    [18,19,20,23,26,25,24,21,18, 0, 1, 2,44,43,42, 9,10,11,45,46,47],
    [27,28,29,32,35,34,33,30,27, 6, 7, 8,51,52,53,15,16,17,38,37,36],
    [ 9,10,11,14,17,16,15,12, 9,18,21,24,36,39,42,27,30,33,45,48,51],
    [ 0, 1, 2, 5, 8, 7, 6, 3, 0,20,23,26,47,50,53,29,32,35,38,41,44],
    [45,46,47,50,53,52,51,48,45, 0, 3, 6,24,25,26,17,14,11,29,28,27],
    [36,37,38,41,44,43,42,39,36, 2, 5, 8,35,34,33,15,12, 9,18,19,20],
    [18,19,20,23,26,25,24,21,18, 0, 1, 2,44,43,42, 9,10,11,45,46,47],
    [27,28,29,32,35,34,33,30,27, 6, 7, 8,51,52,53,15,16,17,38,37,36],
    [ 9,10,11,14,17,16,15,12, 9,18,21,24,36,39,42,27,30,33,45,48,51],
];

const MOVE_IDX_OLD: [[usize;MOVE_SIZE]; N_MOVES] = [
    [ 6, 3, 0, 1, 2, 5, 8, 7, 6,47,50,53,29,32,35,38,41,44,20,23,26],
    [51,48,45,46,47,50,53,52,51,24,25,26,17,14,11,29,28,27, 0, 3, 6],
    [42,39,36,37,38,41,44,43,42,35,34,33,15,12, 9,18,19,20, 2, 5, 8],
    [24,21,18,19,20,23,26,25,24,44,43,42, 9,10,11,45,46,47, 0, 1, 2],
    [33,30,27,28,29,32,35,34,33,51,52,53,15,16,17,38,37,36, 6, 7, 8],
    [15,12, 9,10,11,14,17,16,15,36,39,42,27,30,33,45,48,51,18,21,24],
    [ 2, 5, 8, 7, 6, 3, 0, 1, 2,38,41,44,20,23,26,47,50,53,29,32,35],
    [47,50,53,52,51,48,45,46,47,29,28,27, 0, 3, 6,24,25,26,17,14,11],
    [38,41,44,43,42,39,36,37,38,18,19,20, 2, 5, 8,35,34,33,15,12, 9],
    [20,23,26,25,24,21,18,19,20,45,46,47, 0, 1, 2,44,43,42, 9,10,11],
    [29,32,35,34,33,30,27,28,29,38,37,36, 6, 7, 8,51,52,53,15,16,17],
    [11,14,17,16,15,12, 9,10,11,45,48,51,18,21,24,36,39,42,27,30,33],
];

#[derive(Clone)]
struct CubeState {
    colors: [u8; N_ELEMENTS],
}

impl FromStr for CubeState {
    type Err = ();

    fn from_str(s: &str) -> Result<CubeState, Self::Err> {
        if s.len() != N_ELEMENTS + 2*N_FACES {
            return Err(());
        }

        let mut colors = [0 as u8;N_ELEMENTS];
        let mut next: usize = 0;

        for c in s.chars() {
            if c.is_numeric() {
                colors[next] = u8::try_from(c.to_digit(10).unwrap()).unwrap();
                next += 1;
            }
        }

        Ok(CubeState{colors: colors})
    }
}

impl ToString for CubeState {
    fn to_string(&self) -> String {
        let mut v = Vec::<String>::new();
        for i in 0..N_FACES {
            v.push(FACES_BEGIN[i].to_string());

            for j in 0..9 {
                v.push(self.colors[i*9 + j].to_string());
            }

            v.push(FACES_END[i].to_string());
        }

        return v.join("");
    }
}

impl CubeState {
    // Returns a new CubeState that is already solved.
    fn new() -> CubeState {
        CubeState { colors:
                    [0, 0, 0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 2, 2, 2,
                     3, 3, 3, 3, 3, 3, 3, 3, 3,
                     4, 4, 4, 4, 4, 4, 4, 4, 4,
                     5, 5, 5, 5, 5, 5, 5, 5, 5] }
    }

    fn is_solved(&self) -> bool {
        for i in 0..N_FACES {
            for j in 0..9 {
                if self.colors[i*9 + j] as usize != i {
                    return false;
                }
            }
        }
        true
    }

    fn apply(&mut self, n_move: usize) {
        let mut new_colors = self.colors.clone();

        for i in 0..MOVE_SIZE {
            new_colors[MOVE_IDX_NEW[n_move][i]] = self.colors[MOVE_IDX_OLD[n_move][i]];
        }

        self.colors = new_colors;
    }
}

impl super::Domain for RubiksCube {
    fn name(&self) -> String {
        return String::from("rubiks-cube");
    }

    fn generate(&self, seed: u64) -> State {
        let mut rng = super::new_rng(seed);
        let n_shuffles = rng.gen_range(1..self.max_shuffles);
        let random_move: Uniform<usize> = Uniform::from(0..N_MOVES);

        let mut cube = CubeState::new();

        for _ in 0..n_shuffles {
            cube.apply(random_move.sample(&mut rng));
        }

        cube.to_string()
    }

    fn step(&self, state: State) -> Option<Vec<Action>> {
        let s = CubeState::from_str(state.as_str()).unwrap();

        if s.is_solved() {
            return None;
        }

        let mut actions = Vec::new();

        for i in 0..N_MOVES {
            let mut c2 = s.clone();
            c2.apply(i);
            actions.push(Action {
                    next_state: c2.to_string(),
                    formal_description: MOVE_NAMES[i].to_string(),
                    human_description: MOVE_NAMES[i].to_string(),
                });
        }

        Some(actions)
    }
}

#[cfg(test)]
mod test {
    use std::str::FromStr;

    #[test]
    fn test_parser() {
        let c = super::CubeState::new();
        let d = super::CubeState::from_str(c.to_string().as_str()).unwrap();
        assert!(d.is_solved());
    }

    #[test]
    fn test_step() {
        let mut c = super::CubeState::new();
        assert!(c.is_solved());
        c.apply(0);
        assert!(!c.is_solved());
        c.apply(6);
        assert!(c.is_solved());
    }
}

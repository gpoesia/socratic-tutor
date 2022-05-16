// Key-to-door domain.

use core::str::FromStr;

use rand::Rng;

use super::{State, Action};

const MAX_ROOM_SIZE: u8 = 8;

pub struct KeyToDoor {
    max_apples: usize,
    apple_win_probability: f32,
}

impl KeyToDoor {
    pub fn new(max_apples: usize, apple_win_probability: f32) -> KeyToDoor {
        KeyToDoor { max_apples: max_apples, apple_win_probability: apple_win_probability }
    }
}

#[derive(Clone)]
struct KeyToDoorState {
    current_room: u8,
    room_size: u8,
    seed: u64,
    player_position: (u8, u8),
    key_position: (u8, u8),
    has_key: bool,
    apples: Vec<(u8, u8)>,
    door_position: (u8, u8),
}

impl FromStr for KeyToDoorState {
    type Err = ();

    fn from_str(s: &str) -> Result<KeyToDoorState, Self::Err> {
        let mut state = KeyToDoorState {
            current_room: 0,
            player_position: (0, 0), 
            key_position: (0, 0), 
            has_key: false,
            apples: Vec::new(),
            door_position: (0, 0),
            seed: 0,
            room_size: 0,
        };

        let lines = s.split("\n").map(|s| s.to_string()).collect::<Vec<String>>();
        state.seed = lines[0].parse().unwrap();

        if let Some(c) = lines[1].chars().nth(0) {
            if c == 'A' {
                state.current_room = 0;
            } else if c == 'B' {
                state.current_room = 1;
            } else if c == 'C' {
                state.current_room = 2;
            } else if c == '!' {
                state.current_room = 3;
            } else {
                return Err(());
            }
        }

        if lines[1].chars().nth(1) == Some('K') {
            state.has_key = true;
        }

        state.room_size = (lines.len() - 2) as u8;

        for (i, l) in lines.iter().enumerate() {
            if i < 2 {
                continue;
            }
            for (j, c) in l.chars().enumerate() {
                let pos: (u8, u8) = ((i - 2) as u8, j as u8); // (i.try_into().unwrap(), j.try_into().unwrap());
                if c == 'k' {
                    state.key_position = pos;
                } else if c == 'a' {
                    state.apples.push(pos);
                } else if c == 'x' {
                    state.player_position = pos;
                } else if c == 'D' {
                    state.door_position = pos;
                }
            }
        }

        Ok(state)
    }
}

impl ToString for KeyToDoorState {
    fn to_string(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("{}", self.seed));

        lines.push(format!("{}{}",
                           match self.current_room { 0 => { 'A' },
                                                     1 => { 'B' },
                                                     2 => { 'C' },
                                                     _ => { '!' }, },
                           if self.has_key { "K" } else { "" }));

        for i in 0..self.room_size {
            let mut s = String::with_capacity(self.room_size.into());

            for j in 0..self.room_size {
                s.push(
                    if self.player_position == (i, j) {
                        'x'
                    } else if self.current_room == 0 && self.key_position == (i, j) {
                        'k'
                    } else if self.current_room == 1 && self.apples.contains(&(i, j)) {
                        'a'
                    } else if self.current_room == 2 && self.door_position == (i, j) {
                        'D'
                    } else {
                        '.'
                    }
                );
            }
            lines.push(s);
        }

        lines.join("\n")
    }
}

impl super::Domain for KeyToDoor {
    fn name(&self) -> String {
        return String::from("key-to-door");
    }

    fn generate(&self, seed: u64) -> State {
        let mut rng = super::new_rng(seed);
        let size = rng.gen_range(2..MAX_ROOM_SIZE);

        KeyToDoorState {
            current_room: 0,
            player_position: (0, 0), 
            key_position: (rng.gen_range(1..size), rng.gen_range(0..size)), 
            has_key: false,
            apples: Vec::new(),
            door_position: (0, 0),
            seed: seed,
            room_size: size,
        }.to_string()
    }

    fn step(&self, state: State) -> Option<Vec<Action>> {
        let s = KeyToDoorState::from_str(state.as_str()).unwrap();
        let mut actions = Vec::new();

        if s.current_room == 3 {
            return None;
        }

        let mut rng = super::new_rng(s.seed + s.apples.len() as u64);
        let next_apple_will_win = rng.gen::<f32>() < self.apple_win_probability;

        for d in 0..4 {
            let action_name = (["up", "right", "down", "left"][d]).to_string();

            let dr: i8 = [-1, 0, 1, 0][d];
            let dc: i8 = [0, 1, 0, -1][d];

            let r = (s.player_position.0 as i8) + dr;
            let c = (s.player_position.1 as i8) + dc;

            // Check if player can move.
            if r < 0 || c < 0 || r >= (s.room_size as i8) {
                continue;
            }

            let pos = (r as u8, c as u8);

            // Handle special cases, otherwise just move player if none apply.
            // Room 1: get the key.
            if s.current_room == 0 && pos == s.key_position {
                actions.push(Action {
                    next_state: (KeyToDoorState {
                        has_key: true,
                        player_position: pos,
                        ..s.clone()
                    }).to_string(),
                    formal_description: action_name.clone(),
                    human_description: action_name.clone(),
                });
            // Room 1: move to the second room.
            } else if s.current_room == 0 && c == s.room_size as i8 {
                // Generate apples.
                let mut rng = super::new_rng(s.seed);
                let n_apples = rng.gen_range(0..self.max_apples);
                let mut apples = Vec::new();

                for _i in 0..n_apples {
                    apples.push((rng.gen_range(1..s.room_size),
                                 rng.gen_range(1..s.room_size)));
                }

                actions.push(Action {
                    next_state: (KeyToDoorState {
                        current_room: 1,
                        apples: apples,
                        player_position: (0, 0),
                        ..s.clone()
                    }).to_string(),
                    formal_description: action_name.clone(),
                    human_description: action_name.clone(),
                });
            // Room 2: grab an apple
            } else if s.current_room == 1 && s.apples.contains(&pos) {
                if next_apple_will_win {
                    actions.push(Action {
                        next_state: (KeyToDoorState {
                            current_room: 3,
                            player_position: (0, 0),
                            ..s.clone()
                        }).to_string(),
                        formal_description: action_name.clone(),
                        human_description: action_name.clone(),
                    });
                } else {
                    actions.push(Action {
                        next_state: (KeyToDoorState {
                            player_position: pos,
                            ..s.clone()
                        }).to_string(),
                        formal_description: action_name.clone(),
                        human_description: action_name.clone(),
                    });
                }
            // Room 2: go to the last room.
            } else if s.current_room == 1 && c == s.room_size as i8 {
                actions.push(Action {
                    next_state: (KeyToDoorState {
                        current_room: 2,
                        player_position: (0, 0),
                        door_position: (rng.gen_range(1..s.room_size),
                                        rng.gen_range(0..s.room_size)),
                        ..s.clone()
                    }).to_string(),
                    formal_description: action_name.clone(),
                    human_description: action_name.clone(),
                });
            // Room 3: enter the door if has_key.
            } else if s.current_room == 2 && pos == s.door_position {
                if s.has_key {
                    actions.push(Action {
                        next_state: (KeyToDoorState {
                            current_room: 3,
                            player_position: (0, 0),
                            ..s.clone()
                        }).to_string(),
                        formal_description: action_name.clone(),
                        human_description: action_name.clone(),
                    });
                }
            // Otherwise: just move.
            } else if c < s.room_size as i8 {
                actions.push(Action {
                    next_state: (KeyToDoorState {
                        player_position: pos,
                        ..s.clone()
                    }).to_string(),
                    formal_description: action_name.clone(),
                    human_description: action_name.clone(),
                });
            }
        }

        Some(actions)
    }
}

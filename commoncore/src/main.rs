use commoncore::{DOMAINS, domain};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

fn bfs(d: &Arc<dyn domain::Domain>, s: &domain::State, max_edges: u32) -> bool {
    let mut seen = HashSet::new();
    let mut queue = VecDeque::new();
    let mut edges_traversed: usize = 0;

    queue.push_back(s.clone());
    seen.insert(s.clone());

    while !queue.is_empty() && edges_traversed < max_edges as usize {
        let next = queue.pop_front().unwrap();

        if let Some(actions) = d.step(next) {
            edges_traversed += actions.len();
            for action in actions.iter() {
                if seen.insert(action.next_state.clone()) {
                    queue.push_back(action.next_state.clone());
                }
            }
        } else {
            return true;
        }
    }

    false
}

fn main() {
    DOMAINS.with(|domains| {
        const MAX_EDGES: u32 = 1000000;
        const N_PROBLEMS: u32 = 100;

        println!("Found {} environments.", domains.borrow().len());

        for (name, domain) in domains.borrow().iter() {
            println!("Benchmarking {}...", name);
            let mut n_successes: u32 = 0;
            let mut branching_factor: f32 = 0.0;

            for i in 0..N_PROBLEMS {
                let problem = domain.generate(i.into());
                if let Some(actions) = domain.step(problem.to_string()) {
                    branching_factor += actions.len() as f32;
                }
                if bfs(domain, &problem, MAX_EDGES) {
                    n_successes += 1;
                }
            }

            println!("Success rate: {}, Avg. Branching Factor: {}",
                     (n_successes as f32) / (N_PROBLEMS as f32),
                     branching_factor / (N_PROBLEMS as f32));
        }
    });
}

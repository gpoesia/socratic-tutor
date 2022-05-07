use core::str::FromStr;
use std::borrow::Borrow;
use std::rc::Rc;

use rand::Rng;
use rand_pcg::Pcg64;

use super::{State, Action};
use pest::Parser;
use pest::iterators::Pair;
use pest::error::{Error as PestError};

#[derive(Copy, Clone, Debug, PartialEq)]
enum Operator {
    Add, Sub, Times
}

use Operator::{Add, Sub, Times};

impl ToString for Operator {
    fn to_string(&self) -> String {
        (match self {
            Operator::Add => "+",
            Operator::Sub => "-",
            Operator::Times => "*",
        }).to_string()
    }
}

impl Operator {
    #[allow(dead_code)]
    fn to_name(&self) -> String {
        (match self {
            Operator::Add => "add",
            Operator::Sub => "sub",
            Operator::Times => "mul",
        }).to_string()
    }

    #[allow(dead_code)]
    fn evaluate(&self, lhs: &i64, rhs: &i64) -> i64 {
        match self {
            Add => lhs + rhs,
            Sub => lhs - rhs,
            Times => lhs * rhs,
        }
    }

    fn is_commutative(&self) -> bool {
        match self {
            Add | Times => true,
            _ => false,
        }
    }
}

impl FromStr for Operator {
    type Err = ();
    fn from_str(s: &str) -> Result<Operator, ()> {
        match s {
            "+" => Ok(Add),
            "-" => Ok(Sub),
            "*" => Ok(Times),
            _ => Err(())
        }
    }
}

#[derive(Parser)]
#[grammar = "domain/grammars/multiplication.pest"]
struct MultiplicationParser;

#[derive(Clone, PartialEq)]
enum Term {
    BinaryOperation(Operator, Rc<SizedTerm>, Rc<SizedTerm>),
    Number(i64, i8), // (a, b) = a x 10^b
}

use Term::{BinaryOperation, Number};

#[derive(Clone, PartialEq)]
struct SizedTerm {
    t: Rc<Term>,
    size: usize
}

impl SizedTerm {
    fn collect_children(&self, v: &mut Vec<SizedTerm>) {
        v.push(self.clone());

        if let BinaryOperation(_, t1, t2) = self.t.borrow() {
            t1.collect_children(v);
            t2.collect_children(v);
        }
    }

    fn replace_at_index(&self, index: usize, new_term: &SizedTerm) -> SizedTerm {
        if index == 0 {
            return new_term.clone();
        }

        match self.t.borrow() {
            BinaryOperation(op, t1, t2) => {
                if index <= t1.size {
                    Self::new_unsized(BinaryOperation(*op,
                                                      Rc::new(t1.replace_at_index(index - 1, new_term)),
                                                      Rc::clone(&t2)))
                } else {
                    Self::new_unsized(BinaryOperation(*op,
                                                      Rc::clone(&t1),
                                                      Rc::new(t2.replace_at_index(index - 1 - t1.size, new_term))))
                }
            },
            _ => unreachable!()
        }
    }

    fn new(t: Term, size: usize) -> SizedTerm {
        SizedTerm { t: Rc::new(t), size: size }
    }

    fn new_unsized(t: Term) -> SizedTerm {
        SizedTerm { t: Rc::new(t), size: 0}
    }
}

impl ToString for SizedTerm {
    fn to_string(&self) -> String {
        self.t.to_string()
    }
}

impl ToString for Term {
    fn to_string(&self) -> String {
        match self {
            Number(n, p) => {
                format!("{}e{}", n, p)
            }
            BinaryOperation(op, t1, t2) => {
                format!("({} {} {})",
                        t1.t.to_string(),
                        op.to_string(),
                        t2.t.to_string())
            }
        }
    }
}

impl FromStr for SizedTerm {
    type Err = PestError<Rule>;

    fn from_str(s : &str) -> Result<SizedTerm, PestError<Rule>> {
        let root = MultiplicationParser::parse(Rule::term, s)?.next().unwrap();

        fn parse_value(pair: Pair<Rule>) -> SizedTerm {
            let rule = pair.as_rule();
            let s = pair.as_str();
            let sub : Vec<Pair<Rule>> = pair.into_inner().collect();

            match rule {
                Rule::number | Rule::neg_number => {
                    let parts = s.split("e").collect::<Vec<&str>>();
                    SizedTerm::new(Number(parts[0].parse::<i64>().unwrap(), parts[1].parse::<i8>().unwrap()), 1)
                }

                Rule::bin_op => {
                    let t1 = Rc::new(parse_value(sub[0].clone()));
                    let t2 = Rc::new(parse_value(sub[2].clone()));
                    let s = t1.size + t2.size + 1;
                    let op = Operator::from_str(sub[1].as_str()).unwrap();
                    SizedTerm::new(BinaryOperation(op, t1, t2), s)
                }

                Rule::term
                | Rule::expr
                | Rule::op
                | Rule::WHITESPACE
                | Rule::EOI => unreachable!(),
            }
        }

        Ok(parse_value(root))
    }
}

impl SizedTerm {
    fn randomize_numbers(&self, rng: &mut Pcg64) -> SizedTerm {
        match self.t.borrow() {
            BinaryOperation(op, t1, t2) =>
                SizedTerm::new(BinaryOperation(*op,
                                               Rc::new(t1.randomize_numbers(rng)),
                                               Rc::new(t2.randomize_numbers(rng))),
                               self.size),

            Number(_, _) => SizedTerm::new(
                Term::Number(
                    rng.gen_range(1..1000),
                    rng.gen_range(1..5),
                ),
                1),
        }
    }

    fn is_solved(&self) -> bool {
        match self.t.borrow() {
            Number(_, _) => true,
            _ => false,
        }
    }
}

pub struct Multiplication {}

impl super::Domain for Multiplication {
    fn name(&self) -> String {
        return "multiplication".to_string();
    }

    fn generate(&self, seed: u64) -> State {
        let mut rng = super::new_rng(seed);
        let i = rng.gen_range(0..MULTIPLICATION_TEMPLATES.len());
        let template = MULTIPLICATION_TEMPLATES[i];
        let term = SizedTerm::from_str(template).unwrap();
        term.randomize_numbers(&mut rng).to_string()
    }

    fn step(&self, state: State) -> Option<Vec<Action>> {
        let t = SizedTerm::from_str(&state).unwrap();
        if t.is_solved() {
            return None;
        }

        let mut actions = Vec::new();

        let mut children = Vec::new();
        let rct = Rc::new(t);
        rct.collect_children(&mut children);

        for local_rewrite_tactic in &[a_commutativity,
                                      a_distributivity,
                                      a_eval,
                                      a_split,
                                      a_trivial] {
            for (i, st) in children.iter().enumerate() {
                if let Some((nt, fd, hd)) = local_rewrite_tactic(st, i) {
                    let next_state = rct.replace_at_index(i, &nt);
                    actions.push((next_state, fd, hd));
                }
            }
        }

        Some(actions.into_iter().map(|(t, fd, hd)| Action { next_state: t.to_string(),
                                                            formal_description: fd,
                                                            human_description: hd }).collect())
    }
}

// Tactics
fn a_commutativity(t: &SizedTerm, i: usize) -> Option<(SizedTerm, String, String)> {
    if let BinaryOperation(op, t1, t2) = t.t.borrow() {
        if op.is_commutative() {
            return Some((SizedTerm::new(BinaryOperation(*op, Rc::clone(t2), Rc::clone(t1)), t.size),
                         format!("comm {}, {}", i, t.to_string()),
                         format!("Commute the terms in {}", t.to_string())))
        }
    }
    None
}

fn is_single_digit(n : i64) -> bool {
    -10 < n && n < 10
}

fn number_power_to_number(mut n : i64, mut p : i8) -> i64 {
    while p > 0 {
        n *= 10;
        p -= 1;
    }
    n
}

fn a_eval(t: &SizedTerm, i: usize) -> Option<(SizedTerm, String, String)> {
    if let BinaryOperation(op, t1, t2) = t.t.borrow() {
        if let (Number(n1, e1), Number(n2, e2)) = (t1.t.borrow(), t2.t.borrow()) {
            if *op == Operator::Times && (is_single_digit(*n1) || is_single_digit(*n2)) {
                return Some((SizedTerm::new_unsized(Number(n1 * n2, e1 + e2)),
                             format!("sdmul {}, {}", i, t.to_string()),
                             format!("Calculate {}", t.to_string())))
            } else if *op == Operator::Add {
                return Some((SizedTerm::new_unsized(Number(number_power_to_number(*n1, *e1) + number_power_to_number(*n2, *e2), 0)),
                             format!("add {}, {}", i, t.to_string()),
                             format!("Calculate {}", t.to_string())))
            }
        }
    }
    None
}

fn a_distributivity(t: &SizedTerm, i: usize) -> Option<(SizedTerm, String, String)> {
    if let BinaryOperation(Operator::Times, t1, t2) = t.t.borrow() {
        if let BinaryOperation(Operator::Add, t3, t4) = t2.t.borrow() {
            return Some((SizedTerm::new_unsized(BinaryOperation(Operator::Add,
                                                                Rc::new(SizedTerm::new_unsized(BinaryOperation(Operator::Times,
                                                                                                       t1.clone(),
                                                                                                       t3.clone()))),
                                                                Rc::new(SizedTerm::new_unsized(BinaryOperation(Operator::Times,
                                                                                                       t1.clone(),
                                                                                                       t4.clone()))))),
                         format!("dist {}, {}", i, t.to_string()),
                         format!("Distribute {}", t.to_string())))
        }
    }
    None
}

fn a_split(t: &SizedTerm, i: usize) -> Option<(SizedTerm, String, String)> {
    if let Number(n, e) = t.t.borrow() {
        return Some((SizedTerm::new_unsized(BinaryOperation(Operator::Add,
                                                            Rc::new(SizedTerm::new_unsized(Number(n / 10, e + 1))),
                                                            Rc::new(SizedTerm::new_unsized(Number(n % 10, *e))))),
                         format!("split {}, {}", i, t.to_string()),
                         format!("Split the number {}", t.to_string())))
    }
    None
}

fn a_trivial(t: &SizedTerm, i: usize) -> Option<(SizedTerm, String, String)> {
    // mul0
    if let BinaryOperation(Operator::Times, _, t2) = t.t.borrow() {
        if let Number(0, _) = t2.t.borrow() {
            return Some((SizedTerm::new_unsized(Number(0, 0)),
                         format!("mul0 {}, {}", i, t.to_string()),
                         format!("Eliminate the multiplication by zero in {}", t.to_string())))
        }
    }

    // mul1
    if let BinaryOperation(Operator::Times, t1, t2) = t.t.borrow() {
        if let Number(1, 0) = t2.t.borrow() {
            return Some(((t1.borrow() as &SizedTerm).clone(),
                         format!("mul1 {}, {}", i, t.to_string()),
                         format!("Eliminate the multiplication by one in {}", t.to_string())))
        }
    }

    // add0
    if let BinaryOperation(Operator::Add, t1, t2) = t.t.borrow() {
        if let Number(0, _) = t2.t.borrow() {
            return Some(((t1.borrow() as &SizedTerm).clone(),
                         format!("add0 {}, {}", i, t.to_string()),
                         format!("Eliminate the addition of zero in {}", t.to_string())))
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use crate::domain::Domain;

    #[test]
    fn test_parsing_templates() {
        for s in super::MULTIPLICATION_TEMPLATES.iter() {
            assert_eq!(super::SizedTerm::from_str(s).unwrap().to_string(), *s);
        }
    }

    #[test]
    fn test_generate() {
        for i in 1..1000 {
            let m = super::Multiplication {};
            println!("{}", m.generate(i));
        }
    }
}

const MULTIPLICATION_TEMPLATES: &[&str] = &[
    "(1e0 * 2e0)",
    "((1e0 * 2e0) + 5e0)",
    "((1e0 * 2e0) + (3e0 * 4e0))",
    "((1e0 + 2e0) * (3e0 + 4e0))",
    "((1e0 * 2e0) * 3e0)",
    "(((1e0 * 2e0) * 3e0) * 4e0)",
];

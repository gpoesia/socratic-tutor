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
    pub fn to_name(&self) -> String {
        (match self {
            Operator::Add => "add",
            Operator::Sub => "sub",
            Operator::Times => "mul",
        }).to_string()
    }

    fn evaluate(&self, lhs: &i32, rhs: &i32) -> i32 {
        match self {
            Add => lhs + rhs,
            Sub => lhs - rhs,
            Times => lhs * rhs,
        }
    }

    fn random(rng: &mut Pcg64) -> Operator {
        let i = rng.gen_range(0..3);
        if i == 0 {
            Add
        } else if i == 1 {
            Sub
        } else {
            Times
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
#[grammar = "domain/grammars/fractions.pest"]
struct FractionsParser;

#[derive(Clone, PartialEq)]
enum Term {
    Fraction(Rc<SizedTerm>, Rc<SizedTerm>),
    FractionOperation(Operator, Rc<SizedTerm>, Rc<SizedTerm>),
    NumberOperation(Operator, Rc<SizedTerm>, Rc<SizedTerm>),
    Number(i32),
}

use Term::{Fraction, FractionOperation, NumberOperation, Number};

#[derive(Clone, PartialEq)]
struct SizedTerm {
    t: Rc<Term>,
    size: usize,
}

impl SizedTerm {
    fn collect_children(&self, v: &mut Vec<SizedTerm>, index: &String, child_indices: &mut Vec<String>) {
        v.push(self.clone());
        child_indices.push(index.clone());

        match self.t.borrow() {
            Fraction(t1, t2) => {
                t1.collect_children(v, &(index.clone() + ".0"), child_indices);
                t2.collect_children(v, &(index.clone() + ".1"), child_indices);
            },
            FractionOperation(_, t1, t2) => {
                t1.collect_children(v, &(index.clone() + ".0"), child_indices);
                t2.collect_children(v, &(index.clone() + ".1"), child_indices);
            },
            NumberOperation(_, t1, t2) => {
                t1.collect_children(v, &(index.clone() + ".0"), child_indices);
                t2.collect_children(v, &(index.clone() + ".1"), child_indices);
            },
            _ => (),
        }
    }

    fn replace_at_index(&self, index: usize, new_term: &SizedTerm) -> SizedTerm {
        if index == 0 {
            return new_term.clone();
        }

        match self.t.borrow() {
            Fraction(t1, t2) => {
                if index <= t1.size {
                    Self::new_unsized(Fraction(Rc::new(t1.replace_at_index(index - 1, new_term)),
                                               Rc::clone(&t2)))
                } else {
                    Self::new_unsized(Fraction(Rc::clone(&t1),
                                               Rc::new(t2.replace_at_index(index - 1 - t1.size, new_term))))
                }
            },
            FractionOperation(op, t1, t2) => {
                if index <= t1.size {
                    Self::new_unsized(FractionOperation(*op,
                                                        Rc::new(t1.replace_at_index(index - 1, new_term)),
                                                        Rc::clone(&t2)))
                } else {
                    Self::new_unsized(FractionOperation(*op,
                                                        Rc::clone(&t1),
                                                        Rc::new(t2.replace_at_index(index - 1 - t1.size, new_term))))
                }
            },
            NumberOperation(op, t1, t2) => {
                if index <= t1.size {
                    Self::new_unsized(NumberOperation(*op,
                                                      Rc::new(t1.replace_at_index(index - 1, new_term)),
                                                      Rc::clone(&t2)))
                } else {
                    Self::new_unsized(NumberOperation(*op,
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

fn format_number(n: i32) -> String {
    if n >= 0 {
        n.to_string()
    } else {
        format!("{}", n.to_string())
    }
}

impl ToString for Term {
    fn to_string(&self) -> String {
        match self {
            Number(n) => {
                format_number(*n)
            },

            Fraction(t1, t2) => {
                format!("[{}]/[{}]", t1.to_string(), t2.to_string())
            }

            FractionOperation(op, t1, t2) => {
                format!("{} {} {}", t1.to_string(), op.to_string(), t2.to_string())
            }

            NumberOperation(op, t1, t2) => {
                format!("({} {} {})", t1.to_string(), op.to_string(), t2.to_string())
            }
        }
    }
}

impl FromStr for SizedTerm {
    type Err = PestError<Rule>;

    fn from_str(s : &str) -> Result<SizedTerm, PestError<Rule>> {
        let root = FractionsParser::parse(Rule::term, s)?.next().unwrap();

        fn parse_value(pair: Pair<Rule>) -> SizedTerm {
            let rule = pair.as_rule();
            let s = pair.as_str();
            let sub : Vec<Pair<Rule>> = pair.into_inner().collect();

            match rule {
                Rule::fraction => {
                    let t1 = Rc::new(parse_value(sub[0].clone()));
                    let t2 = Rc::new(parse_value(sub[1].clone()));
                    let sz = 1 + t1.size + t2.size;
                    SizedTerm::new(Fraction(t1, t2), sz)
                }

                Rule::number | Rule::neg_number => {
                    SizedTerm::new(Number(s.parse::<i32>().unwrap()), 1)
                }

                Rule::number_op => {
                    let t1 = Rc::new(parse_value(sub[0].clone()));
                    let t2 = Rc::new(parse_value(sub[2].clone()));
                    let sz = t1.size + t2.size + 1;
                    let op = Operator::from_str(sub[1].as_str()).unwrap();
                    SizedTerm::new(NumberOperation(op, t1, t2), sz)
                }

                Rule::fraction_op => {
                    let t1 = Rc::new(parse_value(sub[0].clone()));
                    let t2 = Rc::new(parse_value(sub[2].clone()));
                    let sz = t1.size + t2.size + 1;
                    let op = Operator::from_str(sub[1].as_str()).unwrap();
                    SizedTerm::new(FractionOperation(op, t1, t2), sz)
                }

                Rule::term
                | Rule::op
                | Rule::number_expr
                | Rule::WHITESPACE
                | Rule::EOI => unreachable!(),
            }
        }

        Ok(parse_value(root))
    }
}

fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

impl SizedTerm {
    fn is_solved(&self) -> bool {
        match self.t.borrow() {
            Number(_) => true,
            Fraction(a, b) => {
                match (a.t.borrow(), b.t.borrow()) {
                    (Term::Number(n), Term::Number(m)) => gcd(*n, *m) == 1 && *m != 1,
                    _ => false,
                }
            },
            _ => false,
        }
    }
}

pub struct Fractions {
    primes: Vec<i32>,
    max_factors: usize
}

impl Fractions {
    pub fn new(n_primes: usize, max_factors: usize) -> Fractions {
        let mut primes = Vec::new();

        let mut i = 2;
        while primes.len() < n_primes {
            let mut is_prime = true;
            let mut j = 2;

            while j*j <= i {
                if i % j == 0 {
                    is_prime = false;
                    break;
                }
                j += 1;
            }

            if is_prime {
                primes.push(i);
            }

            i += 1;
        }

        Fractions { primes: primes, max_factors: max_factors }
    }

    pub fn random_number(&self, rng: &mut Pcg64) -> i32 {
        let n_factors = rng.gen_range(0..self.max_factors);

        let mut number = 1;

        for _ in 0..n_factors {
            number *= self.primes[rng.gen_range(0..self.primes.len())];
        }

        number
    }


    fn generate_fraction_or_number(&self, rng: &mut Pcg64) -> SizedTerm {
        let kind = rng.gen_range(0..2);

        // Generate a fraction.
        if kind == 0 {
            let num = self.random_number(rng);
            let den = self.random_number(rng);
            SizedTerm::new(
                Fraction(
                    Rc::new(SizedTerm::new(Number(num), 1)),
                    Rc::new(SizedTerm::new(Number(den), 1)),
                ),
                3)
        } else {
            let n = self.random_number(rng);
            SizedTerm::new(Number(n), 1)
        }
    }
}

// Cap size of the expressions to avoid overly large states.
// Otherwise, the number of steps available can explode if the agent takes
// random wrong actions.
const MAX_SIZE: usize = 15;

impl super::Domain for Fractions {
    fn name(&self) -> String {
        return "fractions".to_string();
    }

    fn generate(&self, seed: u64) -> State {
        let mut rng = super::new_rng(seed);
        let kind = rng.gen_range(0..4);

        // Single fraction expression.
        if kind == 0 {
            self.generate_fraction_or_number(&mut rng).to_string()
        } else {
            SizedTerm::new_unsized(FractionOperation(
                Operator::random(&mut rng),
                Rc::new(self.generate_fraction_or_number(&mut rng)),
                Rc::new(self.generate_fraction_or_number(&mut rng)),
            )).to_string()
        }
    }

    fn step(&self, state: State) -> Option<Vec<Action>> {
        let t = SizedTerm::from_str(&state).unwrap();

        if t.is_solved() {
            return None;
        }

        if t.size > MAX_SIZE {
            return Some(Vec::new());
        }

        let mut actions = Vec::new();

        let mut children = Vec::new();
        let mut indexes = Vec::new();
        let rct = Rc::new(t);
        rct.collect_children(&mut children, &String::new(), &mut indexes);

        for local_rewrite_tactic in &[a_factorize,
                                      a_eval,
                                      a_cancel,
                                      a_mul_frac,
                                      a_simpl_div_1,
                                      a_make_frac,
                                      a_merge_fracs,
                                      ] {
            for (i, st) in children.iter().enumerate() {
                for (nt, fd, hd) in local_rewrite_tactic(st, &indexes[i], &self) {
                    let next_state = rct.replace_at_index(i, &nt);
                    actions.push((next_state, fd, hd));
                }
            }
        }

        Some(actions.into_iter().map(|(t, fd, hd)| Action { next_state: t.to_string(),
                                                            formal_description: fd,
                                                            human_description: hd }).collect())
    }

    fn apply(&self, state: State, axiom: &str, _path: &Option<String>) -> Option<Vec<Action>> {
        let t = SizedTerm::from_str(&state).unwrap();

        if t.is_solved() {
            return None;
        }

        if t.size > MAX_SIZE {
            return Some(Vec::new());
        }

        let mut actions = Vec::new();

        let mut children = Vec::new();
        let mut indexes = Vec::new();
        let rct = Rc::new(t);
        rct.collect_children(&mut children, &String::new(), &mut indexes);

        let local_rewrite_tactic : Option<fn(&SizedTerm, &str, &Fractions) ->
                                          Vec<(SizedTerm, String, String)>> = match axiom {
                    "factorize" => Some(a_factorize),
                    "eval" => Some(a_eval),
                    "cancel" => Some(a_cancel),
                    "scale" => Some(a_mul_frac),
                    "simpl1" => Some(a_simpl_div_1),
                    "mfrac" => Some(a_make_frac),
                    "mul" | "combine" => Some(a_merge_fracs),
                    _ => None,
                };


        if let Some(local_rewrite_tactic) = local_rewrite_tactic {
            for (i, st) in children.iter().enumerate() {
                for (nt, fd, hd) in local_rewrite_tactic(st, &indexes[i], &self) {
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
fn a_factorize(st: &SizedTerm, index: &str, _f: &Fractions) -> Vec<(SizedTerm, String, String)> {
    if let Number(n) = st.t.borrow() {
        let mut v = Vec::new();

        let mut i = 2;
        while i*i <= n.abs() {
            if n % i == 0 {
                v.push(
                    (SizedTerm::new(
                        NumberOperation(Times,
                                        Rc::new(SizedTerm::new(Number(i), 1)),
                                        Rc::new(SizedTerm::new(Number(n / i), 1))),
                        3),
                     format!("factorize {}, {}, {}*{}", index, n, i, n / i),
                     format!("Factorize {} as {}*{}", n, i, n / i)
                     ))
            }
            i += 1;
        }

        return v;
    }
    vec![]
}

fn a_eval(st: &SizedTerm, index: &str, _f: &Fractions) -> Vec<(SizedTerm, String, String)> {
    if let NumberOperation(op, a, b) = st.t.borrow() {
        if let (Number(n1), Number(n2)) = (a.t.borrow(), b.t.borrow()) {
            return vec![(SizedTerm::new(Number(op.evaluate(n1, n2)), 1),
                         format!("eval {}, {} {} {}", index, n1, op.to_string(), n2),
                         format!("Calculate {} {} {}", n1, op.to_string(), n2))];
        }
    }
    vec![]
}

fn a_cancel(st: &SizedTerm, index: &str, _f: &Fractions) -> Vec<(SizedTerm, String, String)> {
    let mut v: Vec<(SizedTerm, String, String)>  = Vec::new();

    if let Fraction(num, den) = st.t.borrow() {
        if let NumberOperation(Times, num_a, num_b) = num.t.borrow() {
            if let NumberOperation(Times, den_a, den_b) = den.t.borrow() {
                // [[n*X]]/[[n*Y]]
                if let (Number(num_a_n), Number(den_a_n)) = (num_a.t.borrow(), den_a.t.borrow()) {
                    if num_a_n == den_a_n {
                        v.push((SizedTerm::new_unsized(Fraction(num_b.clone(), den_b.clone())),
                                format!("cancel {}, {}", index, num_a_n),
                                format!("Cancel the common factor {}", num_a_n)));
                    }
                }
                if den_a != den_b {
                    // [[n*X]]/[[Y*n]]
                    if let (Number(num_a_n), Number(den_b_n)) = (num_a.t.borrow(), den_b.t.borrow()) {
                        if num_a_n == den_b_n {
                            v.push((SizedTerm::new_unsized(Fraction(num_b.clone(), den_a.clone())),
                                    format!("cancel {}, {}", index, num_a_n),
                                    format!("Cancel the common factor {}", num_a_n)));
                        }
                    }
                }
                if num_b != num_a {
                    // [[X*n]]/[[n*Y]]
                    if let (Number(num_b_n), Number(den_a_n)) = (num_b.t.borrow(), den_a.t.borrow()) {
                        if num_b_n == den_a_n {
                            v.push((SizedTerm::new_unsized(Fraction(num_a.clone(), den_b.clone())),
                                    format!("cancel {}, {}", index, num_b_n),
                                    format!("Cancel the common factor {}", num_b_n)));
                        }
                    }
                    if den_a != den_b {
                        // [[X*n]]/[[Y*n]]
                        if let (Number(num_b_n), Number(den_b_n)) = (num_b.t.borrow(), den_b.t.borrow()) {
                            if num_b_n == den_b_n {
                                v.push((SizedTerm::new_unsized(Fraction(num_a.clone(), den_a.clone())),
                                        format!("cancel {}, {}", index, num_b_n),
                                        format!("Cancel the common factor {}", num_b_n)));
                            }
                        }
                    }
                }
            }

            if let (Number(num_a_n), Number(den_n)) = (num_a.t.borrow(), den.t.borrow()) {
                // [[n*X]]/[[n]]
                if num_a_n == den_n {
                    v.push((SizedTerm::clone(num_b),
                            format!("cancel {}, {}", index, num_a_n),
                            format!("Cancel the common factor {}", num_a_n)));
                }
            }

            if num_b != num_a {
                if let (Number(num_b_n), Number(den_n)) = (num_b.t.borrow(), den.t.borrow()) {
                    // [[X*n]]/[[n]]
                    if num_b_n == den_n {
                        v.push((SizedTerm::clone(num_a),
                                format!("cancel {}, {}", index, num_b_n),
                                format!("Cancel the common factor {}", num_b_n)));
                    }
                }
            }
        }

        if let NumberOperation(Times, den_a, den_b) = den.t.borrow() {
            if let (Number(num_n), Number(den_a_n)) = (num.t.borrow(), den_a.t.borrow()) {
                if num_n == den_a_n {
                    v.push((SizedTerm::new_unsized(Fraction(Rc::new(SizedTerm::new_unsized(Number(1))),
                                                            den_b.clone())),
                            format!("cancel {}, {}", index, num_n),
                            format!("Cancel the common factor {}", num_n)));
                }
            }
            if den_a != den_b {
                if let (Number(num_n), Number(den_b_n)) = (num.t.borrow(), den_b.t.borrow()) {
                    if num_n == den_b_n {
                        v.push((SizedTerm::new_unsized(Fraction(Rc::new(SizedTerm::new_unsized(Number(1))),
                                                                den_a.clone())),
                                format!("cancel {}, {}", index, num_n),
                                format!("Cancel the common factor {}", num_n)));
                    }
                }
            }
        }
    }
    v
}

fn a_mul_frac(st: &SizedTerm, index: &str, f: &Fractions) -> Vec<(SizedTerm, String, String)> {
    let mut v: Vec<(SizedTerm, String, String)>  = Vec::new();

    if let Fraction(num, den) = st.t.borrow() {
        for prime in f.primes.iter() {
            v.push(
                (SizedTerm::new_unsized(
                    Fraction(
                        Rc::new(SizedTerm::new_unsized(
                            NumberOperation(Times,
                                            Rc::new(SizedTerm::new_unsized(Number(*prime))),
                                            Rc::clone(num)))),
                        Rc::new(SizedTerm::new_unsized(
                            NumberOperation(Times,
                                            Rc::new(SizedTerm::new_unsized(Number(*prime))),
                                            Rc::clone(den)))))),
                    format!("scale {}, {}", index, prime),
                    format!("Multiply both the numerator and denominator by {}", prime)));
        }
    }

    v
}

fn a_simpl_div_1(st: &SizedTerm, index: &str, _f: &Fractions) -> Vec<(SizedTerm, String, String)> {
    if let Fraction(num, den) = st.t.borrow() {
        if let Number(1) = den.t.borrow() {
            return vec![(SizedTerm::clone(num),
                         format!("simpl1 {}", index),
                         format!("Use that a fraction with denominator 1 is equal to its numerator."))];
        }
    }
    vec![]
}

fn a_make_frac(st: &SizedTerm, index: &str, _f: &Fractions) -> Vec<(SizedTerm, String, String)> {
    let mut v = Vec::new();
    if let FractionOperation(op, a, b) = st.t.borrow() {
        if let Number(n) = a.t.borrow() {
            v.push((
                SizedTerm::new_unsized(
                    FractionOperation(*op,
                                      Rc::new(SizedTerm::new_unsized(
                                          Fraction(Rc::clone(a),
                                                   Rc::new(SizedTerm::new_unsized(Number(1)))))),
                                      Rc::clone(b))),
                format!("mfrac {}, {}", index, n),
                format!("Rewrite {} as a fraction.", n)));
        }

        if let Number(n) = b.t.borrow() {
            v.push((
                SizedTerm::new_unsized(
                    FractionOperation(*op,
                                      Rc::clone(a),
                                      Rc::new(SizedTerm::new_unsized(
                                          Fraction(Rc::clone(b),
                                                   Rc::new(SizedTerm::new_unsized(Number(1)))))))),
                format!("mfrac {}, {}", index /* + 1 + a.size */, n),
                format!("Rewrite {} as a fraction.", n)));
        }
    }
    v
}

fn a_merge_fracs(st: &SizedTerm, index: &str, _f: &Fractions) -> Vec<(SizedTerm, String, String)> {
    if let FractionOperation(op, a, b) = st.t.borrow() {
        if let (Fraction(a_num, a_den), Fraction(b_num, b_den)) = (a.t.borrow(), b.t.borrow()) {
            if *op == Times {
                return vec![
                    (SizedTerm::new_unsized(Fraction(
                        Rc::new(SizedTerm::new_unsized(NumberOperation(Times,
                                                                       Rc::clone(a_num),
                                                                       Rc::clone(b_num)))),
                        Rc::new(SizedTerm::new_unsized(NumberOperation(Times,
                                                                       Rc::clone(a_den),
                                                                       Rc::clone(b_den)))),
                    )),
                    format!("mul {}", index),
                    format!("Multiply both fractions"))
                ];
            } else if a_den == b_den {
                return vec![
                    (SizedTerm::new_unsized(Fraction(
                        Rc::new(SizedTerm::new_unsized(NumberOperation(*op,
                                                                       Rc::clone(a_num),
                                                                       Rc::clone(b_num)))),
                        Rc::clone(a_den)
                    )),
                     format!("combine {}", index),
                     format!("Combine the fractions with equal denominator"))
                ];
            }
        }
    }

    vec![]
}

mod tests {
    #[test]
    fn test_generate() {
        use crate::domain::Domain;

        for i in 1..1000 {
            let eq = super::Fractions::new(5, 7);
            println!("{}", eq.generate(i));
        }
    }
}

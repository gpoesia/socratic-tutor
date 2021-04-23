use core::str::FromStr;
use std::borrow::Borrow;
use std::collections::HashSet;
use std::rc::Rc;

use rand::Rng;
use rand_pcg::Pcg64;

use super::{State, Action};
use num_rational::Rational32;
use pest::Parser;
use pest::iterators::Pair;
use pest::error::{Error as PestError};

#[derive(Copy, Clone, Debug, PartialEq)]
enum Operator {
    Add, Sub, Times, Div
}

use Operator::{Add, Sub, Times, Div};

impl ToString for Operator {
    fn to_string(&self) -> String {
        (match self {
            Operator::Add => "+",
            Operator::Sub => "-",
            Operator::Times => "*",
            Operator::Div => "/",
        }).to_string()
    }
}

impl Operator {
    fn to_name(&self) -> String {
        (match self {
            Operator::Add => "add",
            Operator::Sub => "sub",
            Operator::Times => "mul",
            Operator::Div => "div",
        }).to_string()
    }

    fn is_commutative(&self) -> bool {
        match self {
            Add | Times => true,
            _ => false
        }
    }

    fn associates_over(&self, rhs: Operator) -> bool {
        match self {
            Add => rhs == Add || rhs == Sub,
            Times => rhs == Times || rhs == Div,
            _ => false
        }
    }

    fn distributes_over(&self, rhs: Operator) -> bool {
        match self {
            Times | Div => rhs == Add || rhs == Sub,
            _ => false
        }
    }

    fn evaluate(&self, lhs: &Rational32, rhs: &Rational32) -> Rational32 {
        match self {
            Add => lhs + rhs,
            Sub => lhs - rhs,
            Times => lhs * rhs,
            Div => lhs / rhs,
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
            "/" => Ok(Div),
            _ => Err(())
        }
    }
}

#[derive(Parser)]
#[grammar = "domain/grammars/equations.pest"]
struct EquationsParser;

#[derive(Clone, PartialEq)]
enum Term {
    Equality(Rc<SizedTerm>, Rc<SizedTerm>),
    BinaryOperation(Operator, Rc<SizedTerm>, Rc<SizedTerm>),
    UnaryMinus(Rc<SizedTerm>),
    Variable(String),
    AnyNumber,
    Number(Rational32),
}

use Term::{Equality, BinaryOperation, UnaryMinus, Variable, AnyNumber, Number};

#[derive(Clone, PartialEq)]
struct SizedTerm {
    t: Rc<Term>,
    size: usize
}

impl SizedTerm {
    fn is_predicate(&self) -> bool {
        match self.t.borrow() {
            Equality(_, _) => true,
            _ => false,
        }
    }

    fn collect_children(&self, v: &mut Vec<SizedTerm>) {
        v.push(self.clone());

        match self.t.borrow() {
            Equality(t1, t2) => { t1.collect_children(v); t2.collect_children(v); }
            BinaryOperation(_, t1, t2) => { t1.collect_children(v); t2.collect_children(v); }
            UnaryMinus(t) => { t.collect_children(v); }
            _ => (),
        }
    }

    fn replace_at_index(&self, index: usize, new_term: &SizedTerm) -> SizedTerm {
        if index == 0 {
            return new_term.clone();
        }

        match self.t.borrow() {
            Equality(t1, t2) => {
                if index <= t1.size {
                    Self::new_unsized(Equality(Rc::new(t1.replace_at_index(index - 1, new_term)),
                                               Rc::clone(&t2)))
                } else {
                    Self::new_unsized(Equality(Rc::clone(&t1),
                                               Rc::new(t2.replace_at_index(index - 1 - t1.size, new_term))))
                }
            },
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
            UnaryMinus(t) => Self::new_unsized(UnaryMinus(Rc::new(t.replace_at_index(index - 1, new_term)))),
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

fn format_rational(n: &Rational32) -> String {
    if n.is_integer() {
        n.numer().to_string()
    } else {
        format!("[{}/{}]", n.numer(), n.denom())
    }
}

impl ToString for Term {
    fn to_string(&self) -> String {
        match self {
            AnyNumber => "?".to_string(),
            Number(n) => {
                let inner = format_rational(n);
                return if *n < Rational32::new(0, 1) { format!("({})", inner) } else { inner };
            }
            Variable(v) => v.to_string(),
            UnaryMinus(e) => format!("-{}", e.to_string()),
            Equality(t1, t2) => format!("{} = {}", t1.to_string(), t2.to_string()),
            BinaryOperation(op, t1, t2) => {
                match (op, t1.t.borrow(), t2.t.borrow()) {
                    (Times, Number(n), Variable(v)) =>
                        format!("{}{}", format_rational(n), v.to_string()),
                    _ => format!("({} {} {})",
                                 t1.to_string(),
                                 op.to_string(),
                                 t2.to_string()),
                }
            }
        }
    }
}

impl FromStr for SizedTerm {
    type Err = PestError<Rule>;

    fn from_str(s : &str) -> Result<SizedTerm, PestError<Rule>> {
        let root = EquationsParser::parse(Rule::equality, s)?.next().unwrap();

        fn parse_value(pair: Pair<Rule>) -> SizedTerm {
            let rule = pair.as_rule();
            let s = pair.as_str();
            let sub : Vec<Pair<Rule>> = pair.into_inner().collect();

            match rule {
                Rule::equality => {
                    let t1 = Rc::new(parse_value(sub[0].clone()));
                    let t2 = Rc::new(parse_value(sub[1].clone()));
                    let s = 1 + t1.size + t2.size;
                    SizedTerm::new(Equality(t1, t2), s)
                }

                Rule::any_number => SizedTerm::new(AnyNumber, 1),

                Rule::number | Rule::neg_number => {
                    SizedTerm::new(Number(Rational32::new(s.parse::<i32>().unwrap(), 1)), 1)
                }

                Rule::neg_var => {
                    let st = Rc::new(SizedTerm::new(Variable(sub[0].as_str().to_string()), 1));
                    SizedTerm::new(UnaryMinus(st), 2)
                }

                Rule::int_frac => {
                    SizedTerm::new(
                        Number(Rational32::new(sub[0].as_str().parse::<i32>().unwrap(),
                                               sub[1].as_str().parse::<i32>().unwrap())), 1)
                }

                Rule::variable => SizedTerm::new(Variable(s.to_string()), 1),
                Rule::sum_or_sub | Rule::prod_or_div => {
                    let t1 = Rc::new(parse_value(sub[0].clone()));
                    let t2 = Rc::new(parse_value(sub[2].clone()));
                    let s = t1.size + t2.size + 1;
                    let op = Operator::from_str(sub[1].as_str()).unwrap();
                    SizedTerm::new(BinaryOperation(op, t1, t2), s)
                }

                Rule::varcoeff => {
                    let t1 = Rc::new(parse_value(sub[0].clone()));
                    let t2 = Rc::new(parse_value(sub[1].clone()));
                    let s = t1.size + t2.size + 1;
                    SizedTerm::new(BinaryOperation(Times, t1, t2), s)
                }

                Rule::term
                | Rule::predicate
                | Rule::expr
                | Rule::expr_l1
                | Rule::expr_l2
                | Rule::expr_l3
                | Rule::expr_l4
                | Rule::sub_sum_op
                | Rule::prod_div_op
                | Rule::paren_expr
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
            Equality(t1, t2) => SizedTerm::new(Equality(Rc::new(t1.randomize_numbers(rng)),
                                                        Rc::new(t2.randomize_numbers(rng))),
                                               self.size),
            BinaryOperation(op, t1, t2) =>
                SizedTerm::new(BinaryOperation(*op,
                                               Rc::new(t1.randomize_numbers(rng)),
                                               Rc::new(t2.randomize_numbers(rng))),
                               self.size),

            UnaryMinus(t) => SizedTerm::new(UnaryMinus(Rc::new(t.randomize_numbers(rng))),
                                            self.size),

            Number(_) => SizedTerm::new(
                Term::Number(
                    Rational32::new(
                        if rng.gen_bool(0.5) {
                            rng.gen_range(1..11)
                        } else {
                            rng.gen_range((-10)..0)
                        },
                        1)
                ),
                1),

            _ => self.clone()
        }
    }

    fn is_solved(&self) -> bool {
        match self.t.borrow() {
            Equality(t1, t2) => {
                match (t1.t.borrow(), t2.t.borrow()) {
                    (Term::Variable(_), Term::Number(_)) => true,
                    _ => false,
                }
            },
            _ => false,
        }
    }
}

pub struct Equations {}

// Cap size of the equations to avoid overly large states.
// Otherwise, the number of steps available can grow out of control.
const MAX_SIZE: usize = 30;

impl super::Domain for Equations {
    fn name(&self) -> String {
        return "equations".to_string();
    }

    fn generate(&self, seed: u64) -> State {
        let mut rng = super::new_rng(seed);
        let i = rng.gen_range(0..COGNITIVE_TUTOR_TEMPLATES.len());
        let template = COGNITIVE_TUTOR_TEMPLATES[i];
        let term = SizedTerm::from_str(template).unwrap();
        term.randomize_numbers(&mut rng).to_string()
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

        for tactic in &[a_reflexivity] {
            actions.extend(tactic(&t));
        }

        let mut children = Vec::new();
        let rct = Rc::new(t);
        rct.collect_children(&mut children);

        for local_rewrite_tactic in &[a_commutativity,
                                      a_associativity,
                                      a_distributivity,
                                      a_eval,
                                      a_cancel_ops,
                                      a_identity_ops] {
            for (i, st) in children.iter().enumerate() {
                if let Some((nt, fd, hd)) = local_rewrite_tactic(st, i) {
                    let next_state = rct.replace_at_index(i, &nt);
                    actions.push((next_state, fd, hd));
                }
            }
        }

        let mut seen_before = HashSet::new();

        // Apply an operation to both sides.
        for st in children.iter() {
            if st.size <= 3 &&
                !st.is_predicate() &&
                *(st.t.borrow() as &Term) != Number(Rational32::from_integer(0)) {
                for op in &[Add, Sub, Times, Div] {
                    let (next_state, fd, hd) = a_op_both_sides(Rc::clone(&rct), *op, st);
                    if seen_before.insert(fd.clone()) {
                        actions.push((next_state, fd, hd));
                    }
                }
            }
        }

        Some(actions.into_iter().map(|(t, fd, hd)| Action { next_state: t.to_string(),
                                                            formal_description: fd,
                                                            human_description: hd }).collect())
    }
}

// Tactics

fn a_reflexivity(st: &SizedTerm) -> Vec<(SizedTerm, String, String)> {
    if let Equality(t1, t2) = st.t.borrow() {
        return vec![(SizedTerm::new(Equality(Rc::clone(&t2), Rc::clone(&t1)), st.size),
                     String::from("refl"),
                     String::from("Swap both sides of the equation"))];
    }
    vec![]
}

fn a_op_both_sides(rct: Rc<SizedTerm>, op: Operator, st: &SizedTerm) -> (SizedTerm, String, String) {
    if let Equality(t1, t2) = rct.t.borrow() {
        return (
            SizedTerm::new_unsized(
                Equality(
                    Rc::new(SizedTerm::new(BinaryOperation(op, Rc::clone(t1), Rc::new(st.clone())), t1.size + 2)),
                    Rc::new(SizedTerm::new(BinaryOperation(op, Rc::clone(t2), Rc::new(st.clone())), t2.size + 2)),
                )),
            format!("{} {}", op.to_name(), st.to_string()),
            format!("{} {}", op.to_name(), st.to_string())
        )
    }
    unreachable!()
}

fn a_commutativity(t: &SizedTerm, i: usize) -> Option<(SizedTerm, String, String)> {
    if let BinaryOperation(op, t1, t2) = t.t.borrow() {
        if op.is_commutative() {
            return Some((SizedTerm::new(BinaryOperation(*op, Rc::clone(t2), Rc::clone(t1)), t.size),
                         format!("comm {}, {}", i, t.to_string()),
                         format!("Swap the terms in {}", t.to_string())))
        }
        if *op == Sub {
            if let BinaryOperation(Sub, t3, t4) = t1.t.borrow() {
                return Some((SizedTerm::new(BinaryOperation(Sub,
                                                            Rc::new(SizedTerm::new(
                                                                BinaryOperation(Sub,
                                                                                Rc::clone(t3),
                                                                                Rc::clone(t2)),
                                                                t3.size + t2.size + 1
                                                            )),
                                                            Rc::clone(t4)),
                                            t.size),
                             format!("sub_comm {}, {}", i, t.to_string()),
                             format!("Swap the order of subtractions in {}", t.to_string())))
            }

            if let BinaryOperation(Sub, t3, t4) = t2.t.borrow() {
                return Some((SizedTerm::new(BinaryOperation(Sub, Rc::clone(t1),
                                                            Rc::new(SizedTerm::new(
                                                                BinaryOperation(Sub,
                                                                                Rc::clone(t4),
                                                                                Rc::clone(t3)), t2.size))),
                                            t.size),
                             format!("sub_comm {}, {}", i, t.to_string()),
                             format!("Swap the order of subtractions in {}", t.to_string())))
            }
        }
    }
    None
}

fn a_associativity(t: &SizedTerm, i: usize) -> Option<(SizedTerm, String, String)> {
    if let BinaryOperation(op1, t1, t2) = t.t.borrow() {
        // t1 op1 (t3 op2 t4) => (t1 op1 t3) op2 t4
        if let BinaryOperation(op2, t3, t4) = t2.t.borrow() {
            if op1.associates_over(*op2) {
                return Some((SizedTerm::new(
                    BinaryOperation(*op2,
                                    Rc::new(SizedTerm::new(BinaryOperation(*op1, Rc::clone(t1), Rc::clone(t3)),
                                                           1 + t1.size + t3.size)),
                                    Rc::clone(t4)),
                    t.size),
                             format!("assoc {}, {}", i, t.to_string()),
                             format!("Change the order of operations in {}", t.to_string())));
            }
        }
        // (t3 op2 t4) op1 t2 => t3 op2 (t4 op1 t2)
        if let BinaryOperation(op2, t3, t4) = t1.t.borrow() {
            if op2.associates_over(*op1) {
                return Some((SizedTerm::new(
                    BinaryOperation(*op2,
                                    Rc::clone(t3),
                                    Rc::new(SizedTerm::new(BinaryOperation(*op1, Rc::clone(t4), Rc::clone(t2)),
                                                           1 + t4.size + t2.size))),
                                    t.size),
                             format!("assoc {}, {}", i, t.to_string()),
                             format!("Change the order of operations in {}", t.to_string())));
            }
        }
    }
    None
}

fn a_distributivity(t: &SizedTerm, i: usize) -> Option<(SizedTerm, String, String)> {
    if let BinaryOperation(op1, t1, t2) = t.t.borrow() {
        if let BinaryOperation(op2, t3, t4) = t2.t.borrow() {
            // Forward direction: 5*(x + 2) => 5x + 5*2
            // t1 op1 (t3 op2 t4) => (t1 op1 t3) op2 (t1 op2 t4)
            if op1.distributes_over(*op2) {
                return Some((
                    SizedTerm::new(
                        BinaryOperation(
                            *op2,
                            Rc::new(SizedTerm::new(BinaryOperation(*op1, Rc::clone(t1), Rc::clone(t3)),
                                                   1 + t1.size + t3.size)),
                            Rc::new(SizedTerm::new(BinaryOperation(*op1, Rc::clone(t1), Rc::clone(t4)),
                                                   1 + t1.size + t4.size))),
                        3 + 2*t1.size + t3.size + t4.size),
                    format!("dist {}, {}", i, t.to_string()),
                    format!("Apply distributivity in {}", t.to_string())));
            }

            // Inverse direction: 5x + 3x => (5 + 3)x
            // (t5 op3 t4) op1 (t3 op3 t4) => (t5 op1 t3) op3 t4
            if let BinaryOperation(op3, t5, t6) = t1.t.borrow() {
                if *op3 == *op2 && *t4 == *t6 && op2.distributes_over(*op1) {
                return Some((
                    SizedTerm::new(
                        BinaryOperation(
                            *op3,
                            Rc::new(SizedTerm::new(BinaryOperation(*op1, Rc::clone(t5), Rc::clone(t3)),
                                                   1 + t5.size + t3.size)),
                            Rc::clone(t4)),
                        2 + t5.size + t3.size + t4.size),
                    format!("dist {}, {}", i, t.to_string()),
                    format!("Apply distributivity in {}", t.to_string())));
                }
            }
        }
    }
    None
}

fn a_eval(t: &SizedTerm, i: usize) -> Option<(SizedTerm, String, String)> {
    if let BinaryOperation(op, t1, t2) = t.t.borrow() {
        if let (Number(n1), Number(n2)) = (t1.t.borrow(), t2.t.borrow()) {
            if *op != Div || !n2.is_integer() || n2.to_integer() != 0 {
                return Some((SizedTerm::new(Number(op.evaluate(n1, n2)), 1),
                             format!("eval {}, {}", i, t.to_string()),
                             format!("Calculate {}", t.to_string())))
            }
        }
    }
    None
}

fn a_identity_ops(t: &SizedTerm, i: usize) -> Option<(SizedTerm, String, String)> {
    if let BinaryOperation(Add, t1, t2) = t.t.borrow() {
        if let Number(n2) = t2.t.borrow() {
            if *n2.numer() == 0 {
                return Some((((t1.borrow() as &SizedTerm).clone()),
                             format!("add0 {}, {}", i, t.to_string()),
                             format!("Remove the addition of 0")))
            }
        }
        if let Number(n1) = t1.t.borrow() {
            if *n1.numer() == 0 {
                return Some((((t2.borrow() as &SizedTerm).clone()),
                             format!("add0 {}, {}", i, t.to_string()),
                             format!("Remove the addition of 0")))
            }
        }
    }
    if let BinaryOperation(Sub, t1, t2) = t.t.borrow() {
        if let Number(n2) = t2.t.borrow() {
            if *n2.numer() == 0 {
                return Some((((t1.borrow() as &SizedTerm).clone()),
                             format!("sub0 {}, {}", i, t.to_string()),
                             format!("Remove the subtraction of 0")))
            }
        }
    }
    if let BinaryOperation(Times, t1, t2) = t.t.borrow() {
        if let Number(n2) = t2.t.borrow() {
            if *n2 == Rational32::from_integer(1) {
                return Some((((t1.borrow() as &SizedTerm).clone()),
                             format!("mul1 {}, {}", i, t.to_string()),
                             format!("Remove the multiplication by one")))
            }
        }
        if let Number(n1) = t1.t.borrow() {
            if *n1 == Rational32::from_integer(1) {
                return Some((((t2.borrow() as &SizedTerm).clone()),
                             format!("mul1 {}, {}", i, t.to_string()),
                             format!("Remove the multiplication by one")))
            }
        }
    }
    if let BinaryOperation(Div, t1, t2) = t.t.borrow() {
        if let Number(n2) = t2.t.borrow() {
            if *n2 == Rational32::from_integer(1) {
                return Some((((t1.borrow() as &SizedTerm).clone()),
                             format!("div1 {}, {}", i, t.to_string()),
                             format!("Remove the division by one")))
            }
        }
    }

    None
}


fn a_cancel_ops(t: &SizedTerm, i: usize) -> Option<(SizedTerm, String, String)> {
    if let BinaryOperation(Div, t1, t2) = t.t.borrow() {
        if t1 == t2 {
            return Some((SizedTerm::new(Number(Rational32::from_integer(1)), 1),
                         format!("div_self {}, {}", i, t.to_string()),
                         // Technically would need to assume its non-zero, but this is
                         // enough for this domain.
                         format!("Use that anything over itself is one")))
        }
    }
    if let BinaryOperation(Sub, t1, t2) = t.t.borrow() {
        if t1 == t2 {
            return Some((SizedTerm::new(Number(Rational32::from_integer(0)), 1),
                         format!("sub_self {}, {}", i, t.to_string()),
                         format!("Use that anything minus itself is zero")))
        }
    }
    None
}

mod tests {
    // use std::str::FromStr;
    // use crate::domain::Domain;

    #[test]
    fn test_parsing_cognitive_tutor_templates() {
        for s in super::COGNITIVE_TUTOR_TEMPLATES.iter() {
            assert_eq!(super::SizedTerm::from_str(s).unwrap().to_string(), *s);
        }
    }

    #[test]
    fn test_generate() {
        for i in 1..1000 {
            let eq = super::Equations {};
            println!("{}", eq.generate(i));
        }
    }

    #[test]
    fn test_step() {
        let eq = super::Equations {};
        for s in &["x + 3 = (1 - 2) - 3"] {
            println!("Stepping {}", s);
            eq.step(s.to_string());
        }
    }
}

const COGNITIVE_TUTOR_TEMPLATES: &[&str] = &[
    "(1 / 2) = x",
    "(((-1) / x) * x) = -2x",
    "((-1) + 2x) = (3x + (-4))",
    "(-1x + (-2)) = (-3x + (-4))",
    "(-1x + 2) = (-3x + 4)",
    "(1x - 2) = -3x",
    "(1x + 2) = x",
    "1x = (((-2) + 3x) + 4)",
    "(-1) = ((-2) + ((-3) / x))",
    "(((-1) - (2 / x)) * x) = -3x",
    "(-1x / x) = -2x",
    "(-1x + (-2)) = ((-3) + -4x)",
    "((-1x - 2) + 3x) = (4x + 5x)",
    "(1x + (-2)) = ((-3) + 4x)",
    "((x - 1) - 2x) = 3x",
    "(1x - 2x) = (((3 - 4x) - 5) - 6x)",
    "((1 / x) + 2) = (-3)",
    "(-1) = ((2 / x) + 3)",
    "(1x + (-2)) = (-3x + 4)",
    "((-1x + 2) - 3) = -4x",
    "(-1) = (2 + 3x)",
    "(1x + (-2)) = (3 + -4x)",
    "((-1) / 2) = x",
    "((-1) / x) = (-2)",
    "(1x - 2) = ((-3) - 4x)",
    "-1x = (2x + 3)",
    "(1 + -2x) = (-3x + (-4))",
    "(1 + -2x) = ((-3) + -4x)",
    "((-1) + (2 / x)) = 3",
    "(-1x + 2x) = (-3)",
    "(1 - (-2)) = (((3 / x) + (-4)) - (-5))",
    "(1 / x) = 2",
    "((-1) + -2x) = (-3x + (-4))",
    "((-1x + 2) - 3x) = 4",
    "1 = ((-2) + -3x)",
    "((-1) - 2x) = (3x - 4)",
    "(((-1x - 2) / 3) + ((4x + 5) / 6)) = (x + ((7x + 8) / 9))",
    "((1x - 2) + 3) = ((4x - 5) + 6)",
    "1 = ((2 - 3x) - 4x)",
    "((1 + 2) * x) = 3",
    "1 = (2 + (3 / x))",
    "x = (-1x + 2)",
    "1 = (2x + 3)",
    "(1 + 2x) = (3x + 4)",
    "1 = (((-2) / x) + (-3))",
    "((-1) + -2x) = (-3)",
    "((-1) + 2x) = (-3x + (-4))",
    "(1x - -2x) = 3",
    "((-1) + 2x) = (3x - 4)",
    "(1 / x) = (2 + 3)",
    "(-1x + 2) = (3x + (-4))",
    "(-1x + 2x) = ((-3) + 4)",
    "1x = (2 + 3)",
    "((-1) - 2x) = (3 - 4x)",
    "(1 / 2) = (((-3) + 4x) / 5)",
    "1 = -2x",
    "((-1) + -2x) = ((-3) + -4x)",
    "-1x = (-2)",
    "((-1) + -2x) = (3 + -4x)",
    "(((-1) / x) + (-2)) = 3",
    "1 = ((-2) + (3 / x))",
    "(((-1) / x) + 2) = 3",
    "(1x + 2) = (3 - 4x)",
    "(1 + -2x) = 3",
    "(((-1) + 2x) + 3x) = ((-4x + (-5)) + 6x)",
    "1x = (2x + 3)",
    "((-1x - 2) + 3) = 4x",
    "((1 - 2) / 3) = x",
    "(-1) = (2 + ((-3) / x))",
    "-1x = -2x",
    "(1x + (-2)) = (-3x + (-4))",
    "(-1x + (-2)) = (-3)",
    "(-1x - 2) = (3 + 4x)",
    "(-1x + 2) = (-3)",
    "((-1) + 2x) = 3",
    "(-1x + (-2)) = ((-3) + 4x)",
    "(-1x + 2) = ((-3) + 4x)",
    "((-1) + -2x) = (-3x + 4)",
    "((-1) - 2x) = (-3)",
    "(-1) = (-2x / (-3))",
    "1x = (((2 - 3x) - 4) + 5x)",
    "(((-1) / x) + (-2)) = (-3)",
    "(1x + (-2)) = ((-3) + -4x)",
    "1x = (-2x - 3)",
    "(-1) = x",
    "((1 + 2x) + 3x) = ((-4x + (-5)) + 6x)",
    "((1 - (2 / x)) - 3) = (4 - 5)",
    "x = ((1 - 2) / (-3))",
    "(1 + -2x) = ((-3) + 4x)",
    "1x = (2x / x)",
    "(1 + -2x) = (-3)",
    "(-1x + 2) = (3 + -4x)",
    "1 = (2x + (-3))",
    "1 = ((2 / x) + (-3))",
    "(-1x + 2) = (3x + 4)",
    "(1x - -2x) = (-3)",
    "(-1x + 2) = 3",
    "(1 + -2x) = (3 + 4x)",
    "(-1) = ((2x + 3x) - 4)",
    "((-1x - 2) - 3x) = (-4)",
    "-1x = ((-2) - 3)",
    "(-1) = ((-2) + (3 / x))",
    "x = (((-1) + 2) / 3)",
    "((1 + 2x) - 3) = -4x",
    "-1x = ((2 / x) * x)",
    "((-1) - 2) = (3x - 4x)",
    "1 = (((-2) / x) + 3)",
    "1x = 2",
    "(1 - 2) = 3x",
    "(1x + 2) = ((-3) + 4x)",
    "(-1x + 2) = ((-3) + -4x)",
    "((-1) + 2) = 3x",
    "((-1) + -2x) = (3 + 4x)",
    "-1x = (2 + 3)",
    "((-1) + 2x) = ((-3) + 4x)",
    "(1 + 2x) = 3",
    "(((-1) / x) * x) = 2x",
    "((-1) + 2x) = (-3)",
    "(1 + 2x) = (-3x + 4)",
    "((-1) - 2) = (((3 / x) + 4) - 5)",
    "-1x = ((-2) + x)",
    "(-1x + 2) = 3x",
    "1 = (2 + 3x)",
    "(1x - 2) = ((-3) + 4x)",
    "-1x = (((-2) + 3x) + 4)",
    "(-1) = ((-2) + 3x)",
    "1 = (((2 / x) - 3) + 4)",
    "(1x + 2) = (-3)",
    "(-1) = ((2 / x) + (-3))",
    "(1x + 2) = (-3x + (-4))",
    "1 = (2x / 3)",
    "(-1x + 2) = (3 + 4x)",
    "1 = (2x - 3x)",
    "1 = ((-2) + ((-3) / x))",
    "(1 - 2x) = -3x",
    "(1 + ((-2) / x)) = (-3)",
    "x = ((1 + 2) / 3)",
    "(-1) = ((-2) + -3x)",
    "(1 + (2 / x)) = 3",
    "((-1) + 2x) = (3 + 4x)",
    "((1 + 2x) + 3) = 4x",
    "(1 + 2x) = (3 + 4x)",
    "((-1x - 2) + 3x) = ((-4x + 5) + 6x)",
    "1x = ((-2) - 3x)",
    "(1x + 2) = (3x + 4)",
    "(((-1) - 2x) + 3) = -4x",
    "((-1x + 2) - 3x) = (((-4) + 5x) - 6x)",
    "(1x + 2) = ((-3) + -4x)",
    "((-1) + 2x) = ((-3) + -4x)",
    "((1 / x) + (-2)) = (-3)",
    "(1x / 2) = (3 / 4)",
    "((-1) - 2x) = -3x",
    "(1x + 2) = (-3x + 4)",
    "((-1) + ((-2) / x)) = (-3)",
    "((1 + 2x) - 3) = 4x",
    "((-1) + -2x) = (3x + (-4))",
    "(((-1) + 2x) + 3x) = ((4 + -5x) + 6x)",
    "-1x = ((-2) + 3)",
    "(-1x / (-2)) = (3 / (-4))",
    "(-1) = (-2x + 3)",
    "(1 + 2x) = ((-3) + 4x)",
    "(1 + 2x) = (3 + -4x)",
    "(-1) = (((-2) / x) + 3)",
    "-1x = (2x - 3)",
    "((1 + -2x) + 3x) = ((-4x + 5) + 6x)",
    "((-1) + -2x) = (3x + 4)",
    "(1x + (-2)) = (3x + (-4))",
    "1 = (2 + ((-3) / x))",
    "((1 / x) + (-2)) = 3",
    "(-1x - 2) = (3 - 4x)",
    "-1x = ((2 + -3x) - 4)",
    "1 = (-2x + (-3))",
    "(1 - 2) = (-3x - 4x)",
    "((-1) - 2) = -3x",
    "1 = (-2x + 3)",
    "1 = ((-2) / x)",
    "(-1) = (2x + 3)",
    "1 = 2x",
    "(1 - 2) = -3x",
    "(1 + 2x) = -3x",
    "(1x - 2) = 3x",
    "(1x / x) = 2x",
    "(-1x + (-2)) = (3x + 4)",
    "((1x - 2) + 3) = 4x",
    "(-1) = ((2x + 3) + 4x)",
    "(1x + 2) = (3x + (-4))",
    "(1 + -2x) = (3x + 4)",
    "(((-1) + 2x) + 3) = ((4 + 5x) + 6)",
    "((-1) / x) = ((-2) - 3)",
    "1x = (-2)",
    "((-1x + (-2)) - 3) = 4x",
    "1x = (2 - 3)",
    "((-1) + -2x) = ((-3) + 4x)",
    "(-1) = (-2x + (-3))",
    "(-1x + (-2)) = 3",
    "(1x + 2) = (3 + 4x)",
    "((1x + 2) + 3x) = 4",
    "(-1x + 2) = (-3x - 4)",
    "1 = ((-2x + 3) + 4x)",
    "(1 + -2x) = (-3x + 4)",
    "-1x = (2 - 3)",
    "(((-1) / x) + 2) = (-3)",
    "(-1) = (((-2) / x) + (-3))",
    "((-1) + 2) = (3 / x)",
    "(1 + 2x) = (-3x + (-4))",
    "(-1) = (2x - 3)",
    "-1x = ((-2x + 3) + 4)",
    "((1 - 2x) - 3) = ((4 + 5x) - 6)",
    "(1x + (-2)) = (3 + 4x)",
    "(((-1) + 2x) + 3) = (4 + 5)",
    "((-1x + 2) + 3x) = (((-4) - 5x) + 6x)",
    "(-1) = (-2x - 3)",
    "((1x + 2) - 3x) = (4x - 5x)",
    "(-1) = (2 + -3x)",
    "1x = (2 + 3x)",
    "((-1) + 2x) = (3x + 4)",
    "(-1x + (-2)) = (3 + 4x)",
    "(-1) = ((2x - 3) + 4x)",
    "1 = (2 - (3 / x))",
    "(1 + 2x) = (3x + (-4))",
    "((-1x - 2) + 3x) = 4",
    "(1 + -2x) = (3x + (-4))",
    "1 = ((-2) + 3x)",
    "(1 + 2x) = (-3)",
    "((1 / x) + 2) = 3",
    "(1 + (2 / x)) = (-3)",
    "((-1) - 2x) = (-3x - 4)",
    "((-1) + ((-2) / x)) = 3",
    "(1x + 2) = 3",
    "((-1) + (2 / x)) = (-3)",
    "1 = (2 + -3x)",
    "((-1) - 2x) = ((-3) - 4x)",
    "(-1x + (-2)) = (3x + (-4))",
    "(-1) = (-2x + 3x)",
    "(1 + ((-2) / x)) = 3",
    "-1x = ((2x - 3) - 4)",
    "((-1) + 2x) = (-3x - 4)",
    "(-1x + (-2)) = (3 + -4x)",
    "(-1) = (((-2) - 3x) + 4x)",
    "(-1) = 2x",
    "(-1) = -2x",
    "-1x = (((-2) + -3x) - 4)",
    "(((1 / x) - 2) * x) = -3x",
    "(-1) = (2 + (3 / x))",
    "((1 + 2x) + 3x) = 4",
    "((-1) + 2) = (((3 / x) - 4) + 5)",
    "(1 + 2x) = ((-3) + -4x)",
    "1x = (-2x + 3)",
    "(1x + 2) = (3 + -4x)",
    "(1 / (-2)) = x",
    "(-1x + 2) = (-3x + (-4))",
    "(1 / x) = ((-2) - 3)",
    "(-1x - 2x) = ((-3) + 4)",
    "((-1x + 2) + 3) = -4x",
    "(1 + -2x) = (3 + -4x)",
    "-1x = ((-2) + 3x)",
    "((-1) + 2x) = (3 + -4x)",
    "(1x + (-2)) = (3x + 4)",
    "-1x = 2",
    "(-1x + (-2)) = (-3x + 4)",
    "((-1) / (-2)) = x",
    "(1 / x) = (-2)",
    "((-1) + -2x) = 3",
    "((-1) - 2) = ((-3x + 4) - 5)",
    "(-1x - 2) = 3x",
    "1 = ((2 / x) + 3)",
    "1x = ((-2) - 3)",
    "(1x - 2) = (3x - 4)",
    "(-1x + 2x) = ((-3) - 4)",
    "(-1) = (2x + (-3))",
    "(((-1) + -2x) - -3x) = (((-4) + -5x) - -6x)",
    "(1x - 2) = (-3)",
    "(1x + (-2)) = 3",
    "(((-1) - 2x) - 3) = 4x",
    "(((1 / x) * x) + 2x) = 3x",
    "(((-1) - 2x) + 3x) = 4",
    "(-1x + 2) = ((-3) - 4x)",
    "x = ((-1) / (-2))",
    "(1x + (-2)) = (-3)",
    "x = ((-1) / 2)",
    "((-1) / x) = 2",
    "((-1x - 2) - 3x) = 4",
    "1x = (2x - 3)",
    "((-1x - 1x) + 2) = (-3)",
    "(1x - -2x) = ((-3x + 4) - -5x)",
    "1x = ((2x - 3) - 4)",
    "(1 - 2x) = (3x - 4)",
    "-1x = ((2 - 3x) + 4)",
    "((-1) + 2x) = (-3x + 4)",
    "x = (1 / 2)",
];

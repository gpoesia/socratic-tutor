; Some examples of the solver finding solutions.
#lang racket

(require "terms.rkt")
(require "solver.rkt")
(require "tactics.rkt")

(define (solve-for variable)
  (Predicate 'Eq (list (Variable variable) AnyNumber)))

(define sr
  (find-solution
    (list (solve-for 'x))
    (list
      (Predicate
        'Eq 
        (list
          (BinOp op+ (Number 12) (BinOp op* (Number 5) (Number 7)))
          (BinOp op+ (Number 3) (BinOp op+ (Variable 'x) (Number -3))))))
    s:cycle
    5))

(printf
  "Solver result:\n  Facts: ~a\n\n  Met goals: ~a\n\n  Unmet goals: ~a\n\n"
    (string-join
      (map format-term (SolverResult-facts sr)) "\n")
    (string-join (map format-term (SolverResult-met-goals sr)) "\n")
    (string-join (map format-term (SolverResult-unmet-goals sr)) "\n")
    )

; Some examples of the solver finding solutions.
#lang racket

(require "terms.rkt")
(require "term-parser.rkt")
(require "solver.rkt")
(require "tactics.rkt")
(require "debug.rkt")

(define (solve-for variable)
  (Predicate 'Eq (list (Variable variable) AnyNumber)))

(define (run-example facts-str goals-str)
  (define facts (map parse-term facts-str))
  (define goals (map parse-term goals-str))
  (printf "Solving:\n~a\nWith goals ~a\n"
          (string-join (map format-term facts) "\n")
          (string-join (map format-term goals) ", "))
  (define sr (find-solution goals facts s:all
                            (prune:keep-smallest-k 200) 30))
  (define succeeded? (empty? (SolverResult-unmet-goals sr)))
  (printf
    "Solver ~a:\n  Facts: ~a\n\n  Met goals: ~a\n\n  Unmet goals: ~a\n\n"
    (if succeeded? "succeeded" "failed")
    (if succeeded? (length (SolverResult-facts sr))
      (string-join (map format-term-debug (SolverResult-facts sr)) "\n"))
    (string-join (map (lambda (g)
                        (format "~a [solves ~a]"
                                (format-term (goal-solution g sr))
                                (format-term g)))
                      (SolverResult-met-goals sr)) ", ")
    (string-join (map format-term-debug (SolverResult-unmet-goals sr)) ", ")
    ))

(run-example (list "7x - 15 = 6x + 1") (list "x = ?"))
(run-example (list "3x - 3 - 2x = 3") (list "x = ?"))
(run-example (list "3 + (x + -3) = 12 + 5*7") (list "x = ?"))
(run-example (list "2x + 1 = 5") (list "x = ?"))
(run-example (list "x + 1 - 1 = 2") (list "x = ?"))
(run-example (list "x + 1 = 4") (list "x = ?"))
(run-example (list "x = 1 + 2 + 3") (list "x = ?"))
(run-example (list "-1 + x + 1 = 1 + 2 + 3") (list "x = ?"))
(run-example (list "10x - 9x = 10") (list "x = ?"))

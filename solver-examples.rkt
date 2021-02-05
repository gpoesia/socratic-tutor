; Some examples of the solver finding solutions.
#lang racket

(require "terms.rkt")
(require "term-parser.rkt")
(require "solver.rkt")
(require "tactics.rkt")
(require "facts.rkt")
(require "debug.rkt")
(require "serialize.rkt")
(require "questions.rkt")

(define show-questions? #t)

(define (generate-all-questions sr)
  (define facts (get-step-by-step-solution sr))
  (for-each (lambda (f)
              (printf "===> Q: ~a\nA: ~a\n"
                      (generate-leading-question (Fact-proof f) facts)
                      (format-fact-i f)))
            (cdr facts)))

(define (run-example facts-str goals-str)
  (define facts (map (compose assumption parse-term) facts-str))
  (define goals (map parse-term goals-str))
  (printf "Solving:\n~a\nWith goals ~a\n"
          (string-join (map format-fact facts) "\n")
          (string-join (map format-term goals) ", "))
  (define sr (from-json-string (to-json-string 
    (find-solution goals facts s:all (prune:keep-smallest-k 10) 7))))
  (define succeeded? (empty? (SolverResult-unmet-goals sr)))
  (define contradiction? (SolverResult-contradiction sr))
  (printf
    "Solver ~a:\n  Facts: ~a\n\n  Met goals: ~a\n\n  Unmet goals: ~a\n\n"
    (if succeeded? "succeeded"
      (if contradiction? "found contradiction" "timed out"))
    (cond
      [succeeded?
        (string-join (map format-fact-v (get-step-by-step-solution sr)) "\n")]
      [contradiction?
        (string-join (map format-fact-v (get-step-by-step-contradiction sr)) "\n")]
      [else 
        (string-join (map format-fact
                          (SolverResult-facts sr)) "\n")])
    (string-join (map (lambda (g)
                        (format "~a (solves ~a)"
                                (format-fact (goal-solution g sr))
                                (format-term g)))
                      (SolverResult-met-goals sr)) ", ")
    (string-join (map format-term-debug (SolverResult-unmet-goals sr)) ", ")
    )
  (if (and succeeded? show-questions?)
    (generate-all-questions sr)
    #f))

; (run-example (list "x = y - 1" "y = 2x") (list "x = ?" "y = ?"))
; (run-example (list "x = 2 + 2" "y = x") (list "x = ?" "y = ?"))
; (run-example (list "2x = 4") (list "x = ?"))
; (run-example (list "x + 1 = 5") (list "x = ?"))
; (run-example (list "x = 1 + 2 + 3") (list "x = ?"))
; (run-example (list "1 + x - 1 = 3") (list "x = ?"))
; (run-example (list "0x = 1") (list "x = ?"))
; (run-example (list "x = 6x + 16 - 6x") (list "x = ?"))
; (run-example (list "7x = 6x + 16") (list "x = ?"))
; (run-example (list "7x - 15 = 6x + 1") (list "x = ?"))
; (run-example (list "x + 1 = 4") (list "x = ?"))
; (run-example (list "x - 1 = 4") (list "x = ?"))
; (run-example (list "3x - 3 - 2x = 3") (list "x = ?"))
; (run-example (list "3 + (x + -3) = 12 + 5*7") (list "x = ?"))
; (run-example (list "x + 1 - 1 = 2") (list "x = ?"))
; (run-example (list "-1 + x + 1 = 1 + 2 + 3") (list "x = ?"))
; (run-example (list "10x - 9x = 10") (list "x = ?"))

(define (run-mcts-example facts-str goals-str)
  (define facts (map (compose assumption parse-term) facts-str))
  (define goals (map parse-term goals-str))
  (printf "Solving:\n~a\nWith goals ~a\n"
          (string-join (map format-fact facts) "\n")
          (string-join (map format-term goals) ", "))
  (define result (solve-problem-smc (Problem facts goals)
                                    d:equations
                                    inverse-term-size-value-function 
                                    20
                                    10))
  (define solution (MCTSResult-terminal result))
  (printf
    "Solver ~a (~a expanded nodes, ~a leafs):\n~a\n\n"
    (if solution "succeeded" "timed out")
    (apply + (map (lambda (n) (if (MCTSNode-is-leaf? n) 0 1)) (MCTSResult-nodes result)))
    (apply + (map (lambda (n) (if (MCTSNode-is-leaf? n) 1 0)) (MCTSResult-nodes result)))
    (if solution
        (string-join (map (lambda (f)
                            (format "~a [~a]"
                              (format-fact f)
                              (generate-step-description
                                (Fact-proof f) (MCTSNode-facts solution))))
                                (MCTSNode-facts solution)) "\n")
        "<no solution found>")))

(run-mcts-example (list "x = 1 + 2 + 3 + 4") (list "x = ?"))
(run-mcts-example (list "2x = 4") (list "x = ?"))
(run-mcts-example (list "x + 1 = 4") (list "x = ?"))

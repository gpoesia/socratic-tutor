; Tutor interaction with the user.
#lang racket

(require brag/support)
(require "terms.rkt")
(require "term-parser.rkt")
(require "solver.rkt")
(require "tactics.rkt")

; Creates a goal to solve for the given variable.
(define (solve-for variable)
  (Predicate 'Eq (list (Variable variable) AnyNumber)))

(define (print-indexed-fact fact index [prefix ""])
  (printf "(~a~a): ~a\n" prefix index (format-term fact)))

(define (print-facts facts [index 1] [prefix ""])
  (if (empty? facts) 
    #f
    (begin
      (print-indexed-fact (car facts) index prefix)
      (print-facts (cdr facts) (+ 1 index)))))

(define (print-goals goals)
  (print-facts goals 1 "G"))

(define (goal-met goal facts)
  (ormap (lambda (f) (goal-matches? goal f)) facts))

(define (all-goals-met goals facts)
  (andmap (lambda (g) (goal-met g facts)) goals))

(define (matches-any-goal f goals)
  (ormap (lambda (g) (if (goal-matches? g f) g #f)) goals))

(define (tutor-repl facts goals)
  (if (all-goals-met goals facts)
    (printf "You're done!\n")
    (begin 
      (printf ">>> ")
      (with-handlers
        ([exn:fail:parsing?
           (lambda (e) 
             (begin (printf "I couldn't parse that.\n")
                    (tutor-repl facts goals)))]
         [exn:fail:contract?
           (lambda (e) (printf "Goodbye!\n"))])
        (let*
          (
           ; Read user input
           [l (read-line)]
           ; Parse it as a fact f.
           [f (parse-term l)]
           ; Use solver to verify f.
           [sr (find-solution (list f) facts s:cycle 50)]
           ; Check whether we could verify it.
           [verified (empty? (SolverResult-unmet-goals sr))]
           ; If verified, check whether it matches any goal.
           [matched-goal (and verified (matches-any-goal f goals))]
           )
          (if verified
            (begin
              (printf "OK! Let's add that to what we know:\n")
              (print-indexed-fact f (+ 1 (length facts)))
              (and matched-goal
                   (printf "Great, this matches the goal ~a\n" (format-term matched-goal)))
              (tutor-repl (append facts (list f)) goals))
            (begin
              (printf "Hmm, I could not verify that. Try again?\n")
              (tutor-repl facts goals))
            ))))))

(define (tutor facts goals)
  (printf "Let's solve a math problem! Given:\n")
  (print-facts facts)
  (printf "You need to meet:\n")
  (print-goals goals)
  (tutor-repl facts goals))

(provide tutor solve-for)

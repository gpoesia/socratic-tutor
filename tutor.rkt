; Tutor interaction with the user.
#lang racket

(require brag/support)
(require "terms.rkt")
(require "facts.rkt")
(require "term-parser.rkt")
(require "solver.rkt")
(require "tactics.rkt")
(require "questions.rkt")

; Creates a goal to solve for the given variable.
(define (solve-for variable)
  (Predicate 'Eq (list (Variable variable) (AnyNumber))))

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
  (ormap (lambda (f) (goal-matches? goal (Fact-term f))) facts))

(define (all-goals-met goals facts)
  (andmap (lambda (g) (goal-met g facts)) goals))

(define (matches-any-goal f goals)
  (ormap (lambda (g) (if (goal-matches? g f) g #f)) goals))

(define (first-non-assumption facts)
  (findf (lambda (f) (not (is-assumption? f))) facts)) 

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
           (lambda (e) (printf "~a\nGoodbye!\n" e))])
        ; Read user input
        (let ([l (read-line)])
          (cond
            [(equal? l "help")
             (printf "Let me see...\n")
             (let* ([sr (find-solution goals facts s:equations
                                       (prune:keep-smallest-k 50) 20)]
                    [solved? (empty? (SolverResult-unmet-goals sr))]
                    [step-by-step (and solved? 
                                       (get-step-by-step-solution sr))]
                    [question (generate-leading-question
                                (Fact-proof (first-non-assumption step-by-step))
                                step-by-step)]
                    )
               (if solved?
                 (printf "~a\n" question)
                 (printf "I'm as lost here as you, sorry!\n"))
               (tutor-repl facts goals))]
            [else
              (let* (
                ; Parse it as a term t.
                [t (parse-term l)]
                ; Check if we can prove f is false.
                [sr (find-solution goals (append facts (list (assumption t))) s:all
                                   (prune:keep-smallest-k 50) 10)]
                [contradiction (SolverResult-contradiction sr)]
                ; If verified, check whether it matches any goal.
                [matched-goal (and (not contradiction) (matches-any-goal t goals))])
                (if (not contradiction)
                  (begin
                    (printf "OK! Let's add that to what we know:\n")
                    (print-indexed-fact t (+ 1 (length facts)))
                    (and matched-goal
                         (printf "Great, this matches the goal ~a\n" (format-term matched-goal)))
                    (tutor-repl (append facts (list (assumption t))) goals))
                  (let* ([step-by-step (get-step-by-step-contradiction sr)]
                         [question (generate-leading-question
                                     (Fact-proof (first-non-assumption step-by-step))
                                     step-by-step)])
                    (printf "Hmm... ~a\n" question)
                    (tutor-repl facts goals))))]))))))

(define (tutor facts goals)
  (printf "Let's solve a math problem! Given:\n")
  (print-facts facts)
  (printf "You need to meet:\n")
  (print-goals goals)
  (tutor-repl (map assumption facts) goals))

(provide
  tutor
  solve-for
  first-non-assumption
  )

; Example problem for the tutor.
#lang algebraic/racket/base

(require "term-parser.rkt")
(require "terms.rkt")
(require "fraction.rkt")
(require "facts.rkt")
(require "solver.rkt")

; Generate a problem automatically
; Problems are of the form A +/- B +/- C; B and C exist by chance
(define generated_problem  (Problem-initial-facts (generate-fraction-problem)))
(define fact (car generated_problem))
(define fact-term (Fact-term fact))
; (printf "fact-term: ~a \n" fact-term)
(printf "generated problem ~a\n" (format-term (Fact-term fact)))

; Generate a problem manually
(define manual_problem (list(parse-term "F (2/3)+ ( -2 / 4)")))
(printf "munaul problem:~a\n" manual_problem)

; list the tactics for the manual problem
(define next-steps (d:fraction (map assumption manual_problem)))
(printf "----tactics for manaul problem-----\n")
(for-each (lambda (arg)
              (printf "tactic: ~a\n" (format-term (Fact-term arg))))
            next-steps)

; # tactics = O(term-size^2).
; This is because of the following tactic:
; B. Try expanding every number into two factors; using factors in the `primes` list + set of number appearing in the expression

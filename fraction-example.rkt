; Example problem for the tutor.
#lang algebraic/racket/base


(require "term-parser.rkt")
(require "tutor.rkt")
(require "terms.rkt")
(require "fraction.rkt")
(require "facts.rkt")
(require "tactics.rkt")

(require "solver.rkt")
(require "sorting.rkt")

(define problem (list(parse-term "F (2*3*6)/(3*1) + 5/(3*1)")))
(printf "problem:~a\n" problem)

(define next-steps (d:fraction (map assumption problem)))
(printf "----tactics:")
(for-each (lambda (arg)
              (printf "tactic: ~a\n" (format-term (Fact-term arg))))
            next-steps)

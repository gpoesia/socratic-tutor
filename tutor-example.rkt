; Example problem for the tutor.
#lang racket

(require "facts.rkt")
(require "fact-parser.rkt")
(require "tutor.rkt")

(tutor
  (list (parse-fact "1 + 2 + 3 = -9x + 10x"))
  (list (solve-for "x")))

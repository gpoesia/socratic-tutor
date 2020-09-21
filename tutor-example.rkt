; Example problem for the tutor.
#lang racket

(require "term-parser.rkt")
(require "tutor.rkt")

(tutor
  (list (parse-term "1 + 2 + 3 = -9x + 10x"))
  (list (solve-for "x")))

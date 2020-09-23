; Example problem for the tutor.
#lang racket

(require "term-parser.rkt")
(require "tutor.rkt")

(tutor
  (list (parse-term "1 + 2 + 3 + 4*5 = -1 + x + 1"))
  (list (solve-for "x")))

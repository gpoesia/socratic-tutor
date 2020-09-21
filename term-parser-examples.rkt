#lang racket

(require "terms.rkt")
(require "term-parser.rkt")

(define (show-example s)
  (define t (parse-term s))
  (printf "~a [~a]\n" (format-term t) t))

(show-example "2x = 9")
(show-example "x + 1 = (9+x)(9 - x)")
(show-example "-1x = (1+2+3)/6")
(show-example "x = ?")

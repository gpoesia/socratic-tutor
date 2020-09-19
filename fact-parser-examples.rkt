#lang racket

(require "fact-parser.rkt")

(printf "~a\n" (parse-fact "2x = 9"))
(printf "~a\n" (parse-fact "x + 1 = (9+x)(9 - x)"))
(printf "~a\n" (parse-fact "-1x = (1+2+3)/6"))
(printf "~a\n" (parse-fact "x = ?"))

#lang racket

(require "learn.rkt")

(define round-number (make-parameter 1))
(define n-problems (make-parameter 1000))
(define depth (make-parameter 5))
(define use-value-function (make-parameter #f))
(define beam-width (make-parameter 10))

(command-line
  #:program "domain-learner"
  #:once-each
  [("-V" "--value-function")
   "Use learned value function server."
   (use-value-function #t)]
  [("-p" "--problems") p
   "Number of problems to solve successfully before stopping."
   (n-problems (string->number p))]
  [("-d" "--depth") d
   "Max search depth."
   (depth (string->number d))]
  [("-b" "--beam") b
   "Beam width for beam search."
   (beam-width (string->number b))]
  [("-n" "--round") n
   "Round number <n> (determines output file)"
   (round-number n)])

(run-equation-solver-round (n-problems) (depth) (beam-width) (round-number) (use-value-function))

#lang racket

(require "learn.rkt")

(define output-file (make-parameter "solver-output.json"))
(define n-problems (make-parameter 1000))
(define negatives (make-parameter 5))
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
  [("-n" "--negatives") n
   "Number of negative examples to extract."
   (negatives (string->number n))]
  [("-b" "--beam") b
   "Beam width for beam search."
   (beam-width (string->number b))]
  [("-o" "--output") o
   "Path to output file"
   (output-file o)])

(run-equation-solver-round (n-problems) (depth) (negatives) (beam-width)
                           (output-file) (use-value-function))

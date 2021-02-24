#lang racket

(require "learn.rkt")

(define output-file (make-parameter "solver-output.json"))
(define n-problems (make-parameter 1000))
(define negatives (make-parameter 5))
(define depth (make-parameter 5))
(define domain (make-parameter "equations"))
(define policy (make-parameter "random"))
(define beam-width (make-parameter 10))
(define server (make-parameter "http://127.0.0.1:9911/"))
(define max-threads (make-parameter 1))

(command-line
  #:program "domain-learner"
  #:once-each
  [("-S" "--server") s
   "URL of the server to access the neural value function."
   (server s)]
  [("-p" "--problems") p
   "Number of problems to solve successfully before stopping."
   (n-problems (string->number p))]
  [("-P" "--policy") P
   "Which search policy to use ('random', 'smallest' or 'neural')"
   (policy P)]
  [("-d" "--depth") d
   "Max search depth."
   (depth (string->number d))]
  [("-T" "--threads") T
   "Max number of threads to use."
   (max-threads (string->number T))]
  [("-D" "--domain") D
   "Domain ('equations' or 'ternary-addition')."
   (domain D)]
  [("-n" "--negatives") n
   "Number of negative examples to extract."
   (negatives (string->number n))]
  [("-b" "--beam") b
   "Beam width for beam search."
   (beam-width (string->number b))]
  [("-o" "--output") o
   "Path to output file"
   (output-file o)])

(run-solver-round (domain) (n-problems) (depth) (negatives) (beam-width)
                  (output-file) (policy)
                  (if (equal? (policy) "neural") (list (server)) empty)
                  (max-threads))

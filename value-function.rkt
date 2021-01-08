; Ranks facts by querying a value function server.

#lang racket

(require net/http-easy)
(require "facts.rkt")
(require "tactics.rkt")
(require "solver.rkt")

(define (rank-facts-value-function all-facts facts)
  (let* ([examples (map (lambda (f) (format-step-by-step
                                     (trace-solver-steps all-facts (list f))
                                     axiom->string))
                        facts)]
         [res (post "http://127.0.0.1:9911/" #:json examples)]
         [examples-with-score (map cons facts (response-json res))]
         [sorted-examples (sort examples-with-score (lambda (a b) (> (cdr a) (cdr b))))])
    (response-close! res)
    (map car sorted-examples)))

(provide rank-facts-value-function)

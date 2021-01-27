; Ranks facts by querying a value function model server.

#lang racket

(require net/http-easy)
(require "facts.rkt")
(require "tactics.rkt")
(require "solver.rkt")

(define (neural-value-function nodes)
  (let* ([examples (map (lambda (node) (format-step-by-step-terms
                                        (MCTSNode-facts node)))
                        nodes)]
         [res (post "http://127.0.0.1:9911/" #:json examples)]
         [scores (response-json res)])
    (response-close! res)
    scores))

(provide neural-value-function)

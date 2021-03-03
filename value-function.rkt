; Ranks facts by querying a value function model server.

#lang racket

(require net/http-easy)
(require "facts.rkt")
(require "terms.rkt")
(require "tactics.rkt")
(require "questions.rkt")
(require "solver.rkt")

(define (make-neural-value-function [address "http://127.0.0.1:9911/"])
  (lambda (nodes)
    (let* ([examples (map (lambda (node)
                            (hash 'state (format-term (Fact-term (list-ref (MCTSNode-facts node)
                                                                           (- 2 (length (MCTSNode-facts node))))))
                                  'action (generate-formal-step-description)))
                          nodes)]
           [res (post address #:json examples)]
           [scores (response-json res)])
     (response-close! res)
     scores)))

(provide make-neural-value-function)

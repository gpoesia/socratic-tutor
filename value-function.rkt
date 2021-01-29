; Ranks facts by querying a value function model server.

#lang racket

(require net/http-easy)
(require "facts.rkt")
(require "tactics.rkt")
(require "solver.rkt")

(define (make-neural-value-function [address "http://127.0.0.1:9911/"])
  (lambda (nodes)
    (let* ([examples (map (lambda (node) (format-step-by-step-terms (MCTSNode-facts node)))
                          nodes)]
          [res (post address #:json examples)]
          [scores (response-json res)])
     (response-close! res)
     scores)))

(provide make-neural-value-function)

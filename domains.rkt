#lang racket

(require "facts.rkt")
(require "generation.rkt")
(require "tactics.rkt")
(require "ternary.rkt")

(struct Domain (name generator verifier step) #:transparent)

(define EquationsDomain
  (Domain
   "equations"
   generate-problem
   fact-solves-goal?
   d:equations))

(define TernaryAdditionDomain
  (Domain
   "ternary-addition"
   generate-ternary-addition-problem
   is-ternary-number-simplified?
   d:ternary))

(define AllDomains (list EquationsDomain TernaryAdditionDomain))

(define (get-domain-by-name name)
  (or
   (findf (lambda (d) (eq? (Domain-name d) name)) AllDomains)
   (raise-user-error (format "Cannot find domain '~a'" name))))

(provide
 Domain-name
 Domain-generator
 Domain-verifier
 Domain-step
 EquationsDomain
 TernaryAdditionDomain
 AllDomains
 get-domain-by-name)

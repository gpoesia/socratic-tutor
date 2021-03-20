#lang racket

(require "solver.rkt")
(require "facts.rkt")
(require "generation.rkt")
(require "tactics.rkt")
(require "ternary.rkt")
(require "sorting.rkt")

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

(define SortingDomain
  (Domain
   "sorting"
   generate-sorting-problem
   is-sorting-list-sorted?
   d:sorting))

(define AllDomains
  (list
   EquationsDomain
   TernaryAdditionDomain
   SortingDomain
   ))

(define (get-domain-by-name name)
  (or
   (findf (lambda (d) (equal? (Domain-name d) name)) AllDomains)
   (raise-user-error (format "Cannot find domain '~a' ~a"
                             name
                             (map Domain-name AllDomains)))))

; Serialization utils.
; Returns a unique string representing each axiom.
(define (axiom->string a)
  (match a
    [(== 'assumption) "assumption"]
    [(== a:flip-equality) "a:flip-equality"]
    [(== a:commutativity) "a:commutativity"]
    [(== a:subtraction-commutativity) "a:subtraction-commutativity"]
    [(== a:subtraction-same) "a:subtraction-same"]
    [(== a:associativity) "a:associativity"]
    [(== a:binop-eval) "a:binop-eval"]
    [(== a:add-zero) "a:add-zero"]
    [(== a:mul-zero) "a:mul-zero"]
    [(== a:mul-one) "a:mul-one"]
    [(== a:distributivity) "a:distributivity"]
    [(== a:substitute-both-sides) "a:substitute-both-sides"]
    [(== a:op-both-sides) "a:op-both-sides"]
    [(== td:add-consecutive) "td:add-consecutive"]
    [(== td:swap) "td:swap"]
    [(== td:erase-zero) "td:erase-zero"]
    [(== sd:swap) "sd:swap"]
    [(== sd:reverse) "sd:reverse"]
    ))

; Inverts axiom->string.
(define (string->axiom s)
  (match s
    [(== "assumption") 'assumption]
    [(== "a:flip-equality") a:flip-equality]
    [(== "a:commutativity") a:commutativity]
    [(== "a:subtraction-commutativity") a:subtraction-commutativity]
    [(== "a:subtraction-same") a:subtraction-same]
    [(== "a:associativity") a:associativity]
    [(== "a:binop-eval") a:binop-eval]
    [(== "a:add-zero") a:add-zero]
    [(== "a:mul-zero") a:mul-zero]
    [(== "a:mul-one") a:mul-one]
    [(== "a:distributivity") a:distributivity]
    [(== "a:substitute-both-sides") a:substitute-both-sides]
    [(== "a:op-both-sides") a:op-both-sides]
    [(== "td:add-consecutive") td:add-consecutive]
    [(== "td:swap") td:swap]
    [(== "td:erase-zero") td:erase-zero]
    [(== "sd:swap") sd:swap]
    [(== "sd:reverse") sd:reverse]
    ))

(provide
 EquationsDomain
 TernaryAdditionDomain
 AllDomains
 get-domain-by-name
 axiom->string
 string->axiom)

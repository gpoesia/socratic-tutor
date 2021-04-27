#lang racket

(require "solver.rkt")
(require "facts.rkt")
(require "generation.rkt")
(require "equations.rkt")
(require "ternary.rkt")
(require "sorting.rkt")
(require "fraction.rkt")

; Definition of all implemented domains.
; Domains come in pairs: domain should be used for training and learning,
; and domain/test should be used for evaluation.
; For now, only equations-ct has any difference between train and test,
; but in the future we'll use the test domains to test for systematic
; generalization (i.e. problems requiring longer solutions, which can be
; obtained by tweaking the difficulty parameter in the generator).

(define EquationsDomain
  (Domain
   "equations"
   generate-problem
   fact-solves-goal?
   d:equations))

(define EquationsTestDomain
  (Domain
   "equations/test"
   generate-problem
   fact-solves-goal?
   d:equations))

; Equations domain where problems come from the Cognitive Tutor logs.
; We take the last 90 templates to be a 'test set', and the
; remaining to be the 'training set' (there are 290 in total).
(define EquationsCTDomain
  (Domain
   "equations-ct"
   (generator-from-templates (take-right cognitive-tutor-templates 90))
   fact-solves-goal?
   d:equations))

; Equations domain where problems come from the Cognitive Tutor logs.
(define EquationsCTTestDomain
  (Domain
   "equations-ct/test"
   (generator-from-templates (drop-right cognitive-tutor-templates 90))
   fact-solves-goal?
   d:equations))

(define TernaryAdditionDomain
  (Domain
   "ternary-addition"
   generate-ternary-addition-problem
   is-ternary-number-simplified?
   d:ternary))

(define TernaryAdditionTestDomain
  (Domain
   "ternary-addition/test"
   generate-ternary-addition-problem
   is-ternary-number-simplified?
   d:ternary))

(define SortingDomain
  (Domain
   "sorting"
   generate-sorting-problem
   is-sorting-list-sorted?
   d:sorting))

(define SortingTestDomain
  (Domain
   "sorting/test"
   generate-sorting-problem
   is-sorting-list-sorted?
   d:sorting))

(define FractionDomain
  (Domain
   "fraction"
   generate-fraction-problem
   is-fraction-simplified?
   d:fraction))

(define AllDomains
  (list
   EquationsDomain EquationsTestDomain
   EquationsCTDomain EquationsCTTestDomain
   TernaryAdditionDomain TernaryAdditionTestDomain
   SortingDomain SortingTestDomain
   FractionDomain
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
    [(== fd:cancel-common-factor) "fd:cancel-common-factor"]
    [(== fd:factorize) "fd:factorize"]
    [(== fd:merge-two-fractions) "fd:merge-two-fractions"]
    [(== fd:mul-scaling-factor) "fd:mul-scaling-factor"]
    [(== fd:binop-eval) "fd:binop-eval"]
    [(== fd:convert-into-fraction) "fd:convert-into-fraction"]
    [(== fd:commutativity) "fd:commutativity"]
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
    [(== "fd:cancel-common-factor") fd:cancel-common-factor]
    [(== "fd:factorize") fd:factorize]
    [(== "fd:merge-two-fractions") fd:merge-two-fractions]
    [(== "fd:mul-scaling-factor") fd:mul-scaling-factor]
    [(== "fd:binop-eval") fd:binop-eval]
    [(== "fd:convert-into-fraction") fd:convert-into-fraction]
    [(== "fd:commutativity") fd:commutativity]
    ))

(provide
 EquationsDomain
 TernaryAdditionDomain
 AllDomains
 get-domain-by-name
 axiom->string
 string->axiom)

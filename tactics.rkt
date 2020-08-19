; Axioms and tactics used by the solver.
#lang racket

(require racket/match)
(require "terms.rkt")
(require "facts.rkt")

; ==============================
; ========   Axioms  ===========
; ==============================
; Axioms are the building blocks of the solutions.

; Trivial axiom from which premises of the problem can be derived.
(define a:premise 'Premise)

; Flip an equality.
(define (a:flip-equality eq)
  (match eq
    [(Fact 'Eq (list t1 t2)) (Fact 'Eq (list t2 t1))]
    [_ #f]))

; Add a term to both sides of an equation.
(define (a:add-to-both-sides eq term)
  (match eq
    [(Fact 'Eq (list t1 t2))
     (Fact 'Eq (list (BinOp op+ t1 term) (BinOp op+ t2 term)))]
    [_ #f]))

; Use equality e1 := t1 = t2 to substitute all occurrences
; of the term t1 by the term t2 in equality e2.
(define (a:substitute-both-sides eq term)
  #f)

; Simplify both sides of an equation.
(define (a:simpl-both-sides eq)
  (match eq
    [(Fact 'Eq (list t1 t2))
     (Fact 'Eq (list (simpl-term t1) (simpl-term t2)))]
    [_ #f]))

; ==============================
; ========   Tactics ===========
; ==============================
; Tactics apply axioms to the currently known facts (old and new).

; Apply a:flip-equality to all new equalities.
(define (t:flip met-goals unmet-goals old-facts new-facts)
  (filter identity (map a:flip-equality new-facts)))

; Apply a:simpl-both-sides to all new equalities.
(define (t:simpl met-goals unmet-goals old-facts new-facts)
  (filter identity (map a:simpl-both-sides new-facts)))

; ==============================
; ======== Strategies ==========
; ==============================
; These guide the solver in applying tactics.

; Cycle between all known tactics.
(define (s:cycle old-facts last-facts unmet-goals strategy-state)
  (match (or strategy-state 0)
    [0 (values t:flip 1)]
    [1 (values t:simpl 0)]))

(provide
  a:premise
  s:cycle)

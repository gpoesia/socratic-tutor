; Simple structures for representing facts.
; For now, all facts are of type 'Eq  (meaning one term equals another),
; but keeping the type will make it easier to generalize to e.g. inequalities
; or other kinds of propositions.
#lang algebraic/racket/base

(require "terms.rkt")
(require racket/match)
(struct Fact (type terms) #:transparent)

(struct DerivedFact (fact tactic params))

(define term-matches? 
  (function*
    [(x x) #t]
    [(AnyNumber (Number x)) #t]
    [_ #f]
  ))

(define (goal-matches? a b)
  (if (and (Fact? a) (Fact? b))
    (and (eq? (Fact-type a) (Fact-type b))
         (andmap goal-matches? (Fact-terms a) (Fact-terms b)))
    (term-matches? a b)))

(define (format-fact f)
  (match f
    [(Fact 'Eq (list t1 t2))
      (format "~a = ~a" (format-term t1) (format-term t2))]
    [_ (format "~a" f)]))

(provide
  Fact
  DerivedFact
  format-fact
  goal-matches?)

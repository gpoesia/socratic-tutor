; Representation for concrete facts (either assumptions or derivations).

#lang algebraic/racket/base

(require racket/string)
(require racket/format)
(require "terms.rkt")

(struct FactProof (axiom parameters))
(struct Fact (id term proof))

(define (new-fact term proof)
  (Fact (equal-hash-code (cons term proof)) term proof))

(define assumption-proof (FactProof 'Assumption '()))

(define (assumption term) (new-fact term assumption-proof))

(define (fact term proof) (new-fact term proof))

(define (format-fact-proof fp)
  (format "~a(~a)"
          (FactProof-axiom fp)
          (string-join (map ~s (FactProof-parameters fp)))))

(define (format-fact f [proof? #f])
  (if proof?
    (format "~a [~a]"
            (format-term (Fact-term f))
            (format-fact-proof (Fact-proof f)))
    (format-term (Fact-term f))))


(define (fact-terms-equal? f1 f2)
  (equal? (Fact-term f1) (Fact-term f2)))

(define (fact-solves-goal? f g)
  (goal-matches? g (Fact-term f)))

(provide
  FactProof FactProof-axiom FactProof-parameters
  Fact Fact-id Fact-term Fact-proof
  assumption
  fact
  format-fact
  format-fact-proof
  fact-terms-equal?
  fact-solves-goal?
  )

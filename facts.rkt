; Representation for concrete facts (either assumptions or derivations).

#lang algebraic/racket/base

(require racket/string)
(require racket/format)
(require "terms.rkt")

(struct FactProof (axiom parameters) #:transparent)

; Used to tag FactProof parameters that refer to previous facts.
(struct FactId (id) #:transparent)

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

(define (format-fact f [proof? #f] [id? #f])
  (format "~a~a~a"
    (if id? (format "(~a) " (Fact-id f)) "")
    (format-term (Fact-term f))
    (if proof? (format " [~a]"
                       (format-fact-proof (Fact-proof f)))
      "")))

(define (format-fact-v f) (format-fact f #t #t))

(define (fact-terms-equal? f1 f2)
  (equal? (Fact-term f1) (Fact-term f2)))

(define (fact-solves-goal? f g)
  (goal-matches? g (Fact-term f)))

(provide
  FactProof FactProof? FactProof-axiom FactProof-parameters
  Fact Fact? Fact-id Fact-term Fact-proof
  FactId FactId? FactId-id
  assumption
  fact
  format-fact
  format-fact-v
  format-fact-proof
  fact-terms-equal?
  fact-solves-goal?
  )

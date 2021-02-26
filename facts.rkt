; Representation for concrete facts (either assumptions or derivations).

#lang algebraic/racket/base

(require racket/string)
(require racket/list)
(require racket/function)
(require racket/format)
(require "terms.rkt")

(struct FactProof (axiom parameters) #:transparent)

; Used to tag FactProof parameters that refer to previous facts.
(struct FactId (id) #:transparent)

; Used to tag a FactProof argument that provides legible information about the
; proof, but that is not interpreted formally. This is used to facilitate the
; job of the value function learner.
(struct FactProofAnnotation (content) #:transparent)

(struct Fact (id term proof))

(define (new-fact term proof)
  (Fact (equal-hash-code term) term proof))

(define assumption-proof (FactProof 'assumption '()))

(define (assumption term) (new-fact term assumption-proof))

(define (fact term proof) (new-fact term proof))

(define (is-assumption? fact) (equal? (FactProof-axiom (Fact-proof fact))
                                      'assumption))

(define (fact-dependencies f)
  (filter identity (map (lambda (a) (if (FactId? a) (FactId-id a) #f))
                        (FactProof-parameters (Fact-proof f)))))

(define (format-proof-argument pa)
  (cond
    [(FactId? pa) (format "f:(~a)" (FactId-id pa))]
    [(FactProofAnnotation? pa) (format "a:(~a)" (FactProofAnnotation-content pa))]
    [(Term? pa) (format "t:(~a)" (format-term pa))]
    [#t (~s pa)]))

(define (format-fact-proof fp [format-axiom ~s])
  (format "~a [~a]"
          (format-axiom (FactProof-axiom fp))
          (string-join (map format-proof-argument (FactProof-parameters fp)) ",")))

(define (format-fact f [proof? #f] [id? #f])
  (format "~a~a~a"
    (if id? (format "(~a) " (Fact-id f)) "")
    (format-term (Fact-term f))
    (if proof? (format " :: ~a"
                       (format-fact-proof (Fact-proof f)))
      "")))

(define (format-fact-i f) (format-fact f #f #t))

(define (format-fact-v f) (format-fact f #t #t))

(define (fact-terms-equal? f1 f2)
  (equal? (Fact-term f1) (Fact-term f2)))

(define (fact-solves-goal? f g)
  (goal-matches? g (Fact-term f)))

; Formats a step-by-step solution, returning a string for each step.
(define (format-step-by-step facts format-axiom)
  (map (lambda (f) (format "(~a) ~a :: ~a"
                           (Fact-id f)
                           (format-term (Fact-term f))
                           (format-fact-proof (Fact-proof f) format-axiom)))
       facts))

; Formats a step-by-step solution with just the terms in each step, not the proof.
(define (format-step-by-step-terms facts)
  (map (lambda (f) (format-term (Fact-term f))) facts))

(define (format-step-by-step-terms-tex facts)
  (map (lambda (f) (format-term-tex (Fact-term f))) facts))

(define (sort-facts-by-size facts)
  (sort (shuffle facts)
        (lambda (a b) (< (term-size (Fact-term a))
                         (term-size (Fact-term b))))))

(provide
  FactProof FactProof? FactProof-axiom FactProof-parameters
  Fact Fact? Fact-id Fact-term Fact-proof
  FactId FactId? FactId-id
  FactProofAnnotation FactProofAnnotation? FactProofAnnotation-content
  assumption
  is-assumption?
  fact
  fact-dependencies
  format-fact
  format-fact-i
  format-fact-v
  format-step-by-step
  format-step-by-step-terms
  format-step-by-step-terms-tex
  fact-terms-equal?
  fact-solves-goal?
  sort-facts-by-size
  )

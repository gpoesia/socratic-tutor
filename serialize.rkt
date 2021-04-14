; Serializes terms and facts to/from JSON.

#lang algebraic/racket/base

(require "terms.rkt")
(require "term-parser.rkt")
(require "equations.rkt")
(require "domains.rkt")
(require "facts.rkt")
(require "solver.rkt")
(require json)

(define (to-json object out)
  (write-json (to-jsexpr object) out))

(define (to-json-string object)
  (jsexpr->string (to-jsexpr object)))

(define (to-jsexpr obj)
  (cond
    [(Term? obj)
      (hash 'type "Term" 'value (format-term obj))]
    [(FactId? obj)
      (hash 'type "FactId" 'value (FactId-id obj))]
    [(FactProofAnnotation? obj)
      (hash 'type "FactProofAnnotation" 'content (FactProofAnnotation-content obj))]
    [(FactProof? obj)
      (hash 'type "FactProof"
            'axiom (axiom->string (FactProof-axiom obj))
            'parameters (map to-jsexpr (FactProof-parameters obj)))]
    [(Fact? obj)
     (hash 'type "Fact"
           'id (to-jsexpr (Fact-id obj))
           'term (to-jsexpr (Fact-term obj))
           'proof (to-jsexpr (Fact-proof obj)))]
    [(SolverResult? obj)
     (hash 'type "SolverResult"
           'facts (to-jsexpr
                    (map to-jsexpr (SolverResult-facts obj)))
           'met-goals (to-jsexpr
                        (map to-jsexpr (SolverResult-met-goals obj)))
           'unmet-goals (to-jsexpr
                          (map to-jsexpr (SolverResult-unmet-goals obj)))
           'contradiction (to-jsexpr (SolverResult-contradiction obj))
           )]
    [(Operator? obj)
     (hash 'type "Operator" 'operator (op->string obj))]
    [(Problem? obj)
     (hash 'type "Problem"
           'initial-facts (to-jsexpr (Problem-initial-facts obj))
           'goals (to-jsexpr (Problem-goals obj)))]
    [(list? obj) (map to-jsexpr obj)]
    [(hash? obj)
     (make-hash
       (hash-map obj (lambda (k v) (cons k (to-jsexpr v)))))]
    [else obj]
    ))

(define (from-json port)
  (from-jsexpr (read-json port)))

(define (obj-type obj type)
  (and (hash? obj) (equal? (hash-ref obj 'type) type)))

(define (from-jsexpr obj)
  (cond
    [(obj-type obj "Term") (parse-term (hash-ref obj 'value))]
    [(obj-type obj "FactId") (FactId (hash-ref obj 'value))]
    [(obj-type obj "FactProofAnnotation") (FactProofAnnotation (hash-ref obj 'content))]
    [(obj-type obj "Operator") (string->op (hash-ref obj 'operator))]
    [(obj-type obj "FactProof")
     (FactProof
       (string->axiom (hash-ref obj 'axiom))
       (map from-jsexpr (hash-ref obj 'parameters)))]
    [(obj-type obj "Fact")
     (Fact
       (from-jsexpr (hash-ref obj 'id))
       (from-jsexpr (hash-ref obj 'term))
       (from-jsexpr (hash-ref obj 'proof)))]
    [(obj-type obj "SolverResult")
     (SolverResult
       (map from-jsexpr (hash-ref obj 'facts))
       (map from-jsexpr (hash-ref obj 'met-goals))
       (map from-jsexpr (hash-ref obj 'unmet-goals))
       (from-jsexpr (hash-ref obj 'contradiction)))]
    [(list? obj) (map from-jsexpr obj)]
    [(hash? obj)
     (make-hash
       (hash-map obj (lambda (k v) (cons k (from-jsexpr v)))))]
    [else obj]))

(define (from-json-string s)
  (from-jsexpr (string->jsexpr s)))

(provide
  to-json
  to-json-string
  to-jsexpr
  from-json
  from-json-string
  from-jsexpr)

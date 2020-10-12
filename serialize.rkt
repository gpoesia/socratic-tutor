; Serializes terms and facts to/from JSON.

#lang algebraic/racket/base

(require "terms.rkt")
(require "tactics.rkt")
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
    [(FactProof? obj)
      (hash 'type "FactProof"
            'axiom (axiom->string (FactProof-axiom obj))
            'parameters (map to-jsexpr (FactProof-parameters obj)))]
    [(Fact? obj)
     (hash 'type "Fact"
           'term (to-jsexpr (Fact-term obj))
           'proof (to-jsexpr (Fact-proof obj)))]
    [(SolverResult? obj)
     (hash 'type "SolverResult"
           'facts (to-jsexpr
                    (map to-jsexpr (SolverResult-facts obj))))]
    [(SolverResult? obj)
     (hash 'type "SolverResult"
           'facts (to-jsexpr
                    (map to-jsexpr (SolverResult-facts obj)))
           'met-goals (to-jsexpr
                        (map to-jsexpr (SolverResult-met-goals obj)))
           'unmet-goals (to-jsexpr
                          (map to-jsexpr (SolverResult-unmet-goals obj)))
           )]
    [(Operator? obj)
     (hash 'type "Operator" 'operator (op->string obj))]
    [else obj]
    ))

(provide
  to-json
  to-json-string
  to-jsexpr)

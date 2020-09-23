; Axioms and tactics used by the solver.
#lang algebraic/racket/base

(require algebraic/data)
(require algebraic/function)
(require algebraic/racket/base/forms)
(require racket/function)
(require racket/match)
(require "terms.rkt")
(require "debug.rkt")

; ==============================
; ========   Axioms  ===========
; ==============================
; Axioms are the building blocks of the solutions.

; Trivial axiom from which premises of the problem can be derived.
(define a:premise 'Premise)

; Flip an equality.
(define a:flip-equality
  (function
    [(Predicate 'Eq `(,t1 ,t2)) (Predicate 'Eq (list t2 t1))]
    [t #f]))

; Add a term to both sides of an equation.
(define (a:add-to-both-sides eq term)
  ((function
    [(Predicate 'Eq `(,t1 ,t2))
     (Predicate 'Eq ((BinOp op+ t1 term) (BinOp op+ t2 term)))]
    [t #f]) eq))

; Use equality e1 := t1 = t2 to substitute occurrences
; of the term t1 by the term t2 in equality e2.
(define (a:substitute-both-sides eq term)
  #f)

; Commutes both sides of a binary operation.
(define a:commutativity?
  (function
    [(BinOp op _ _) (is-commutative? op)]
    [_ #f]))

(define a:commutativity
  (phi (BinOp op l r) (BinOp op r l)))

; Rearranges an associative operation.
(define a:associativity?
  (function
    [(BinOp op a (BinOp op b c)) (is-associative? op)]
    [(BinOp op (BinOp op a b) c) (is-associative? op)]
    [_ #f]))

(define a:associativity
  (function
    [(BinOp op a (BinOp op b c)) (BinOp op (BinOp op a b) c)]
    [(BinOp op (BinOp op a b) c) (BinOp op a (BinOp op b c))]
    [_ #f]))

; Evaluates a binary operation on numbers.
(define a:binop-eval?
  (function
    [(BinOp op (Number n1) (Number n2)) #t]
    [_ #f]))

(define a:binop-eval
  (phi (BinOp op (Number n1) (Number n2))
       (Number (compute-bin-op op n1 n2))))

; Simplifies addition to zero.
(define a:add-zero?
  (function
    [(BinOp (op #:if (eq? op op+)) (Number 0) t) #t]
    [(BinOp (op #:if (eq? op op+)) t (Number 0)) #t]
    [t #f]))

(define a:add-zero
  (function
    [(BinOp (op #:if (eq? op op+)) (Number 0) t) t]
    [(BinOp (op #:if (eq? op op+)) t (Number 0)) t]
    [t #f]))

; Simplifies multiplication by zero.
(define a:mul-zero?
  (function
    [(BinOp (op #:if (eq? op op*)) (Number 0) t) #t]
    [(BinOp (op #:if (eq? op op*)) t (Number 0)) #t]
    [t #f]))

(define a:mul-zero
  (function
    [(BinOp (op #:if (eq? op op*)) (Number 0) t) (Number 0)]
    [(BinOp (op #:if (eq? op op*)) t (Number 0)) (Number 0)]
    [t #f]))

; Simplifies multiplication by one.
(define a:mul-one?
  (function
    [(BinOp (op #:if (eq? op op*)) (Number 1) t) #t]
    [(BinOp (op #:if (eq? op op*)) t (Number 1)) #t]
    [t #f]))

(define a:mul-one
  (function
    [(BinOp (op #:if (eq? op op*)) (Number 1) t) t]
    [(BinOp (op #:if (eq? op op*)) t (Number 1)) t]
    [t #f]))

; ==============================
; ========   Tactics ===========
; ==============================
; Tactics apply axioms to the new known facts, producing others.

; Apply a:flip-equality.
(define (t:flip met-goals unmet-goals old-facts new-facts)
  (filter identity (map a:flip-equality new-facts)))

; Meta-tactic that applies a simple term-level transform pair
; to all new facts, in all terms that satisfy the given predicate.
(define-syntax-rule (local-rewrite-tactic name predicate transform)
  (define (name mew-goals unmet-goals old-facts new-facts)
    (apply append
      (map (lambda (f)
             (let ([indices (filter-subterms f predicate)])
               (map (lambda (i)
                        (let ([rewritten (rewrite-subterm f transform i)])
                          (log-debug "~a rewrote ~a => ~a\n"
                                     #(name)
                                     (format-term f)
                                     (format-term rewritten))
                          rewritten))
                    indices)))
           new-facts))))

; Apply a:binop-eval.
(local-rewrite-tactic t:eval a:binop-eval? a:binop-eval)

; Apply a:associativity.
(local-rewrite-tactic t:associativity a:associativity? a:associativity)

; Apply a:commutativity.
(local-rewrite-tactic t:commutativity a:commutativity? a:commutativity)

; Apply a:add-zero.
(local-rewrite-tactic t:add-zero a:add-zero? a:add-zero)

; Apply a:mul-zero.
(local-rewrite-tactic t:mul-zero a:mul-zero? a:mul-zero)

; Apply a:mul-one.
(local-rewrite-tactic t:mul-one a:mul-one? a:mul-one)

; Applies all tactics.
(define (t:all met-goals unmet-goals old-facts new-facts)
  (apply append
         (map (lambda (t) (t met-goals unmet-goals old-facts new-facts))
              (list
                t:flip
                t:eval
                t:associativity
                t:commutativity
                t:add-zero
                t:mul-zero
                t:mul-one
                ))))

; ==============================
; ======== Strategies ==========
; ==============================
; These guide the solver in applying tactics.

; Always apply all known tactics.
(define (s:all old-facts last-facts unmet-goals strategy-state)
  (values t:all #f))

(provide
  a:premise
  s:all)

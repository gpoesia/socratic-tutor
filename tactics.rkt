; Axioms and tactics used by the solver.
#lang algebraic/racket/base

(require algebraic/data)
(require algebraic/function)
(require algebraic/racket/base/forms)
(require racket/list)
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
    [(BinOp op1 a (BinOp op2 b c)) (is-associative? op1 op2)]
    [(BinOp op1 (BinOp op2 a b) c) (is-associative? op2 op1)]
    [_ #f]))

(define a:associativity
  (function
    [(BinOp op1 a (BinOp op2 b c)) (BinOp op2 (BinOp op1 a b) c)]
    [(BinOp op1 (BinOp op2 a b) c) (BinOp op2 a (BinOp op1 b c))]
    [_ #f]))

; Evaluates a binary operation on numbers.
(define a:binop-eval?
  (function
    [(BinOp op (Number n1) (Number n2))
     #:if (not (and (eq? op op/) (= n2 0)))
     #t]
    [_ #f]))

(define a:binop-eval
  (phi (BinOp op (Number n1) (Number n2))
       (Number (compute-bin-op op n1 n2))))

; Simplifies adding/subtracting zero.
(define a:add-zero?
  (function
    [(BinOp (op #:if (eq? op op+)) (Number 0) t) #t]
    [(BinOp (op #:if (eq? op op+)) t (Number 0)) #t]
    [(BinOp (op #:if (eq? op op-)) t (Number 0)) #t]
    [t #f]))

(define a:add-zero
  (function
    [(BinOp (op #:if (eq? op op+)) (Number 0) t) t]
    [(BinOp (op #:if (eq? op op+)) t (Number 0)) t]
    [(BinOp (op #:if (eq? op op-)) t (Number 0)) t]
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

; Simplifies multiplication/division by one.
(define a:mul-one?
  (function
    [(BinOp (op #:if (eq? op op*)) (Number 1) t) #t]
    [(BinOp (op #:if (eq? op op*)) t (Number 1)) #t]
    [(BinOp (op #:if (eq? op op/)) t (Number 1)) #t]
    [t #f]))

(define a:mul-one
  (function
    [(BinOp (op #:if (eq? op op*)) (Number 1) t) t]
    [(BinOp (op #:if (eq? op op*)) t (Number 1)) t]
    [(BinOp (op #:if (eq? op op/)) t (Number 1)) t]
    [t #f]))

; Applies distributivity law.
(define a:distributivity?
  (function
    [(BinOp op1 a (BinOp op2 b c))
     #:if (is-distributive? op1 op2)
     #t]
    [(BinOp op2 (BinOp op1 a1 b) (BinOp op1 a2 c))
     #:if (and (is-distributive? op1 op2) (equal? a1 a2))
     #t]
    [(BinOp op2 (BinOp op1 a b1) (BinOp op1 c b2))
     #:if (and (is-distributive? op1 op2) (equal? b1 b2))
     #t]
    [t #f]))

(define a:distributivity
  (function
    [(BinOp op1 a (BinOp op2 b c))
     #:if (is-distributive? op1 op2)
     (BinOp op2 (BinOp op1 a b) (BinOp op1 a c))]
    [(BinOp (BinOp op1 a1 b) (BinOp op2 a2 c))
     #:if (and (is-distributive? op1 op2) (equal? a1 a2))
     (BinOp op1 a1 (BinOp op2 b c))]
    [(BinOp op2 (BinOp op1 a b1) (BinOp op1 c b2))
     #:if (and (is-distributive? op1 op2) (equal? b1 b2))
     (BinOp op1 (BinOp op2 a c) b2)]
    [t #f]))

; Applies an operation op with term t to both sides of an equation.
(define a:op-both-sides
  (function*
    [((Predicate 'Eq (a b)) t op)
     (Predicate 'Eq (list (BinOp op a t) (BinOp op b t)))]
    [(_ _ _) #f]))

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
  (define (name met-goals unmet-goals old-facts new-facts)
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

; Apply a:distributivity.
(local-rewrite-tactic t:distributivity a:distributivity? a:distributivity)

; Apply a:add-zero.
(local-rewrite-tactic t:add-zero a:add-zero? a:add-zero)

; Apply a:mul-zero.
(local-rewrite-tactic t:mul-zero a:mul-zero? a:mul-zero)

; Apply a:mul-one.
(local-rewrite-tactic t:mul-one a:mul-one? a:mul-one)

; Tactic that applies a:op-both-sides using terms from the equations.
(define produce-new-equalities
  (function
    [(Predicate 'Eq (a b)) #:as p
     (let ([all-terms (append (enumerate-subterms a) (enumerate-subterms b))])
       (map
         (lambda (t-op) (a:op-both-sides p (car t-op) (cadr t-op)))
         (cartesian-product all-terms (list op- op/ op+))))
     ]
    [t (list)]))

(define (t:apply-op-both-sides met-goals unmet-goals old-facts new-facts)
  (apply append (map produce-new-equalities new-facts)))

; Applies all tactics.
(define (t:all met-goals unmet-goals old-facts new-facts)
  (apply append
         (map (lambda (t) (t met-goals unmet-goals old-facts new-facts))
              (list
                ; t:flip
                t:eval
                t:associativity
                t:commutativity
                t:distributivity
                t:add-zero
                t:mul-zero
                t:mul-one
                t:apply-op-both-sides
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
